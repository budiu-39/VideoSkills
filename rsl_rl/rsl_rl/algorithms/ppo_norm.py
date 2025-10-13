from rsl_rl.algorithms import PPO
from rsl_rl.utils.running_mean_std import RunningMeanStd
import torch, torch.nn as nn
import copy

# ★ 新增：AMP imports
from contextlib import nullcontext
from torch.cuda.amp import GradScaler


class PPONorm(PPO):
    def __init__(self, *args, num_obs, normalize_value=False, normalize_obs=False, **kwargs):
        # 这几个键本类自己处理，避免重复传到父类
        kwargs.pop('normalize_obs', None)
        kwargs.pop('normalize_value', None)
        kwargs.pop('num_obs', None)
        kwargs.pop('num_critic_obs', None)

        # ★ 新增：读取 AMP 开关（若算法配置传了 use_mixed_precision）
        self.use_mixed_precision = bool(kwargs.pop('use_mixed_precision', False))

        # 先调父类构造（优化器等在父类里建）
        super().__init__(*args, **kwargs)

        # ★ 新增：为 PPONorm 自己准备 GradScaler（父类没有也没关系）
        self.scaler = GradScaler(enabled=self.use_mixed_precision)

        self.normalize_value = normalize_value
        self.normalize_obs = normalize_obs
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.device)

        if self.normalize_obs:
            self.obs_mean_std = RunningMeanStd((num_obs,)).to(self.device)
            self.obs_mean_std_temp = None

    def set_train(self):
        self.actor_critic.train()
        if self.normalize_obs:
            self.obs_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def set_eval(self):
        self.actor_critic.eval()
        if self.normalize_obs:
            self.obs_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def compute_returns(self, last_critic_obs):
        if self.normalize_obs:
            last_critic_obs = self.obs_mean_std_temp(last_critic_obs)
        super().compute_returns(last_critic_obs)
        if self.normalize_value:
            self.value_mean_std.train()
            self.storage.returns = self.value_mean_std(self.storage.returns.view(-1, 1)).view_as(self.storage.returns)
            self.storage.values  = self.value_mean_std(self.storage.values.view(-1, 1)).view_as(self.storage.values)

    def act(self, obs, critic_obs):
        if self.normalize_obs:
            self.obs_mean_std.train()
            self.obs_mean_std(obs)

        if self.normalize_obs:
            obs = self.obs_mean_std_temp(obs)
            critic_obs = self.obs_mean_std_temp(critic_obs)

        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # AMP 上下文（A100 推荐 bfloat16）
        from torch.cuda.amp import autocast as cuda_autocast
        ctx_factory = (lambda: cuda_autocast(enabled=True, dtype=torch.float16)) \
            if self.use_mixed_precision else (lambda: nullcontext())

        for (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
             old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch) in generator:

            with ctx_factory():
                # 前向与 PPO 损失（原逻辑不变）
                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch,
                                                         hidden_states=hid_states_batch[1])

                if self.normalize_value:
                    self.value_mean_std.eval()
                    value_batch = self.value_mean_std(value_batch)
                    self.value_mean_std.train()

                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL 自适应学习率（原样）
                if self.desired_kl is not None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / (old_sigma_batch + 1e-8) + 1e-5) +
                            (old_sigma_batch.pow(2) + (old_mu_batch - mu_batch).pow(2)) /
                            (2.0 * (sigma_batch.pow(2) + 1e-8)) - 0.5, dim=-1)
                        kl_mean = torch.mean(kl)
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for g in self.optimizer.param_groups:
                            g['lr'] = self.learning_rate

                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # 反向/更新：AMP 与非 AMP 两条路径
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss

    def _refresh_temp_rms(self):
        if not self.normalize_obs:
            return
        self.obs_mean_std_temp = copy.deepcopy(self.obs_mean_std)
        self.obs_mean_std_temp.freeze()

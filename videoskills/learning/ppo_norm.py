from rsl_rl.algorithms import PPO              # 假设原 PPO 放在这里
from videoskills.utils.running_mean_std import RunningMeanStd
import torch, torch.nn as nn
import copy

class PPONorm(PPO):
    def __init__(self, *args, num_obs, normalize_value=False, normalize_obs=False,**kwargs):
        kwargs.pop('normalize_obs', None)  # remove normalize_obs from kwargs if it exists
        kwargs.pop('normalize_value', None)
        kwargs.pop('num_obs', None)
        kwargs.pop('num_critic_obs', None)
        super().__init__(*args, **kwargs)
        self.normalize_value = normalize_value
        self.normalize_obs = normalize_obs
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.device)

        if self.normalize_obs:
            self.obs_mean_std = RunningMeanStd((num_obs,)).to(self.device)
            self.obs_mean_std_temp = None

    def compute_returns(self, last_critic_obs):
        if self.normalize_obs:
            last_critic_obs = self.obs_mean_std_temp(last_critic_obs)

        super().compute_returns(last_critic_obs)
        if self.normalize_value:
            self.value_mean_std.train()
            # self.storage.returns = self.value_rms(self.storage.returns)
            # self.storage.values  = self.value_rms(self.storage.values)
            self.storage.returns = self.value_mean_std(self.storage.returns.view(-1, 1)).view_as(self.storage.returns)
            self.storage.values  = self.value_mean_std(self.storage.values.view(-1, 1)).view_as(self.storage.values)


    def act(self, obs, critic_obs):
        if self.normalize_obs:
            self.obs_mean_std.train()  # ensure in update mode
            self.obs_mean_std(obs)

        if self.normalize_obs:
            obs = self.obs_mean_std_temp(obs)
            critic_obs = self.obs_mean_std_temp(critic_obs)

        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        # if self.use_value_norm:
        #     self.value_rms.eval()

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

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

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                    torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                    2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
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
from rsl_rl.runners.runner_eval import OnPolicyRunnerEval
from rsl_rl.utils.amp_discriminator import AMPDiscriminator
from rsl_rl.utils.replay_buffer import ReplayBuffer
from collections import deque
import torch
import os
import time
import wandb
import statistics
from rsl_rl.utils.running_mean_std import RunningMeanStd

class OnPolicyRunnerAMP(OnPolicyRunnerEval):
    def __init__(self, env, train_cfg, log_dir, device):
        super().__init__(env, train_cfg, log_dir, device)

        amp_cfg = train_cfg["amp_config"]
        self.disc_batch = amp_cfg.get("disc_batch", 512)
        self.n_disc_updates = amp_cfg.get("disc_updates", 1)
        self.disc_coef = amp_cfg.get("reward_coef", 0.5)
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)

        # 判别器
        # TODO: 好像没有写AMP的保存
        self.amp_disc = AMPDiscriminator(
            state_dim=amp_cfg["state_dim"],
            hidden_dims=amp_cfg["hidden_dims"],
            lr=amp_cfg.get("lr", 3e-4),
            grad_penalty_coef=amp_cfg.get("grad_penalty_coef", 1.0),
            logit_l2_coef=amp_cfg.get("logit_l2_coef", 1e-5),
            weight_decay=amp_cfg.get("weight_decay", 1e-6),
            device=device,
        )

        # Replay Buffers
        buf_size = amp_cfg["dataset_cfg"].get("replay_buffer_size", 4096)
        demo_buf_size = amp_cfg["dataset_cfg"].get("demo_buffer_size", 4096)
        self.replay_buf = ReplayBuffer(buf_size, device)
        self.demo_buf = ReplayBuffer(demo_buf_size, device)

        if self.normalize_obs:
            self.amp_obs_mean_std = RunningMeanStd((amp_cfg["state_dim"],)).to(self.device)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        self.alg.set_train()  # switch to train mode (for dropout for example)
        self.amp_obs_mean_std.train()  # ensure in update mode

        ep_infos = []

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            self.alg._refresh_temp_rms()
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    critic_obs = privileged_obs.to(self.device) if privileged_obs is not None else obs

                    amp_obs_sim = infos["amp_state"].to(self.device)
                    amp_obs_sim = self.amp_obs_mean_std(amp_obs_sim)
                    amp_obs_demo = self.env.fetch_amp_obs_demo(self.env.num_envs)
                    amp_obs_demo =  self.amp_obs_mean_std(amp_obs_demo)

                    self.replay_buf.store({"state": amp_obs_sim})
                    self.demo_buf.store({"state": amp_obs_demo})

                    with torch.no_grad():
                        disc_rew = self.amp_disc.compute_reward(amp_obs_sim)
                        disc_rew = self.disc_coef * disc_rew.squeeze(-1)
                    env_rew = rewards + disc_rew

                    self.alg.process_env_step(env_rew, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        self.rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        self.lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                self.alg.compute_returns(obs)

            start = stop

            disc_loss = 0.0
            if (self.replay_buf.get_total_count() >= self.disc_batch and
                    self.demo_buf.get_total_count() >= self.disc_batch):
                fake = self.replay_buf.sample(self.disc_batch)["state"]
                real = self.demo_buf.sample(self.disc_batch)["state"]

                disc_info = self.amp_disc.train_step(fake, real, self.n_disc_updates)

            disc_reward_mean = disc_rew.mean().item()
            disc_reward_std = disc_rew.std().item()
            infos_out = self.env.extras if hasattr(self.env, "extras") else {}

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))


    def eval(self, motion_ids=None):
        """
        Evaluate the current policy for a given number of episodes.
        """
        self.amp_obs_mean_std.eval()
        super().eval(motion_ids=motion_ids)


    def log(self, locs, width=80, pad=35):
        it = locs['it']
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']

        # ========== 构建 wandb metrics ==========
        wandb_metrics = {
            "Loss/value_function": locs['mean_value_loss'],
            "Loss/surrogate": locs['mean_surrogate_loss'],
            "Loss/learning_rate": self.alg.learning_rate,
            "Perf/collection_time": locs['collection_time'],
            "Perf/learning_time": locs['learn_time'],
            "Perf/total_fps": self.num_steps_per_env * self.env.num_envs / (
                    locs['collection_time'] + locs['learn_time']),
            "Policy/mean_noise_std": self.alg.actor_critic.std.mean().item(),
        }

        # 训练指标
        if len(locs['rewbuffer']) > 0:
            mean_rew = statistics.mean(locs['rewbuffer'])
            mean_len = statistics.mean(locs['lenbuffer'])
            wandb_metrics.update({
                "Train/mean_reward": mean_rew,
                "Train/mean_episode_length": mean_len,
            })

        # imitation rewards
        for key in ['reward_pos', 'reward_rot', 'reward_vel', 'reward_ang_vel']:
            if key in locs['infos']:
                val = locs['infos'][key]
                wandb_metrics[f"Imitation/{key}"] = val.mean().item() if isinstance(val, torch.Tensor) else float(
                    np.mean(val))

        # imitation errors
        for key in ['pos_err', 'rot_err', 'vel_err', 'ang_vel_err']:
            if key in locs['infos']:
                val = locs['infos'][key]
                wandb_metrics[f"Imitation/{key}"] = val.mean().item() if isinstance(val, torch.Tensor) else float(
                    np.mean(val))

        # ep_infos
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                vals = [ep[key].item() if isinstance(ep[key], torch.Tensor) else ep[key] for ep in locs['ep_infos']]
                wandb_metrics[f"Episode/{key}"] = sum(vals) / len(vals)

        if 'disc_info' in locs:
            disc_info = locs['disc_info']
            wandb_metrics.update({
                "disc/loss": disc_info["loss"],
                "disc/agent_acc": disc_info["agent_acc"],
                "disc/demo_acc": disc_info["demo_acc"],
            })

        if wandb.run is not None:
            wandb.log(wandb_metrics, step=it)

        ep_info_str = ""
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                vals = [ep[key].item() if isinstance(ep[key], torch.Tensor) else ep[key] for ep in locs['ep_infos']]
                ep_mean = sum(vals) / len(vals)
                if key in ["rew_imitation", "rew_dof_force"]:
                    ep_info_str += f"  {key}: {ep_mean:.4f}"

        summary = f"[{self.cfg['run_name']} it {it:05d}]"
        if mean_rew is not None and mean_len is not None:
            summary += f" Reward: {mean_rew:.3f} | EpLen: {mean_len:.2f}"
        summary += f" | Collect: {locs['collection_time']:.2f}s  Learn: {locs['learn_time']:.2f}s |"
        summary += ep_info_str
        if 'disc_info' in locs:
            summary += f" rew_disc: {locs['disc_rew'].mean():.4f} |"
            summary += f"disc_agent_acc: {locs['disc_info']['agent_acc']:.3f} "
            summary += f"disc_demo_acc: {locs['disc_info']['demo_acc']:.3f} "
        print(summary)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'obs_rms_state_dict': self.alg.obs_mean_std.state_dict() if self.alg.normalize_obs else None,
            'value_rms_state_dict': self.alg.value_mean_std.state_dict() if self.alg.normalize_value else None,
            'iter': self.current_learning_iteration,
            'amp_obs_mean_std': self.amp_obs_mean_std.state_dict() if self.normalize_obs else None,
            'amp_disc_state_dict': self.amp_disc.state_dict(),
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']

        if self.alg.normalize_obs:
            self.alg.obs_mean_std.load_state_dict(loaded_dict["obs_rms_state_dict"])
            self.amp_obs_mean_std.load_state_dict(loaded_dict['amp_obs_mean_std'])
        if self.alg.normalize_value:
            self.alg.value_mean_std.load_state_dict(loaded_dict["value_rms_state_dict"])

        self.amp_disc.load_state_dict(loaded_dict['amp_disc_state_dict'])

        return loaded_dict['infos']
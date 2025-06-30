from rsl_rl.runners import OnPolicyRunner
from videoskills.learning.runner_eval import OnPolicyRunnerEval
from videoskills.learning.algorithms.amp_discriminator import AMPDiscriminator
from videoskills.learning.algorithms.replay_buffer import ReplayBuffer
from collections import deque
import torch
import os
import time
import statistics


class OnPolicyRunnerAMP(OnPolicyRunnerEval):
    def __init__(self, env, train_cfg, log_dir, device):
        super().__init__(env, train_cfg, log_dir, device)

        amp_cfg = train_cfg["amp_cfg"]
        self.disc_batch = amp_cfg.get("disc_batch", 512)
        self.n_disc_updates = amp_cfg.get("disc_updates", 1)
        self.disc_coef = amp_cfg.get("reward_coef", 0.5)

        # 判别器
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

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        obs = self.env.get_observations().to(self.device)
        critic_obs = obs.clone()
        self.alg.actor_critic.train()

        ep_infos = []  # <- 重要！
        rewbuffer, lenbuffer = deque(maxlen=100), deque(maxlen=100)

        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            roll_start = time.time()

            for i in range(self.num_steps_per_env):
                act = self.alg.act(obs, critic_obs)
                obs, _, env_rew, done, infos = self.env.step(act)
                critic_obs = obs.to(self.device)
                obs = obs.to(self.device)

                # AMP reward
                # 思考了一下这样设计会导致 disc 总是能同时看到同一帧的 sim 和 ref，这样会导致 disc 过拟合（？）总之非常严格。
                amp_obs_sim = infos["amp_state"].to(self.device)
                amp_obs_demo = self.env.fetch_amp_obs_demo(self.env.num_envs)

                self.replay_buf.store({"state": amp_obs_sim})
                self.demo_buf.store({"state": amp_obs_demo})

                with torch.no_grad():
                    disc_rew = self.amp_disc.compute_reward(amp_obs_sim)

                env_rew = env_rew + self.disc_coef * disc_rew.squeeze(-1)
                self.alg.process_env_step(env_rew, done, infos)

                # 统计
                cur_reward_sum += env_rew.squeeze(-1)
                cur_episode_length += 1
                new_ids = (done > 0).nonzero(as_tuple=False)
                rewbuffer.extend(cur_reward_sum[new_ids].cpu().tolist())
                lenbuffer.extend(cur_episode_length[new_ids].cpu().tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0

                if 'episode' in infos:
                    ep_infos.append(infos['episode'])

            collection_time = time.time() - roll_start

            self.alg.compute_returns(obs)
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            learn_time = time.time() - roll_start - collection_time

            disc_loss = 0.0
            if (self.replay_buf.get_total_count() >= self.disc_batch and
                    self.demo_buf.get_total_count() >= self.disc_batch):
                fake = self.replay_buf.sample(self.disc_batch)["state"]
                real = self.demo_buf.sample(self.disc_batch)["state"]
                disc_loss = self.amp_disc.train_step(fake, real, self.n_disc_updates)

            disc_reward_mean = disc_rew.mean().item()
            disc_reward_std = disc_rew.std().item()
            infos_out = self.env.extras if hasattr(self.env, "extras") else {}

            # log
            locs = dict(
                it=it,
                num_learning_iterations=num_learning_iterations,
                collection_time=collection_time,
                learn_time=learn_time,
                ep_infos=ep_infos,
                rewbuffer=list(rewbuffer),
                lenbuffer=list(lenbuffer),
                mean_value_loss=mean_value_loss,
                mean_surrogate_loss=mean_surrogate_loss,
                infos=infos_out,
                disc_loss=disc_loss,
                disc_agent_acc=getattr(self.amp_disc, "last_agent_acc", 0.0),
                disc_demo_acc=getattr(self.amp_disc, "last_demo_acc", 0.0),
                disc_grad_penalty=getattr(self.amp_disc, "last_gp", 0.0),
                disc_reward_mean=disc_reward_mean,
                disc_reward_std=disc_reward_std,
            )
            self.log(locs)

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ''
        if locs.get('ep_infos'):
            for key in locs['ep_infos'][0]:
                vals = torch.cat([torch.atleast_1d(ep[key]).to(self.device) for ep in locs['ep_infos']])
                mean_val = vals.mean()
                self.writer.add_scalar(f'Episode/{key}', mean_val, locs['it'])
                ep_string += f"{f'Mean episode {key}:':>{pad}} {mean_val:.4f}\n"

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)

        w = self.writer
        it = locs['it']
        w.add_scalar('Loss/value_function',     locs['mean_value_loss'], it)
        w.add_scalar('Loss/surrogate',          locs['mean_surrogate_loss'], it)
        w.add_scalar('Loss/learning_rate',      self.alg.learning_rate, it)
        w.add_scalar('Policy/mean_noise_std',   mean_std.item(), it)
        w.add_scalar('Perf/total_fps',          fps, it)
        w.add_scalar('Perf/collection time',    locs['collection_time'], it)
        w.add_scalar('Perf/learning_time',      locs['learn_time'], it)

        if locs['rewbuffer']:
            flat_rewbuffer = [r for sublist in locs['rewbuffer'] for r in
                              (sublist if isinstance(sublist, list) else [sublist])]
            flat_lenbuffer = [l for sublist in locs['lenbuffer'] for l in
                              (sublist if isinstance(sublist, list) else [sublist])]

            mean_rew = statistics.mean(flat_rewbuffer)
            mean_len = statistics.mean(flat_lenbuffer)
            w.add_scalar('Train/mean_reward', mean_rew, it)
            w.add_scalar('Train/mean_episode_length', mean_len, it)
            w.add_scalar('Train/mean_reward/time', mean_rew, self.tot_time)
            w.add_scalar('Train/mean_episode_length/time', mean_len, self.tot_time)

        if 'infos' in locs:
            info = locs['infos']
            for tag in ("reward_pos", "reward_rot", "reward_vel", "reward_ang_vel",
                        "pos_err", "rot_err", "vel_err", "ang_vel_err"):
                if tag in info:
                    w.add_scalar(f'Imitation/{tag}', info[tag].mean().item(), it)

        if 'disc_loss' in locs:
            w.add_scalar('Disc/loss', locs['disc_loss'], it)
        if 'disc_agent_acc' in locs:
            w.add_scalar('Disc/agent_acc', locs['disc_agent_acc'], it)
        if 'disc_demo_acc' in locs:
            w.add_scalar('Disc/demo_acc', locs['disc_demo_acc'], it)
        if 'disc_grad_penalty' in locs:
            w.add_scalar('Disc/grad_penalty', locs['disc_grad_penalty'], it)
        if 'disc_reward_mean' in locs:
            w.add_scalar('AMP/reward_mean', locs['disc_reward_mean'], it)
            w.add_scalar('AMP/reward_std', locs['disc_reward_std'], it)

        header = f"\033[1m Learning iteration {it}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m"
        log_str = (f"{'#' * width}\n{header.center(width)}\n\n"
                   f"{'Computation:':>{pad}} {fps:>5} steps/s "
                   f"(collection {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
                   f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
                   f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n")
        if locs['rewbuffer']:
            log_str += (f"{'Mean reward:':>{pad}} {mean_rew:.2f}\n"
                        f"{'Mean episode length:':>{pad}} {mean_len:.2f}\n")
        if 'disc_loss' in locs:
            log_str += (f"{'Disc loss:':>{pad}} {locs['disc_loss']:.4f}\n"
                        f"{'Disc agent acc:':>{pad}} {locs['disc_agent_acc']:.3f}\n"
                        f"{'Disc demo  acc:':>{pad}} {locs['disc_demo_acc']:.3f}\n")

        if 'disc_reward_mean' in locs:
            log_str += (f"{'Disc reward mean:':>{pad}} {locs['disc_reward_mean']:.4f}\n"
                        f"{'Disc reward std:':>{pad}} {locs['disc_reward_std']:.4f}\n")

        log_str += ep_string
        log_str += f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
        print(log_str)

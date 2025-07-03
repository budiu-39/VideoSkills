from rsl_rl.runners import OnPolicyRunner
import numpy as np
import torch
from tqdm import tqdm
import wandb
import os
import joblib
import time
from rsl_rl.env import VecEnv
import statistics
from collections import defaultdict
from videoskills.utils.metrics import compute_metrics
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from videoskills.utils.running_mean_std import RunningMeanStd
import copy

from rsl_rl.algorithms import PPO
from videoskills.learning.ppo_norm import PPONorm
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

class OnPolicyRunnerEval(OnPolicyRunner):
    def __init__(self, env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(self.env.num_obs,
                                                       num_critic_obs,
                                                       self.env.num_actions,
                                                       **self.policy_cfg).to(self.device)
        # alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg = PPONorm(actor_critic, device=self.device,
                           normalize_value= self.cfg['normalize_value'], **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

        ###

        self.normalize_obs = train_cfg["runner"].get("normalize_obs", True)

        if self.normalize_obs:
            obs_shape = (self.env.num_obs,)  # assuming 1D vector input
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
            self.running_mean_std_temp = None

        self.normalize_value = train_cfg["runner"].get("normalize_value", False)
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.device)

        self.eval_output_path = os.path.join(log_dir,"eval_outputs")
        os.makedirs(self.eval_output_path, exist_ok=True)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            self._refresh_temp_rms()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    if self.normalize_obs:
                        obs_proc = self.running_mean_std_temp(obs)
                        critic_proc = self.running_mean_std_temp(critic_obs)
                    else:
                        obs_proc, critic_proc = obs, critic_obs

                    actions = self.alg.act(obs_proc, critic_proc)

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)

                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    critic_obs = privileged_obs.to(self.device) if privileged_obs is not None else obs

                    if self.normalize_obs:
                        self.running_mean_std.train()  # ensure in update mode
                        self.running_mean_std(obs)

                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop

                if self.normalize_obs:
                    critic_proc = self.running_mean_std_temp(critic_obs)
                else:
                    critic_proc = critic_obs.to(self.device)

                self.alg.compute_returns(critic_proc)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def eval(self, motion_ids=None):
        """Evaluate policy over multiple motions in parallel across environments."""
        self.alg.actor_critic.eval()
        self.env.eval_mode = True
        self.env.early_termination_distance = torch.tensor([0.5] * len(self.env.early_termination_distance)
                                                           , device=self.device) ** 2

        num_envs = self.env.num_envs
        motion_lib = self.env._motion_lib
        device = self.device

        if motion_ids is None:
            motion_ids = list(range(motion_lib.num_motions()))

        self._refresh_temp_rms()

        total_rewards = []
        success_flags = []
        reward_until_fail_list = []
        failed_keys = []
        global_metrics = defaultdict(list)
        metrics_success = defaultdict(list)
        pbar = tqdm(range(0, len(motion_ids), num_envs), desc="Evaluating motions", dynamic_ncols=True)
        for i in pbar:
            self.env.done_flags = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self.env.enable_data_recording() # enable recording during evaluation
            batch_ids = motion_ids[i: i + num_envs]
            batch_size = len(batch_ids)
            num_pad = num_envs - batch_size
            random_pad = np.random.choice(motion_ids, size=num_pad, replace=True).tolist()
            padded_ids = torch.tensor(batch_ids + random_pad, device=device)

            with torch.inference_mode():
                self.env.reset_with_motion_ids(padded_ids)
            self.env.gym.simulate(self.env.sim)
            with torch.inference_mode():# safe reset
                obs = self.env.reset_with_motion_ids(padded_ids)
            torch.cuda.empty_cache()

            cum_rewards = torch.zeros(num_envs, device=device)
            reward_until_fail = torch.zeros(num_envs, device=device)
            first_fail_recorded = torch.zeros(num_envs, dtype=torch.bool, device=device)

            episode_lengths = torch.zeros(num_envs, dtype=torch.int32, device=device)
            done_flags = torch.zeros(num_envs, dtype=torch.bool, device=device)

            motion_lengths = (motion_lib._motion_lengths[batch_ids]/self.env.dt).int()
            max_steps = motion_lengths.max().item()
            done_flags[batch_size:] = True

            for step in range(max_steps):  # max(range) = length + 1, therefore
                with torch.inference_mode():
                    if self.normalize_obs:
                        obs = self.running_mean_std_temp(obs)

                    action = self.alg.actor_critic.act_inference(obs)
                    obs, _, rewards, dones, extras = self.env.step(action)
                    rewards = rewards.squeeze()
                    rewards[done_flags] = 0.0
                    cum_rewards += rewards

                    episode_lengths += (~done_flags).int()
                    # dones[episode_lengths == motion_lib._motion_lengths[padded_ids]] = True

                    newly_done = dones.squeeze() & (~done_flags)
                    done_flags |= dones.squeeze()

                    for env_id in newly_done.nonzero(as_tuple=False).squeeze(-1).tolist():
                        if env_id >= batch_size:
                            continue
                        ep_len = episode_lengths[env_id].item()
                        expected_len = motion_lengths[env_id].item()
                        if ep_len < expected_len and not first_fail_recorded[env_id]:
                            reward_until_fail[env_id] = cum_rewards[env_id]
                            first_fail_recorded[env_id] = True

                    if done_flags[:batch_size].all():
                        break

                alive = ~done_flags[:batch_size]

                max_alive_ref_len = motion_lengths[alive].max().item()

                pbar.set_postfix(step=step, max_alive_step=max_alive_ref_len)

                if done_flags[:batch_size].all():
                    break

                self.env.done_flags = done_flags.clone().detach()

            for env_id in range(batch_size):
                ep_len = episode_lengths[env_id].item()
                expected_len = motion_lengths[env_id].item()
                success = ep_len >= expected_len
                success_flags.append(success)
                total_rewards.append(cum_rewards[env_id].item())

                if not success:
                    reward_until_fail_list.append(reward_until_fail[env_id].item())
                    failed_keys.append(motion_lib._motion_keys[batch_ids[env_id]])

            motion_id_to_data = defaultdict(list)
            for env_id in range(batch_size):
                motion_id = batch_ids[env_id]
                motion_id_to_data[motion_id].extend(self.env.recorded_data[env_id])
                self.env.recorded_data[env_id].clear()  # 清空缓存，避免污染下一个 batch

            pred_pos_all, gt_pos_all, pred_rot_all, gt_rot_all = [], [], [], []

            for motion_id in sorted(motion_id_to_data.keys()):
                frames = motion_id_to_data[motion_id]
                if len(frames) == 0:
                    continue  # skip empty
                pred_pos_all.append(np.stack([f["body_pos"] for f in frames], axis=0))  # (T, J, 3)
                gt_pos_all.append(np.stack([f["ref_body_pos"] for f in frames], axis=0))  # (T, J, 3)
                pred_rot_all.append(np.stack([f["body_rot"] for f in frames], axis=0))  # (T, J, 4)
                gt_rot_all.append(np.stack([f["ref_body_rot"] for f in frames], axis=0))  # (T, J, 4)

            # 3. 计算并打印指标
            batch_metrics, valid_mask = compute_metrics(pred_pos_all, gt_pos_all, pred_rot_all, gt_rot_all)
            motion_ids_sorted = sorted(motion_id_to_data.keys())

            for k, v in batch_metrics.items():
                global_metrics[k].extend(v)
                for j, valid in enumerate(valid_mask):
                    if not valid:
                        continue
                    key = motion_lib._motion_keys[motion_ids_sorted[j]]
                    if key not in failed_keys:
                        metrics_success[k].append(v.pop(0))

        print("\n[Eval] Overall Metrics:")
        for k, v in global_metrics.items():
            print(f"   {k}: {np.mean(v):.3f}")

        print("\n[Eval] Metrics for Successful Motions:")
        for k, v in metrics_success.items():
            print(f"   {k}: {np.mean(v):.3f}")

        num_success = sum(success_flags)
        num_total = len(success_flags)
        success_rate = num_success / num_total
        mean_rew = np.mean(total_rewards)

        # save failed keys and update soft sampling weight
        failed_key_path = os.path.join(self.eval_output_path, f"failed_keys_iter{self.current_learning_iteration}.pkl")
        joblib.dump(failed_keys, failed_key_path, compress=True)
        motion_lib.update_soft_sampling_weight(failed_keys)
        motion_sampling_state_path = os.path.join(self.log_dir, f"motion_sampling_state.pkl")
        motion_lib.export_sampling_state(motion_sampling_state_path)

        print(f"[Eval] Success rate: {success_rate:.2%}")
        print(f"[Eval] Mean reward across {len(motion_ids)} motions: {mean_rew:.2f}")
        print(f"[Eval] Avg. reward until failure (only failed): {np.mean(reward_until_fail_list):.2f}")

        self.env.disable_data_recording()
        self.env.early_termination_distance = torch.tensor(self.env.cfg.early_termination.distance, device=self.device) ** 2
        self.env.eval_mode = False
        with torch.inference_mode():
            self.env.reset()

        if wandb.run is not None:
            wandb.log({
                "Eval/mean_reward": mean_rew,
                "Eval/success_rate": success_rate,
                "Eval/reward_until_fail_mean_failed": np.mean(
                    reward_until_fail_list) if reward_until_fail_list else 0.0,
                "Eval/num_success": num_success,
                "Eval/num_total": num_total,
            })

            wandb_metric_dict = {}
            for k, v in global_metrics.items():
                if len(v) > 0:
                    wandb_metric_dict[f"Eval/{k}"] = np.mean(v).item()

            for k, v in metrics_success.items():
                if len(v) > 0:
                    wandb_metric_dict[f"Eval/{k}_success"] = np.mean(v).item()

            wandb.log(wandb_metric_dict, step=self.current_learning_iteration)

        return

    def _refresh_temp_rms(self):
        # 深拷贝后冻结，让 rollout 期间使用的均值方差保持不变
        if not self.normalize_obs:
            return
        self.running_mean_std_temp = copy.deepcopy(self.running_mean_std)
        self.running_mean_std_temp.freeze()

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

        # ========== wandb 记录 ==========
        if wandb.run is not None:
            wandb.log(wandb_metrics, step=it)

        ep_info_str = ""
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                vals = [ep[key].item() if isinstance(ep[key], torch.Tensor) else ep[key] for ep in locs['ep_infos']]
                ep_mean = sum(vals) / len(vals)
                wandb_metrics[f"Episode/{key}"] = ep_mean
                if key in ["rew_imitation", "rew_torques"]:
                    ep_info_str += f"  {key}: {ep_mean:.4f}"

        summary = f"[{self.cfg['run_name']} it {it:05d}]"
        if mean_rew is not None and mean_len is not None:
            summary += f" Reward: {mean_rew:.3f} | EpLen: {mean_len:.2f}"
        summary += f" | Collect: {locs['collection_time']:.2f}s  Learn: {locs['learn_time']:.2f}s |"
        summary += ep_info_str
        print(summary)
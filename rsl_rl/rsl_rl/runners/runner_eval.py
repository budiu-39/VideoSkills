from rsl_rl.runners import OnPolicyRunner
import numpy as np
import torch
from tqdm import tqdm
import wandb
import os
import joblib
import time
import statistics
from collections import defaultdict
from videoskills.utils.metrics import compute_metrics, physical_metrics
from collections import deque
# from torch.utils.tensorboard import SummaryWriter
from rsl_rl.algorithms.distill_dagger_z import build_student as build_actor_critic_with_z
from videoskills.utils.helpers import dict_to_class

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCriticMLP, ActorCriticRecurrent, ActorCritic_Attention, ActorCritic
from rsl_rl.backbone import BodyGraphBackbone
from rsl_rl.network import FiLMNetwork, MLPBackbone

class OnPolicyRunnerEval(OnPolicyRunner):
    def __init__(self, env,
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

        if 'use_z' not in self.policy_cfg:
            self.policy_cfg['use_z'] = False

        # 旧版
        if self.policy_cfg['use_z']:
            train_cfg=  dict_to_class(train_cfg)
            actor_critic = build_actor_critic_with_z(env, train_cfg, self.device)

        else:
            actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
            actor_critic = actor_critic_class(self.policy_cfg['actor_input_dim'],
                                                   self.policy_cfg['critic_input_dim'],
                                                   self.env.num_actions,
                                                   **self.policy_cfg).to(self.device)

        # TODO: 这里暂时注释掉
        self.alg_cfg['num_obs'] = self.env.num_obs
        self.alg_cfg['num_critic_obs'] = num_critic_obs

        # self.alg_cfg['num_obs'] = self.policy_cfg['actor_input_dim']
        # self.alg_cfg['num_critic_obs'] = self.policy_cfg['critic_input_dim']

        # alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg = PPO(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        init_storage = self.cfg.get("init_storage", True)  # 新增：默认不为 eval 分配storage
        if init_storage:
            # self.alg.init_storage(self.env.num_envs, self.num_steps_per_env,
            #                       [self.policy_cfg['actor_input_dim']],
            #                       [self.policy_cfg['critic_input_dim']], [self.env.num_actions])

            self.alg.init_storage(self.env.num_envs, self.num_steps_per_env,
                                  [self.env.num_obs], [num_critic_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

        self.eval_output_path = os.path.join(log_dir,"eval_outputs")
        self.rollouts_path = os.path.join(log_dir, "refine_results")
        self.rollouts_succeed_path = os.path.join(log_dir, "refine_results","succeed")
        self.rollouts_failed_path = os.path.join(log_dir, "refine_results", "failed")
        os.makedirs(self.eval_output_path, exist_ok=True)
        os.makedirs(self.rollouts_succeed_path, exist_ok=True)
        os.makedirs(self.rollouts_failed_path, exist_ok=True)

        self.rewbuffer = deque(maxlen=100)  # episdoe returns (对外可读)
        self.lenbuffer = deque(maxlen=100)  # episode lengths (可选)
        self.ETbuffer = deque(maxlen=20)  # 每迭代的 early termination rate (对外可读)
        self._recent_iterations_rewards = []  # 仅跨迭代临时累积，供 pop 使用
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.max_iterations = self.cfg.get("max_iterations", 100000)


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        # if self.log_dir is not None and self.writer is None:
        #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        self.alg.set_train()  # switch to train mode (for dropout for example)

        ep_infos = []
        # terbuffer = deque(maxlen = min(100, num_learning_iterations))

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            progress = it / self.max_iterations
            prob = max(0.0, 1.0 - progress) * 0.5
            self.env.visual_prob = prob
            start = time.time()
            early_termination_sum = 0
            dones_sum = 0
            # self.alg.actor_critic.refresh_temp_rms()
            for i in range(self.num_steps_per_env):
                with torch.no_grad():
                    if self.policy_cfg['use_z']:
                        obs = obs[:, :-self.env.num_actions]
                        critic_obs = obs
                        actions_z = self.alg.act(obs, critic_obs)
                        actions, _ = self.env.compute_z_action(actions_z, use_prior=False, sample=True)
                    else:
                        # student_obs = obs[:, :-645]
                        # # actions = self.alg.actor_critic.act_inference(obs_student)
                        # critic_obs = student_obs
                        actions = self.alg.act(obs, critic_obs)
                        # actions = self.alg.act(obs, critic_obs)
                obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                obs = obs.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                critic_obs = privileged_obs.to(self.device) if privileged_obs is not None else obs
                # critic_obs = obs[:, :-645]
                self.alg.process_env_step(rewards, dones, infos)
                early_termination_sum += sum(dones.cpu().numpy()) - sum(infos['time_outs'].cpu().numpy())
                dones_sum += sum(dones.cpu().numpy())

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    ended_rews = self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                    self.rewbuffer.extend(ended_rews)
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

                    # TODO: Here should implement a counter logic counting the sum of dones - ET and ET
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            ET_rate = early_termination_sum/(dones_sum + 1e-8)
            self.ETbuffer.append(float(ET_rate))
            self.alg.compute_returns(critic_obs)

            # === 打印 Early Termination 占比并清零 ===
            if getattr(self.env, "et_counter", None) is not None:
                et = self.env.et_counter
                total = et["total"]
                if total > 0:
                    print(f"[Iter {it}] Early Termination breakdown: "
                          f"robot={et['robot'] / total:.2%}, "
                          f"object={et['object'] / total:.2%}, "
                          f"ig={et['ig'] / total:.2%}, "
                          f"contact={et['contact'] / total:.2%}, total={total:.0f}")
                else:
                    print(f"[Iter {it}] No early termination this iteration.")
                # 清零计数器
                for k in et:
                    et[k] = 0

            start = stop
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self._recent_iterations_rewards.append(np.mean(self.rewbuffer))  # <--- 新增：本迭代收集
            if self.log_dir is not None:
                self.log(locals())

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        # self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def eval(self, motion_ids=None, log = True, rollout = False):
        """Evaluate policy over multiple motions in parallel across environments."""
        self.alg.set_eval()  # switch to eval mode (for dropout for example)
        state_init = self.env._state_init
        self.env._state_init = 'start'
        if hasattr(self.alg.actor_critic, "set_update_rms"):
            self.alg.actor_critic.set_update_rms(False)
        self.env.eval_mode = True
        self.env.early_termination_distance = torch.tensor(self.env.cfg.early_termination.eval_distance * len(self.env.early_termination_distance)
                                                           , device=self.device) ** 2

        num_envs = self.env.num_envs
        motion_lib = self.env._motion_lib
        device = self.device
        saved_count = 0

        if motion_ids is not None:
            eval_total = len(motion_ids)
        else:
            eval_total = motion_lib.num_motions()

        self.env.begin_eval(motion_ids)

        total_rewards = []
        success_flags = []
        failed_details = [] # (key, frame)
        reward_until_fail_list = []
        failed_keys = []
        success_keys = []
        global_metrics = defaultdict(list)
        metrics_success = defaultdict(list)
        pbar = tqdm(total=eval_total, desc="Evaluating motions", dynamic_ncols=True)
        seen = 0

        motion_lib = self.env._motion_lib
        while True:
            self.env.done_flags = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self.env.enable_data_recording() # enable recording during evaluation

            padded_ids, done_all = self.env.next_eval_batch_ids()  # [num_envs]
            batch_size = min(num_envs, eval_total - seen) if seen < eval_total else num_envs
            batch_ids = padded_ids[:batch_size].clone()
            batch_size = len(batch_ids)

            self.env.reset_with_motion_ids(padded_ids)
            self.env.gym.simulate(self.env.sim)
            obs = self.env.reset_with_motion_ids(padded_ids)

            cum_rewards = torch.zeros(num_envs, device=device)
            reward_until_fail = torch.zeros(num_envs, device=device)
            first_fail_recorded = torch.zeros(num_envs, dtype=torch.bool, device=device)

            episode_lengths = torch.zeros(num_envs, dtype=torch.int32, device=device)
            done_flags = torch.zeros(num_envs, dtype=torch.bool, device=device)

            motion_lengths = (motion_lib._motion_lengths[batch_ids]/self.env.dt).int()
            max_steps = motion_lengths.max().item()
            done_flags[batch_size:] = True

            for step in range(max_steps):  # max(range) = length + 1, therefore
                with torch.no_grad():
                    if self.policy_cfg['use_z']:
                        obs = obs[:, :-self.env.num_actions]
                        actions_z = self.alg.actor_critic.act_inference(obs)
                        actions, _ = self.env.compute_z_action(actions_z, use_prior=False, sample=False)
                    else:
                        # obs_front = obs[:, :1432]
                        # # action + task obs
                        # obs_back = obs[:, -645:]
                        # obs_student = torch.cat((obs_front, obs_back), dim=1)
                        # obs_student = obs[:, :-645]
                        actions = self.alg.actor_critic.act_inference(obs)
                obs, _, rewards, dones, extras = self.env.step(actions)
                rewards = rewards.squeeze()
                rewards[done_flags] = 0.0
                cum_rewards += rewards

                episode_lengths += (~done_flags).int()

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
                failed_rate = ((reward_until_fail[:batch_size] > 0).sum().float() / batch_size).item()
                pbar.set_postfix(step=step, max_alive_step=max_alive_ref_len, failed_rate = f"{failed_rate:.2%}")
                self.env.done_flags = done_flags.clone().detach()

            pbar.update(batch_size)
            seen = min(eval_total, seen + batch_size)

            for env_id in range(batch_size):
                ep_len = episode_lengths[env_id].item()
                expected_len = motion_lengths[env_id].item()
                success = ep_len >= expected_len
                success_flags.append(success)
                total_rewards.append(cum_rewards[env_id].item())

                key = motion_lib._motion_keys[batch_ids[env_id]]
                if success:
                    success_keys.append(key)
                else:
                    reward_until_fail_list.append(reward_until_fail[env_id].item())
                    failed_keys.append(key)
                    failed_details.append((key, ep_len))

            motion_id_to_data = defaultdict(list)
            pred_pos_all, gt_pos_all, pred_rot_all, gt_rot_all = [], [], [], []

            for env_id in range(batch_size):
                motion_id = batch_ids[env_id]
                motion_id_to_data[motion_id].extend(self.env.recorded_data[env_id])

            self.env.recorded_data = [[] for _ in range(num_envs)] # 清空缓存，避免污染下一个 batch

            motion_ids_sorted = sorted(motion_id_to_data.keys())
            for motion_id in motion_ids_sorted:
                frames = motion_id_to_data[motion_id]
                key = motion_lib._motion_keys[motion_id]
                if not frames:
                    continue

                # 是否有 GT
                use_gt = ("gt_body_pos" in frames[0]) and ("gt_body_rot" in frames[0])
                tgt_pos_key = "gt_body_pos" if use_gt else "ref_body_pos"
                tgt_rot_key = "gt_body_rot" if use_gt else "ref_body_rot"

                # 堆叠成 (T, J, 3)/(T, J, 4)
                pred_pos = np.stack([f["body_pos"] for f in frames], axis=0)  # (T, J, 3)
                pred_rot = np.stack([f["body_rot"] for f in frames], axis=0)  # (T, J, 4)
                pred_dof = np.stack([f["dof_pos"] for f in frames], axis=0)  # (T, D)
                tgt_pos = np.stack([f[tgt_pos_key] for f in frames], axis=0)  # (T, J, 3)
                tgt_rot = np.stack([f[tgt_rot_key] for f in frames], axis=0)  # (T, J, 4)
                tgt_dof = np.stack([f["ref_dof_pos"] for f in frames], axis=0)  # (T, D)
                action = np.stack([f["actions"] for f in frames], axis=0)  # (T, A)

                if rollout and key in success_keys:
                    rollout_length = len(frames)
                    rollout = {
                        "pred_pos": pred_pos[:rollout_length],
                        # "gt_pos": tgt_pos[:rollout_length],
                        "pred_rot": pred_rot[:rollout_length],
                        "action": action
                        # "gt_rot": tgt_rot[:rollout_length],
                        # "pred_dof_pos": pred_dof[:rollout_length],
                        # "gt_dof_pos": tgt_dof[:rollout_length],
                    }
                    out_path = os.path.join(self.rollouts_succeed_path, f"{key}.pkl")
                    joblib.dump(rollout, out_path, compress=True)
                    saved_count += 1

                pred_pos_all.append(pred_pos)
                pred_rot_all.append(pred_rot)
                gt_pos_all.append(tgt_pos)
                gt_rot_all.append(tgt_rot)

                # 释放内存
                motion_id_to_data[motion_id] = None

            # 3. 计算并打印指标
            batch_metrics, valid_mask = compute_metrics(pred_pos_all, gt_pos_all, pred_rot_all, gt_rot_all)
            physics_metrics = physical_metrics(gt_pos_all)

            for k, v in batch_metrics.items():
                global_metrics[k].extend(v)
                for j, valid in enumerate(valid_mask):
                    if not valid:
                        continue
                    key = motion_lib._motion_keys[motion_ids_sorted[j]]
                    if key not in failed_keys:
                        metrics_success[k].append(v.pop(0))

            for k, v in physics_metrics.items():
                global_metrics[k].extend(v)
                for j, valid in enumerate(valid_mask):
                    if not valid:
                        continue
                    key = motion_lib._motion_keys[motion_ids_sorted[j]]
                    if key not in failed_keys:
                        metrics_success[k].append(v.pop(0))

            if done_all:
                pbar.close()
                break

        print("\n[Eval] Overall Metrics:")
        for k, v in global_metrics.items():
            print(f"   {k}: {np.mean(v):.3f}")

        print("\n[Eval] Metrics for Successful Motions:")
        for k, v in metrics_success.items():
            print(f"   {k}: {np.mean(v):.3f}")

        if rollout:
            print(f"[Eval] Saved {saved_count} rollouts to {self.rollouts_succeed_path}.")

        num_success = sum(success_flags)
        num_total = len(success_flags)
        success_rate = num_success / num_total
        mean_rew = np.mean(total_rewards)

        # save failed keys and update soft sampling weight
        failed_keys_unique = sorted(set(failed_keys))
        failed_key_path = os.path.join(self.eval_output_path, f"failed_keys_iter{self.current_learning_iteration}.pkl")
        joblib.dump(failed_keys_unique, failed_key_path, compress=True)
        motion_lib.update_soft_sampling_weight(failed_keys_unique)
        motion_sampling_state_path = os.path.join(self.log_dir, f"motion_sampling_state.pkl")
        motion_lib.export_sampling_state(motion_sampling_state_path)

        print(f"[Eval] Success rate: {success_rate:.2%}")
        print(f"[Eval] Mean reward across {num_total} motions: {mean_rew:.2f}")
        print(f"[Eval] Avg. reward until failure (only failed): {np.mean(reward_until_fail_list):.2f}")

        self.env.disable_data_recording()
        self.env.early_termination_distance = torch.tensor(self.env.cfg.early_termination.distance, device=self.device) ** 2
        self.env.eval_mode = False
        self.env._state_init = state_init
        self.env.reset()

        if wandb.run is not None and log:
            wandb.log({
                "iteration": self.current_learning_iteration,
                "Eval/mean_reward": mean_rew,
                "Eval/success_rate": success_rate,
                "Eval/reward_until_fail_mean_failed": np.mean(
                    reward_until_fail_list) if reward_until_fail_list else 0.0,
                "Eval/num_success": num_success,
                "Eval/num_total": num_total,
            }, step=self.current_learning_iteration)

            wandb_metric_dict = {}
            for k, v in global_metrics.items():
                if len(v) > 0:
                    wandb_metric_dict[f"Eval/{k}"] = np.mean(v).item()

            for k, v in metrics_success.items():
                if len(v) > 0:
                    wandb_metric_dict[f"Eval/{k}_success"] = np.mean(v).item()

            wandb.log(wandb_metric_dict, step=self.current_learning_iteration)

        success_keys_unique = list(dict.fromkeys(success_keys))  # 保序去重
        result = {
            "mean_reward": mean_rew,
            "success_rate": success_rate,
            "reward_until_fail_mean_failed": np.mean(
                reward_until_fail_list) if reward_until_fail_list else 0.0,
            "num_success": num_success,
            "num_total": num_total,
            "success_keys": success_keys_unique,  # <<< 新增
            "failed_keys": failed_details,
        }
        return result

    def rollout(self, motion_ids=None):
        """
        Assume len(motion_ids) = M <= num_envs.
        Evenly fill ALL envs by repeating these M motions deterministically (no random padding).
        Report per-motion success rate and export ONE successful rollout per motion (skip if all failed).
        """

        self.alg.set_eval()
        self.env.eval_mode = True
        # 放宽早停阈值（与你的 eval 一致做法）
        self.env.early_termination_distance = torch.tensor(
            [0.5] * len(self.env.early_termination_distance), device=self.device
        ) ** 2

        num_envs = self.env.num_envs
        motion_lib = self.env._motion_lib
        device = self.device
        dt = self.env.dt

        # 选择要评估的 motions（默认全量）；保证 M <= num_envs（调用方已保证）
        if motion_ids is None:
            motion_ids = list(range(motion_lib.num_motions()))
        M = len(motion_ids)
        assert M <= num_envs, "refine() assumes M <= num_envs."

        # —— 均匀分配到 num_envs —— #
        q, r = divmod(num_envs, M)
        # indices: 长度 num_envs，值∈[0, M-1]，表示 env 使用第几个 motion_ids
        indices = []
        for i in range(M):
            reps = q + (1 if i < r else 0)
            indices.extend([i] * reps)
        assert len(indices) == num_envs

        env_motion_ids = [motion_ids[i] for i in indices]
        env_motion_ids_tensor = torch.tensor(env_motion_ids, device=device, dtype=torch.long)

        # 映射：env -> motion_key；统计：per-motion 尝试次数
        env_to_key = []
        per_motion_attempts = defaultdict(int)
        env_ids_by_key = defaultdict(list)
        for env_id, local_i in enumerate(indices):
            key = motion_lib._motion_keys[motion_ids[local_i]]
            env_to_key.append(key)
            per_motion_attempts[key] += 1
            env_ids_by_key[key].append(env_id)

            # —— 重置并开启记录 —— #
        self.env.done_flags = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.env.enable_data_recording()
        self.env.reset_with_motion_ids(env_motion_ids_tensor)
        self.env.gym.simulate(self.env.sim)
        obs = self.env.reset_with_motion_ids(env_motion_ids_tensor)
        torch.cuda.empty_cache()

        # 每 env 统计
        cum_rewards = torch.zeros(num_envs, device=device)
        episode_lengths = torch.zeros(num_envs, dtype=torch.int32, device=device)
        done_flags = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # 每 env 对应的参考长度（步数）
        env_motion_lengths = (motion_lib._motion_lengths[env_motion_ids_tensor.tolist()] / dt).int()
        max_steps = env_motion_lengths.max().item()

        # —— rollout —— #
        for _ in range(max_steps):
            with torch.no_grad():
                action = self.alg.actor_critic.act_inference(obs)
            obs, _, rewards, dones, extras = self.env.step(action)
            rewards = rewards.squeeze()
            rewards[done_flags] = 0.0
            cum_rewards += rewards
            episode_lengths += (~done_flags).int()
            done_flags |= dones.squeeze()
            if done_flags.all():
                break

        # —— 成功判定 & 每 motion 聚合 —— #
        per_motion_success = defaultdict(int)
        best_env_for_motion = {}  # key -> (env_id, reward)

        total_rewards_all_envs = []
        success_flags_all_envs = []

        for env_id in range(num_envs):
            key = env_to_key[env_id]
            ep_len = int(episode_lengths[env_id].item())
            exp_len = int(env_motion_lengths[env_id].item())
            success = (ep_len >= exp_len)
            success_flags_all_envs.append(bool(success))
            rew = float(cum_rewards[env_id].item())
            total_rewards_all_envs.append(rew)

            if success:
                per_motion_success[key] += 1
                cur = best_env_for_motion.get(key, None)
                if (cur is None) or (rew > cur[1]):
                    best_env_for_motion[key] = (env_id, rew)

        # —— 导出每个 motion 的一个成功 rollout（若该 motion 全失败则跳过） —— #
        saved_count = 0
        failed_keys = []
        for key, attempts in per_motion_attempts.items():
            succ = per_motion_success.get(key, 0)

            # 选择用于导出的 env_id：
            if succ > 0:
                # 成功则用奖励最高的那个
                env_id, _ = best_env_for_motion[key]
                save_dir = self.rollouts_succeed_path
                rollout_length = episode_lengths[env_id]
            else:
                # 全失败：选 episode 长度最长的那个 env
                candidates = env_ids_by_key.get(key, [])
                if not candidates:
                    failed_keys.append(key)
                    continue
                # 找出这些 env 的 episode_lengths，挑最长的
                best_env = max(candidates, key=lambda eid: int(episode_lengths[eid].item()))
                env_id = best_env
                save_dir = self.rollouts_failed_path
                failed_keys.append(key)  # 仍计入 failed_keys
                rollout_length = episode_lengths[env_id]

            frames = self.env.recorded_data[env_id]
            if not frames:
                continue

            pred_pos = np.stack([f["body_pos"] for f in frames], axis=0)
            gt_pos = np.stack([f["ref_body_pos"] for f in frames], axis=0)
            pred_rot = np.stack([f["body_rot"] for f in frames], axis=0)
            gt_rot = np.stack([f["ref_body_rot"] for f in frames], axis=0)
            pred_dof_pos = np.stack([f["dof_pos"] for f in frames], axis=0)
            gt_dof_pos = np.stack([f["ref_dof_pos"] for f in frames], axis=0)

            rollout = {
                "pred_pos": pred_pos[:rollout_length],
                "gt_pos": gt_pos[:rollout_length],
                "pred_rot": pred_rot[:rollout_length],
                "gt_rot": gt_rot[:rollout_length],
                "pred_dof_pos": pred_dof_pos[:rollout_length],
                "gt_dof_pos": gt_dof_pos[:rollout_length],
            }
            out_path = os.path.join(save_dir, f"{key}.pkl")
            joblib.dump(rollout, out_path, compress=True)
            saved_count += 1

        # —— 打印统计 —— #
        print("\n[Refine] Per-motion success rate:")
        per_motion_sr = {}
        for key in sorted(per_motion_attempts.keys()):
            a = per_motion_attempts[key]
            s = per_motion_success.get(key, 0)
            sr = s / max(1, a)
            per_motion_sr[key] = sr
            print(f"   {key}: {s}/{a} = {sr:.2%}")

        overall_success_rate = (np.mean(success_flags_all_envs) if success_flags_all_envs else 0.0)
        mean_rew = (np.mean(total_rewards_all_envs) if total_rewards_all_envs else 0.0)
        print(f"\n[Refine] Overall env-level success rate: {overall_success_rate:.2%}")
        print(f"[Refine] Mean reward (across all envs): {mean_rew:.2f}")
        print(
            f"[Refine] Saved {saved_count} successful motion rollouts. Skipped {len(failed_keys)} motions (no success).")

        # —— 失败 keys 更新采样权重（可选，与你 eval 保持一致） —— #
        if len(failed_keys) > 0:
            # failed_key_path = os.path.join(self.eval_output_path,
            #                                f"refine_failed_keys_iter{self.current_learning_iteration}.pkl")
            # joblib.dump(failed_keys, failed_key_path, compress=True)
            motion_lib.update_soft_sampling_weight(failed_keys)
            motion_sampling_state_path = os.path.join(self.log_dir, f"motion_sampling_state.pkl")
            motion_lib.export_sampling_state(motion_sampling_state_path)

        # —— 复位环境 —— #
        self.env.disable_data_recording()
        self.env.early_termination_distance = torch.tensor(self.env.cfg.early_termination.distance,
                                                           device=self.device) ** 2
        self.env.eval_mode = False
        self.env.reset()

        # —— wandb 记录（可选） —— #
        if wandb.run is not None:
            wandb.log({
                "Refine/overall_success_rate_env": overall_success_rate,
                "Refine/mean_reward_env": mean_rew,
                "Refine/saved_rollouts_count": saved_count,
                "Refine/failed_motions_count": len(failed_keys),
            }, step=self.current_learning_iteration)

        return {
            "overall_success_rate_env": overall_success_rate,
            "mean_reward_env": mean_rew,
            "per_motion_success_rate": per_motion_sr,
            "saved_rollouts": saved_count,
            "failed_motion_keys": failed_keys,
        }

    def log(self, locs, width=80, pad=35):
        it = locs['it']
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']

        # ------ 取出 step/episode 信息（安全）------
        infos = locs.get('infos', {})  # 最后一个step的infos
        ep_infos = locs.get('ep_infos', [])  # 本迭代收集到的所有episode汇总
        rb = locs.get('rewbuffer', getattr(self, 'rewbuffer', None))
        lb = locs.get('lenbuffer', getattr(self, 'lenbuffer', None))

        # ------ 构建 wandb metrics ------
        wandb_metrics = {
            "Loss/value_function": locs['mean_value_loss'],
            "Loss/surrogate": locs['mean_surrogate_loss'],
            "Loss/learning_rate": self.alg.learning_rate,
            "Perf/collection_time": locs['collection_time'],
            "Perf/learning_time": locs['learn_time'],
            "Perf/total_fps": self.num_steps_per_env * self.env.num_envs / (
                        locs['collection_time'] + locs['learn_time']),
            # "Policy/mean_noise_std": self.alg.actor_critic.std.mean().item(),
            "Imitation/ET_rate": locs['ET_rate']
        }

        # 训练指标
        if rb is not None and len(rb) > 0:
            mean_rew = statistics.mean(rb)
            mean_len = statistics.mean(lb)
            wandb_metrics.update({
                "Train/mean_reward": mean_rew,
                "Train/mean_episode_length": mean_len,
            })
        else:
            mean_rew, mean_len = None, None

        # imitation rewards（来自最后一个step的infos）
        if isinstance(infos, dict):
            for key, val in infos.items():
                if key.startswith("reward"):
                    wandb_metrics[f"Imitation/{key}"] = val.mean().item() if isinstance(val, torch.Tensor) else float(
                        np.mean(val))

            # imitation errors 可选
            for key in ['pos_err', 'rot_err', 'vel_err', 'ang_vel_err']:
                if key in infos:
                    val = infos[key]
                    wandb_metrics[f"Imitation/{key}"] = val.mean().item() if isinstance(val, torch.Tensor) else float(
                        np.mean(val))

        # ====== 关键：按迭代聚合并记录 episode 级指标（包含你的小项）======
        # 需要你的 env 在 reset_idx() 时把小项放入 extras["episode"]，键以 "rew_sub/..." 开头
        if self.log_dir is not None and isinstance(ep_infos, list) and len(ep_infos) > 0:
            merged = {}
            for ep in ep_infos:
                if not isinstance(ep, dict):
                    continue
                for k, v in ep.items():
                    if k.startswith("rew_") or k.startswith("rew_sub/"):
                        merged.setdefault(k, []).append(v.item() if torch.is_tensor(v) else float(v))

            if wandb.run is not None:  # ← 加保护
                for k, vals in merged.items():
                    if k.startswith("rew_sub/"):
                        wandb_key = f"Rewards/{k[len('rew_sub/'):]}"
                    elif k.startswith("rew_"):
                        wandb_key = f"Rewards/{k[len('rew_'):]}"
                    else:
                        wandb_key = f"Rewards/{k}"
                    wandb.log({wandb_key: float(np.mean(vals))}, step=it)

        # ====== 统一推送 wandb_metrics ======
        if wandb.run is not None:
            wandb.log(wandb_metrics, step=it)

        # 控制台摘要
        ep_info_str = ""
        if isinstance(ep_infos, list) and len(ep_infos) > 0:
            first = ep_infos[0]
            for key in first:
                vals = [ep[key].item() if isinstance(ep.get(key), torch.Tensor) else ep.get(key) for ep in ep_infos if
                        key in ep]
                if len(vals) == 0:
                    continue
                ep_mean = sum(vals) / len(vals)
                wandb_metrics[f"Episode/{key}"] = ep_mean
                if key in ["rew_imitation", "rew_dof_force", "rew_torques"]:
                    ep_info_str += f"  {key}: {ep_mean:.4f}"

        summary = f"[{self.cfg['run_name']} it {it:05d}]"
        if mean_rew is not None and mean_len is not None:
            summary += f" Reward: {mean_rew:.3f} | EpLen: {mean_len:.2f}"
        else:
            summary += " Reward: ---- | EpLen: ----"
        summary += f" | ET_rate: {locs['ET_rate']:.2f} |"
        summary += ep_info_str
        print(summary)

    def save(self, path=None, infos=None):
        if path is None:
            os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration))
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)


    def load(self, path=None, load_optimizer=True, load_iteration=True):
        if path is None:
            loaded_dict = torch.load(self.resume_path)
        else:
            loaded_dict = torch.load(path)
        if 'model_state_dict' in loaded_dict:
            self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        elif 'student_state_dict' in loaded_dict:
            self.alg.actor_critic.load_state_dict(loaded_dict['student_state_dict'])

        if load_optimizer:
            try:
                self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            except Exception as e:
                print("[runner_eval.load] skip optimizer due to error:", repr(e))
        if load_iteration:
            self.current_learning_iteration = loaded_dict['iter']

        # return loaded_dict['infos']

    def pop_recent_mean_rewards(self):
        """取出自上次调用以来新结束的所有 episode return，并清空临时缓存。"""
        out = self._recent_iterations_rewards
        self._recent_iterations_rewards = []
        return out

    def mean_et_rate(self, k=20):
        """返回最近 k 次迭代的 ET_rate 均值，若不足 k 次则用全部。"""
        if len(self.ETbuffer) == 0:
            return 1.0  # 没数据时保守认为 ET 高
        k = min(k, len(self.ETbuffer))
        return float(np.mean(list(self.ETbuffer)[-k:]))

    def pop_recent_ET_rate(self, k=20):
        """返回最近 k 次迭代的 ET_rate 均值，若不足 k 次则用全部。"""
        return list(self.ETbuffer)[-k:]

    def reset_motion_lib(self, motion_file):
        self.env._load_motion(motion_file)
        num_envs = self.env.num_envs
        init_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.env.reset_with_motion_ids(init_ids)


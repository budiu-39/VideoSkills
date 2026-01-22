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
            self.alg.init_storage(self.env.num_envs, self.num_steps_per_env,
                                  [self.env.num_obs], [num_critic_obs], [self.env.num_actions])
            if self.policy_cfg['use_z']:
                if self.policy_cfg.get('res_act', False):
                    self.alg.init_storage(self.env.num_envs, self.num_steps_per_env,
                    [self.env.num_obs], [num_critic_obs], [self.policy_cfg['z_dim'] + self.env.num_actions],)
                else:
                    self.alg.init_storage(self.env.num_envs, self.num_steps_per_env,
                    [self.env.num_obs], [num_critic_obs], [self.policy_cfg['z_dim']],)

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
                        # obs = obs[:, :-self.env.num_actions]
                        critic_obs = obs
                        actions_raw = self.alg.act(obs, critic_obs)
                        if self.policy_cfg['res_act']:
                            act_res = actions_raw[:, -self.env.num_actions:]
                            actions_z = actions_raw[:, :-self.env.num_actions]
                            actions, _ = self.env.compute_z_action(actions_z, use_prior=True, sample=False, act_res=act_res)
                        else:
                            actions, _ = self.env.compute_z_action(actions_raw, use_prior=True, sample=False)
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
            et_stats = {}
            total_finished_episodes = dones_sum + 1e-8
            if getattr(self.env, "et_counter", None) is not None:
                et = self.env.et_counter
                total_fails = et["total"]

                total_et_rate = total_fails / total_finished_episodes
                et_stats["ET_Rates/Total_Rate"] = total_et_rate
                et_stats["ET_Rates/Reason_Robot_Fall"] = et['robot'] / total_finished_episodes
                et_stats["ET_Rates/Reason_Object_Fail"] = et['object'] / total_finished_episodes
                et_stats["ET_Rates/Reason_IG_Fail"] = et['ig'] / total_finished_episodes
                et_stats["ET_Rates/Reason_Contact_Fail"] = et['contact'] / total_finished_episodes

                if total_fails > 0 and it % 10 == 0:
                    print(f"[Iter {it}] Early Termination breakdown: "
                          f"robot={et['robot'] / total_fails:.2%}, "
                          f"object={et['object'] / total_fails:.2%}, "
                          f"ig={et['ig'] / total_fails:.2%}, "
                          f"hand={et['cg_hand'] / total_fails:.2%}, "
                          f"body={et['cg_body'] / total_fails:.2%}, "
                          f"no_contact={et['cg_no_contact'] / total_fails:.2%}, "
                          f"penetration={et['penetration'] / total_fails:.2%},"
                          f"contact={et['contact'] / total_fails:.2%}, total={total_fails:.0f}")


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

    def eval(self, motion_ids=None, log = True, rollout = False, enable_early_termination=True):
        """Evaluate policy over multiple motions in parallel across environments."""
        self.alg.set_eval()  # switch to eval mode (for dropout for example)
        state_init = self.env._state_init
        self.env._state_init = 'start'
        if hasattr(self.alg.actor_critic, "set_update_rms"):
            self.alg.actor_critic.set_update_rms(False)
        self.env.eval_mode = True
        self.env.early_termination = enable_early_termination
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
                        obs = obs[:, :self.policy_cfg['proprioception_dim']+self.policy_cfg['task_dim']]
                        actions_raw = self.alg.actor_critic.act_inference(obs)
                        if self.policy_cfg['res_act']:
                            act_res = actions_raw[:, -self.env.num_actions:]
                            actions_z = actions_raw[:, :-self.env.num_actions]
                            actions, _ = self.env.compute_z_action(actions_z, use_prior=True, sample=False, act_res=act_res)
                        else:
                            actions, _ = self.env.compute_z_action(actions_raw, use_prior=True, sample=False)
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
                dones |= extras['early_termination_buf']
                # if dones.any():
                #     print('dones detect')
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

                if 'obj_pos' in frames[0]:
                    # Stack Data: (T, 3) / (T, 4)
                    pred_obj_pos = np.stack([f['obj_pos'] for f in frames], axis=0)
                    gt_obj_pos = np.stack([f['ref_obj_pos'] for f in frames], axis=0)
                    pred_obj_rot = np.stack([f['obj_rot'] for f in frames], axis=0)
                    gt_obj_rot = np.stack([f['ref_obj_rot'] for f in frames], axis=0)

                    # 1. Position Error (Euclidean Distance in meters)
                    # Shape: (T,) -> Scalar Mean
                    obj_pos_err = np.linalg.norm(pred_obj_pos - gt_obj_pos, axis=-1).mean()

                    # 2. Rotation Error (Angle difference in degrees)
                    # Formula: 2 * arccos(|<q1, q2>|)
                    # Dot product over last axis
                    quat_dot = np.abs(np.sum(pred_obj_rot * gt_obj_rot, axis=-1))
                    quat_dot = np.clip(quat_dot, -1.0, 1.0) # Numerical stability
                    obj_rot_err_rad = 2 * np.arccos(quat_dot)
                    obj_rot_err_deg = np.degrees(obj_rot_err_rad).mean()

                    # Add to Global Metrics (Tracked across all motions)
                    global_metrics['Object_Pos_Err'].append(obj_pos_err)
                    global_metrics['Object_Rot_Err'].append(obj_rot_err_deg)

                    # Add to Success-Only Metrics (if this motion was successful)
                    if key not in failed_keys:
                        metrics_success['Object_Pos_Err'].append(obj_pos_err)
                        metrics_success['Object_Rot_Err'].append(obj_rot_err_deg)

                if rollout and key in success_keys:
                    rollout_length = len(frames)
                    rollout = {
                        "pred_pos": pred_pos[:rollout_length],
                        # "gt_pos": tgt_pos[:rollout_length],
                        "pred_rot": pred_rot[:rollout_length],
                        "action": action,
                        # "gt_rot": tgt_rot[:rollout_length],
                        "pred_dof_pos": pred_dof[:rollout_length],
                        # "gt_dof_pos": tgt_dof[:rollout_length],
                        "pred_obj_pos": pred_obj_pos[:rollout_length] if 'obj_pos' in frames[0] else None,
                        "pred_obj_rot": pred_obj_rot[:rollout_length] if 'obj_rot' in frames[0] else None,
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
        self.env.early_termination = self.env.cfg.early_termination.enabled
        self.env.eval_mode = False
        self.env._state_init = state_init
        self.env.reset()

        if wandb.run is not None and log:
            # 1. 基础汇总指标 (Section: Eval_Standard)
            # 使用 Standard 作为前缀，便于一眼看到核心结果
            eval_summary = {
                "Eval_Standard/Success_Rate": success_rate,
                "Eval_Standard/Mean_Reward": mean_rew,
                "Eval_Standard/Reward_Until_Fail": np.mean(reward_until_fail_list) if reward_until_fail_list else 0.0,
                "Eval_Standard/Num_Success": num_success,
                "Eval_Standard/Num_Total": num_total,
                "iteration": self.current_learning_iteration  # 保持时序同步
            }

            # 2. 分类评估指标 (Section: Tracking vs Physical)
            # 定义物理指标关键字，用于自动分流
            physics_keywords = ['skating', 'skate', 'penetration', 'contact', 'floating', 'cg', 'ig', 'sdf']

            detail_metrics = {}

            # --- 处理 A: 全量轨迹指标 (_All) ---
            for k, v in global_metrics.items():
                if len(v) == 0: continue
                val = np.mean(v).item()
                # 判定是物理合理性指标还是动作模仿指标
                if "Object" in k:
                    prefix = "Eval_Object"  # 新增分类：物体误差
                elif any(kw in k.lower() for kw in physics_keywords):
                    prefix = "Eval_Physical"  # 物理指标
                else:
                    prefix = "Eval_Tracking"  # 追踪指标 (人体)

                detail_metrics[f"{prefix}/{k}_All"] = val

            # --- 处理 B: 仅成功轨迹的指标 (_SuccessOnly) ---
            # 这对于观察模型在稳定运行时的姿态质量非常有用
            for k, v in metrics_success.items():
                if len(v) == 0: continue
                val = np.mean(v).item()

                # === [修改点] 增加 Object 分类判断 ===
                if "Object" in k:
                    prefix = "Eval_Object"
                elif any(kw in k.lower() for kw in physics_keywords):
                    prefix = "Eval_Physical"
                else:
                    prefix = "Eval_Tracking"

                detail_metrics[f"{prefix}/{k}_success_only"] = val

            # 3. 一次性推送
            wandb.log(eval_summary, step=self.current_learning_iteration)
            wandb.log(detail_metrics, step=self.current_learning_iteration)

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

        # 初始化最终发送给 WandB 的字典
        wandb_dict = {}

        # 1. 损失与学习率 (Section: Loss)
        wandb_dict.update({
            "Loss/Value_Function": locs.get('mean_value_loss'),
            "Loss/Surrogate_PPO": locs.get('mean_surrogate_loss'),
            "Loss/Learning_Rate": self.alg.learning_rate,
        })

        # 2. 系统性能 (Section: System)
        wandb_dict.update({
            "System/FPS": (self.num_steps_per_env * self.env.num_envs) / (locs['collection_time'] + locs['learn_time']),
            "System/Early_Termination_Rate": locs.get('ET_rate', 0.0),
            "System/Time_Elapsed": self.tot_time,
        })

        # 3. 训练统计（按已完成的 Episode 平均）(Section: Train)
        mean_rew = None
        mean_len = None
        if len(self.rewbuffer) > 0:
            mean_rew = statistics.mean(self.rewbuffer)
            mean_len = statistics.mean(self.lenbuffer)
            wandb_dict["Train/Episode_Reward_Total"] = mean_rew
            wandb_dict["Train/Episode_Length"] = mean_len

        # 4. 解析单步内的实时奖励分量与误差 (Section: Reward_Terms & Errors)
        # 这部分主要从 env.step 的 infos 中获取
        infos = locs.get('infos', {})
        if isinstance(infos, dict):
            for key, val in infos.items():
                # 只处理 Tensor, Array, Number 或 List
                if not isinstance(val, (torch.Tensor, np.ndarray, float, int, list)):
                    continue

                # 统一计算均值：将 Bool/CUDA Tensor 转为 Float
                if isinstance(val, torch.Tensor):
                    v_mean = val.float().mean().item()
                else:
                    try:
                        v_mean = float(np.mean(val))
                    except:
                        continue

                # 分类命名，便于 WandB 侧边栏折叠
                if key.startswith("reward"):
                    # Section: Reward_Terms (奖励项：0~1得分)
                    clean_name = key.replace("reward_", "")
                    wandb_dict[f"Reward_Terms/{clean_name}"] = v_mean

                elif any(kw in key for kw in ['err', 'dist', 'delta', 'pos', 'rot', 'vel']):
                    # Section: Physical_Errors (物理误差：真实的米、度、速度等)
                    # 防止由于命名字典干扰，可以再过滤一下
                    if not key.startswith("reward"):
                        clean_name = key.replace("_err", "")
                        wandb_dict[f"Physical_Errors/{clean_name}"] = v_mean

        et_stats = locs.get('et_stats', {})
        if et_stats:
            wandb_dict.update(et_stats)

        # 5. 解析 Episode 结束时的累计子项 (Section: HOI_Details)
        # 这部分是从 reset_idx 时填入的 extras['episode'] 传过来的
        ep_infos = locs.get('ep_infos', [])
        ep_info_str = ""
        if len(ep_infos) > 0:
            merged_stats = defaultdict(list)
            for ep in ep_infos:
                for k, v in ep.items():
                    val = v.item() if torch.is_tensor(v) else float(v)
                    merged_stats[k].append(val)

            # 定义需要在控制台终端打印的关键指标
            core_summary_keys = ["humanoid", "obj", "ig", "cg", "imitation"]
            found_summary = {}

            for k, vals in merged_stats.items():
                avg = np.mean(vals)
                if k.startswith("rew_sub/"):
                    # 路径：Detailed_Subterms (例如穿模量、具体关节点分数)
                    wandb_dict[f"Detailed_HOI/{k[8:]}"] = avg
                elif k.startswith("rew_"):
                    # 路径：Reward_Summaries (模仿奖励汇总)
                    wandb_dict[f"Reward_Summaries/{k[4:]}"] = avg

                # 模糊匹配以生成终端摘要行
                for target in core_summary_keys:
                    if target in k.lower():
                        found_summary[target] = avg

            # 拼接控制台显示的摘要字符串
            for label, val in found_summary.items():
                ep_info_str += f"  {label}: {val:.3f}"

        # 6. 一次性推送至 WandB (防止短时间内多次调用造成的 UI 错位)
        if wandb.run is not None:
            wandb.log(wandb_dict, step=it)

        # 7. 控制台格式化输出
        summary = f"[{self.cfg['run_name']} it {it:05d}]"

        if mean_rew is not None:
            # 格式： Rew: [总回报] | Len: [总长度]
            summary += f" Rew: {mean_rew:,.1f} | Len: {mean_len:.1f}"
        else:
            summary += " Rew: ---- | Len: ----"

        # 增加早停率显示 (转为百分比)
        summary += f" | ET: {locs.get('ET_rate', 0.0):.2%}"

        # 增加核心子奖励摘要
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

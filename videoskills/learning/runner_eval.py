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



class OnPolicyRunnerEval(OnPolicyRunner):
    def __init__(self, env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        super().__init__(env, train_cfg, log_dir, device)
        self.eval_output_path = os.path.join(log_dir,"eval_outputs")
        os.makedirs(self.eval_output_path, exist_ok=True)

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

            # with torch.inference_mode():
            self.env.reset_with_motion_ids(padded_ids)
            self.env.gym.simulate(self.env.sim)    # safe reset
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
                with torch.no_grad():
                    action = self.alg.actor_critic.act_inference(obs.to(device))
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
                pred_pos_all.append(np.stack([f["key_pos"] for f in frames], axis=0))  # (T, J, 3)
                gt_pos_all.append(np.stack([f["ref_key_pos"] for f in frames], axis=0))  # (T, J, 3)
                pred_rot_all.append(np.stack([f["key_rot"] for f in frames], axis=0))  # (T, J, 4)
                gt_rot_all.append(np.stack([f["ref_key_rot"] for f in frames], axis=0))  # (T, J, 4)

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

        # with torch.inference_mode():
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

            wandb.log(wandb_metric_dict)

        return {
            "Eval/mean_reward": mean_rew,
            "Eval/success_rate": success_rate,
            "Eval/reward_per_motion": total_rewards,
            "Eval/success_flags": success_flags,
            "Eval/reward_until_fail": reward_until_fail_list
        }

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)


            log_reward_pos = statistics.mean(locs['infos']['reward_pos'].cpu().numpy().tolist())
            log_reward_rot = statistics.mean(locs['infos']['reward_rot'].cpu().numpy().tolist())
            log_reward_vel = statistics.mean(locs['infos']['reward_vel'].cpu().numpy().tolist())
            log_reward_ang_vel = statistics.mean(locs['infos']['reward_ang_vel'].cpu().numpy().tolist())
            self.writer.add_scalar('Imitation/reward_pos', log_reward_pos, locs['it'])
            self.writer.add_scalar('Imitation/reward_rot', log_reward_rot, locs['it'])
            self.writer.add_scalar('Imitation/reward_vel', log_reward_vel, locs['it'])
            self.writer.add_scalar('Imitation/reward_ang_vel', log_reward_ang_vel, locs['it'])

            log_pos_err = statistics.mean(locs['infos']['pos_err'].cpu().numpy().tolist())
            log_rot_err = statistics.mean(locs['infos']['rot_err'].cpu().numpy().tolist())
            log_vel_err = statistics.mean(locs['infos']['vel_err'].cpu().numpy().tolist())
            log_ang_vel_err = statistics.mean(locs['infos']['ang_vel_err'].cpu().numpy().tolist())
            self.writer.add_scalar('Imitation/pos_err', log_pos_err, locs['it'])
            self.writer.add_scalar('Imitation/rot_err', log_rot_err, locs['it'])
            self.writer.add_scalar('Imitation/vel_err', log_vel_err, locs['it'])
            self.writer.add_scalar('Imitation/ang_vel_err', log_ang_vel_err, locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n""")

        log_string += ep_string
        log_string += (f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n""")
        print(log_string)
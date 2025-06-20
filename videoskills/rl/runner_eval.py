from rsl_rl.runners import OnPolicyRunner
import numpy as np
import torch
from tqdm import tqdm

class RunnerWithEval(OnPolicyRunner):
    def eval(self, motion_ids=None):
        """Evaluate policy over multiple motions in parallel across environments."""
        self.alg.actor_critic.eval()
        self.env.eval_mode = True

        num_envs = self.env.num_envs
        motion_lib = self.env._motion_lib
        device = self.device

        if motion_ids is None:
            motion_ids = list(range(motion_lib.num_motions()))

        total_rewards = []
        success_flags = []
        reward_until_fail_list = []

        for i in tqdm(range(0, len(motion_ids), num_envs), desc="Evaluating motions"):
            batch_ids = motion_ids[i: i + num_envs]
            batch_size = len(batch_ids)

            padded_ids = torch.tensor(batch_ids + [batch_ids[-1]] * (num_envs - batch_size), device=device)

            with torch.inference_mode():
                obs = self.env.reset_with_motion_ids(padded_ids)
            # self.env.compute_observations()
            # obs = self.env.get_observations()
            #     obs = self.env.reset_with_motion_ids(padded_ids)

            cum_rewards = torch.zeros(num_envs, device=device)
            reward_until_fail = torch.zeros(num_envs, device=device)
            first_fail_recorded = torch.zeros(num_envs, dtype=torch.bool, device=device)

            episode_lengths = torch.zeros(num_envs, dtype=torch.int32, device=device)
            done_flags = torch.zeros(num_envs, dtype=torch.bool, device=device)


            motion_lengths = (motion_lib._motion_lengths[batch_ids]/self.env.dt).int()

            for _ in range(motion_lengths.max()):
                with torch.inference_mode():
                    action = self.alg.actor_critic.act_inference(obs.to(device))
                    obs, _, rewards, dones, _ = self.env.step(action)

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

            for env_id in range(batch_size):
                ep_len = episode_lengths[env_id].item()
                expected_len = motion_lengths[env_id].item()
                success = ep_len >= (expected_len - 1)
                success_flags.append(success)
                total_rewards.append(cum_rewards[env_id].item())

                if not success:
                    reward_until_fail_list.append(reward_until_fail[env_id].item())

        num_success = sum(success_flags)
        num_total = len(success_flags)
        success_rate = num_success / num_total
        mean_rew = np.mean(total_rewards)

        print(f"[Eval] Success rate: {success_rate:.2%}")
        print(f"[Eval] Mean reward across {len(motion_ids)} motions: {mean_rew:.2f}")
        print(f"[Eval] Avg. reward until failure (only failed): {np.mean(reward_until_fail_list):.2f}")

        self.env.eval_mode = False
        return {
            "Eval/mean_reward": mean_rew,
            "Eval/success_rate": success_rate,
            "Eval/reward_per_motion": total_rewards,
            "Eval/success_flags": success_flags,
            "Eval/reward_until_fail": reward_until_fail_list
        }

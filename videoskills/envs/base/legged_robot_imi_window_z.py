import torch
from videoskills.envs.base.legged_robot_config import LeggedRobotCfg
from videoskills.envs.base.legged_robot_imi import LeggedRobotImi


class LeggedRobotImiWinZ(LeggedRobotImi):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # 定义历史回溯: 0代表t-1, 2代表t-3, 5代表t-6
        self.latent_dim = 32
        self.history_strides = [0, 2, 5]
        self.max_history_len = max(self.history_strides) + 2  # 安全起见多留一点空间

        # 计算 Humanoid Obs 维度
        # 13(root) + (K-1)*3(pos) + (K-1)*4(rot) + (K-1)*3(vel) + (K-1)*3(ang_vel)
        # 注意：如果 activate_quat_to_tan_norm=True，rot 是 6 维
        rot_dim = 6 if self.activate_quat_to_tan_norm else 4
        self.history_obs_dim = 13 + (self.num_bodies - 1) * (3 + rot_dim + 3 + 3)

        self.obs_history_buf = torch.zeros(
            self.num_envs,
            self.max_history_len,
            self.history_obs_dim,
            device=self.device,
            dtype=torch.float
        )

    def compute_observations(self):
        # 1. 计算当前的 Humanoid Obs (s_t)
        humanoid_obs = self.compute_humanoid_observations()

        # 2. 【核心修正】先更新 Buffer！
        # 这样能保证如果发生 Reset，Buffer 里的所有内容都被刷成了 s_0
        self._update_history_buf(humanoid_obs)

        # 3. 【核心修正】从更新后的 Buffer 获取历史 (h_t)
        # 此时 Buffer[-1] 是 s_t, Buffer[-2] 是 s_{t-1}
        history_obs_flat = self._get_strided_history()

        # 4. 计算 Task Obs
        progress = (self.episode_length_buf.to(torch.float) + 1) * self.dt
        motion_times = progress + self._motion_start_times
        motion_state = self._motion_lib.get_motion_state(self._sampled_motion_ids, motion_times)

        self.ref_root_pos[:] = motion_state["root_pos"] + self.pos_offset['root']
        self.ref_root_rot[:] = motion_state["root_rot"]
        self.ref_dof_pos[:] = motion_state["dof_pos"]
        self.ref_body_pos = motion_state["key_pos"] + self.pos_offset['body']
        self.ref_body_rot = motion_state["key_rot"]
        self.ref_body_vel = motion_state["key_vel"]
        self.ref_body_ang_vel = motion_state["key_ang_vel"]

        # task_obs 不需要历史，只看当前差值
        z_obs = self._motion_lib.get_motion_z(self._sampled_motion_ids, motion_times)

        # 5. 拼接: [s_t, h_t, task, a_{t-1}]
        self.obs_buf = torch.cat((humanoid_obs, history_obs_flat, z_obs, self.actions), dim=-1)

        # Eval Logic
        if self.eval_mode and hasattr(self, '_motion_lib_gt'):
            motion_state_gt = self._motion_lib_gt.get_motion_state(self._sampled_motion_ids, motion_times)
            self.ref_root_pos_gt = motion_state_gt["root_pos"] + self.pos_offset['root']
            self.ref_root_rot_gt = motion_state_gt["root_rot"]
            self.ref_dof_pos_gt = motion_state_gt["dof_pos"]
            self.ref_body_pos_gt = motion_state_gt["key_pos"] + self.pos_offset['body']
            self.ref_body_rot_gt = motion_state_gt["key_rot"]

    def _update_history_buf(self, new_obs):
        """
        更新历史缓冲区。
        """
        # --- 步骤 1: 标准更新 (左移 + 填入最新) ---
        self.obs_history_buf = torch.roll(self.obs_history_buf, shifts=-1, dims=1)
        self.obs_history_buf[:, -1, :] = new_obs

        # --- 步骤 2: 修正重置环境 (Handle Resets) ---
        # 这里的关键是 reset_idx 已经在 compute_observations 之前把 episode_length_buf 置 0 了
        reset_env_ids = (self.episode_length_buf == 0).nonzero(as_tuple=False).flatten()

        if len(reset_env_ids) > 0:
            # 取出这些环境当前的 obs (即 new_obs)
            s_0 = new_obs[reset_env_ids]  # [N_reset, obs_dim]

            # 将 Buffer 全部刷成 s_0
            s_0_expanded = s_0.unsqueeze(1).expand(-1, self.max_history_len, -1)
            self.obs_history_buf[reset_env_ids] = s_0_expanded

    def _get_strided_history(self):
        """
        提取历史。
        注意：因为我们在 Get 之前先 Update 了，Buffer 现在的结构是 [..., s_{t-2}, s_{t-1}, s_t]
        Last Element (Index -1) 是当前帧 s_t
        """
        # 如果 stride=0 (想要 t-1): 对应 Buffer 的倒数第 2 个 (-2)
        # 如果 stride=2 (想要 t-3): 对应 Buffer 的倒数第 4 个 (-4)
        # 公式: index = -(stride + 2)
        indices = [-(s + 2) for s in self.history_strides]

        selected_history = self.obs_history_buf[:, indices, :]
        return selected_history.view(self.num_envs, -1)
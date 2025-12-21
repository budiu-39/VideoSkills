from isaacgym import gymapi
import time
import torch
from videoskills.envs.base.legged_robot_imi import LeggedRobotImi
from videoskills.envs.base.legged_robot_hoi import LeggedRobotHoi
from videoskills.envs.base.legged_robot_imi import (
    compute_humanoid_observations_jit,
    compute_mimic_observations_jit,
    compute_amp_observations_jit,
)
from isaacgym import gymtorch

SMPL24_BODIES = [
    'Pelvis',
    'L_Hip','L_Knee','L_Ankle','L_Toe',
    'R_Hip','R_Knee','R_Ankle','R_Toe',
    'Torso','Spine','Chest','Neck','Head',
    'L_Thorax','L_Shoulder','L_Elbow','L_Wrist','L_Hand',
    'R_Thorax','R_Shoulder','R_Elbow','R_Wrist','R_Hand',
]

# TODO：No hand 部分其实应该合并到 LeggedRobotImi 里去的，但现在改动太大，先放这里
class SMPLXRobot(LeggedRobotHoi):

    def _build_env(self, env_id, env_ptr, humanoid_asset):
    # filter 默认全为 1
        super()._build_env(env_id, env_ptr, humanoid_asset)

        if self.cfg.asset.self_collisions:
            robot_handle = self.robot_handles[env_id]
            filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            props = self.gym.get_actor_rigid_shape_properties(env_ptr, robot_handle)

            assert (len(filter_ints) == len(props))

            for p_idx in range(len(props)):
                props[p_idx].filter = filter_ints[p_idx]
            self.gym.set_actor_rigid_shape_properties(env_ptr, robot_handle, props)

        return

    def _create_envs(self):

        self._obj_handles = []
        super()._create_envs()

        # SMPL_BODIES_NO_HAND = [
        #     'Pelvis',
        #     'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
        #     'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
        #     'Torso', 'Spine', 'Chest', 'Neck', 'Head',
        #     'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
        #     'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist',
        # ]
        #
        # # sim 中名字 -> 索引
        # name2sim = {n: i for i, n in enumerate(self.body_names)}
        #
        # # 选择出 sim 里的 24 个 body 索引（顺序按 SMPL24_BODIES）
        # self.body_no_hand_ids = torch.as_tensor(
        #     [name2sim[n] for n in SMPL_BODIES_NO_HAND if n in name2sim],
        #     dtype=torch.long, device=self.device
        # )
        # self.dof_no_hand_ids = ((self.body_no_hand_ids[1:]-1).unsqueeze(1)*3 + torch.tensor([0,1,2], device=self.device)).reshape(-1)
        self._build_body_hand_masks()
        self._build_dof_hand_masks()

    def _is_hand_body(self, name: str) -> bool:
        # Wrist 视为“非手”（保留可动）；手掌/手指为“手”
        if "Wrist" in name:
            return False
        keys = ["Hand", "Thumb", "Index", "Middle", "Ring", "Pinky"]
        return any(k in name for k in keys)

    def _build_body_hand_masks(self):   # 基于body编号的hand mask
        is_hand_body = [self._is_hand_body(n) for n in self.body_names]
        self.hand_body_mask = torch.tensor(is_hand_body, device=self.device, dtype=torch.bool)
        self.no_hand_body_mask = ~self.hand_body_mask

    def _is_hand_dof(self, name: str) -> bool:
        # wrist 允许运动，手指锁定；只要名字里含这些子串就当作手指关节
        if "Wrist" in name:
            return False
        hand_keys = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        return any(k in name for k in hand_keys)

    def _build_dof_hand_masks(self): # 基于dof编号的hand mask
        # 与 self.dof_names 一一对应
        is_hand = [self._is_hand_dof(n) for n in self.dof_names]
        self.hand_dof_mask = torch.tensor(is_hand, device=self.device, dtype=torch.bool)
        self.no_hand_dof_mask = ~self.hand_dof_mask

    def compute_humanoid_observations(self):
        return compute_humanoid_observations_jit(self.base_pos, self.base_quat,
                                self.body_pos[:, self.no_hand_body_mask], self.body_rot[:, self.no_hand_body_mask],
                                self.body_vel[:, self.no_hand_body_mask], self.body_ang_vel[:, self.no_hand_body_mask])

    def compute_mimic_observations(self):
        task_obs = compute_mimic_observations_jit(self.base_pos, self.base_quat,
                                                  self.body_pos[:, self.no_hand_body_mask],
                                                  self.body_rot[:, self.no_hand_body_mask],
                                                  self.body_vel[:, self.no_hand_body_mask],
                                                  self.body_ang_vel[:, self.no_hand_body_mask],
                                                  self.ref_body_pos[:, self.no_hand_body_mask],
                                                  self.ref_body_rot[:, self.no_hand_body_mask],
                                                  self.ref_body_vel[:, self.no_hand_body_mask],
                                                  self.ref_body_ang_vel[:, self.no_hand_body_mask])

        return task_obs

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        if not self.headless:
            self.gym.clear_lines(self.viewer)
        self.render()
        for _ in range(self.cfg.control.decimation):
            if self.drive_mode == gymapi.DOF_MODE_EFFORT:
                self.torques = self._compute_torques(self.actions).view(self.torques.shape)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            else:
                if getattr(self.cfg.control, "fixed_hand", False):
                    actions = torch.zeros(self.num_envs, self.num_dofs, device=self.device, dtype=actions.dtype)
                    actions[:, self.hand_dof_mask] = 0.0
                    actions[:, self.no_hand_dof_mask] = self.actions
                pd_tar = self.pd_action_offset + self.pd_action_scale * actions
                pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
                self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)

            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras



    # TODO: 测试一下关节对不对！  这里的代码是错误的因为会改变 self.dof_pos的内存
    # def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,
    #                    key_pos, key_rot, key_vel, key_ang_vel):
    #     self.robot_states[env_ids, 0:3] = root_pos
    #     self.robot_states[env_ids, 3:7] = root_rot
    #     self.robot_states[env_ids, 7:10] = root_vel
    #     self.robot_states[env_ids, 10:13] = root_ang_vel
    #
    #     row = self.dof_pos[env_ids].clone()
    #     row[:, self.dof_no_hand_ids_sim] = dof_pos[:, self.dof_no_hand_ids_sim]
    #     self.dof_pos[env_ids] = row
    #
    #     rowv = self.dof_vel[env_ids].clone()
    #     rowv[:, self.dof_no_hand_ids_sim] = dof_vel[:, self.dof_no_hand_ids_sim]
    #     self.dof_vel[env_ids] = rowv

        # self.dof_pos[env_ids][:, self.dof_no_hand_ids_sim] = dof_pos[:, self.dof_no_hand_ids_sim]
        # self.dof_vel[env_ids][:, self.dof_no_hand_ids_sim] = dof_vel[:, self.dof_no_hand_ids_sim]

        return







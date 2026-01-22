
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from videoskills import LEGGED_GYM_ROOT_DIR
from videoskills.envs.base.legged_robot_config import LeggedRobotCfg
from videoskills.envs.base.legged_robot import LeggedRobot
# from videoskills.utils.motion_lib import MotionLib
from videoskills.utils.motion_lib_z import MotionLibZ as MotionLib
from videoskills.utils.torch_utils import to_torch, quat_mul, quat_conjugate, quat_to_angle_axis
from videoskills.utils.torch_utils import calc_heading_quat_inv, calc_heading_quat, quat_apply, quat_to_tan_norm
from videoskills.utils.torch_utils import exp_map_to_quat
from videoskills.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from torch import Tensor



class LeggedRobotImi(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.early_termination = self.cfg.early_termination.enabled
        self.stiffness = [v * self.cfg.control.pd_scale for v in self.cfg.control.stiffness]
        self.damping = [v * self.cfg.control.pd_scale for v in self.cfg.control.damping]


        # quat_to_tan_norm ablation study
        self.activate_quat_to_tan_norm = self.cfg.env.activate_quat_to_tan_norm
        if self.cfg.env.activate_quat_to_tan_norm:
            self.cfg.env.num_observations = self.cfg.env.norm_num_observations

        self._parse_cfg(self.cfg)

        self.early_termination_buf = torch.zeros(self.cfg.env.num_envs, device=sim_device, dtype=torch.bool)
        self._too_far_count = torch.zeros(self.cfg.env.num_envs, device=sim_device, dtype=torch.int32)

        if isinstance(self.cfg.motion.file, list):
            motion_file = self.cfg.motion.file
        else:
            motion_file = self.cfg.motion.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self._load_motion(motion_file)
        self._motion_start_times = torch.zeros(self.num_envs).to(self.device)
        self._sampled_motion_ids = torch.zeros(self.num_envs).long().to(self.device)

        self._state_init = self.cfg.init_state.type

        self.ref_root_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ref_root_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.ref_dof_pos = torch.zeros(self.num_envs, self.num_dofs, device=self.device)

        if hasattr(self.cfg.early_termination,'reset_body'):
            self.reset_body_id = self._build_key_body_ids_tensor(self.cfg.early_termination.reset_body)
        else:
            self.reset_body_id = torch.arange(0, self.num_bodies, device=self.device)
        self.early_termination_distance = torch.tensor(self.cfg.early_termination.distance,
                                                       device=self.device) ** 2

        self.visual_prob = 1.0
        self.env_visual_mask = torch.zeros(self.num_envs, 1, device=self.device)

        if not self.headless:
            # 1. 设置摄像机位置 (相对于环境原点的坐标)
            # 这里设置为机器人侧后方：x=3米, y=0, z=1.5米高
            cam_pos = gymapi.Vec3(0.0, -3.0, 1.5)

            # 2. 设置观察目标点 (相对于环境原点的坐标)
            # 这里设置为看向机器人的腰部高度：z=1.0米
            cam_target = gymapi.Vec3(0.0, 0.0, 1.0)

            # 3. 将摄像机锁定并看向第一个环境 (self.envs[0])
            # 这样无论机器人在世界坐标的哪里，镜头都会基于第一个环境的相对坐标进行初始化
            self.gym.viewer_camera_look_at(self.viewer, self.envs[-1], cam_pos, cam_target)



    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = self.drive_mode
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # asset is a xml resource
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # marker
        if not self.headless:
            self._sphere_geom = gymutil.WireframeSphereGeometry(
                radius=0.03,
                num_lats=12,  # “纬线”数
                num_lons=12,  # “经线”数
                color=(1.0, 0.2, 0.2))

        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_body_ids = np.arange(1, len(self.body_names)).tolist()
        # TODO: 要改的 因为不一定是 3 dof
        self.dof_offsets = np.linspace(0, len(self.dof_names), len(self.body_names)).astype(int)
        self.cfg.init_state.default_joint_angles = {dof_name: 0.0 for dof_name in self.dof_names}

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

        self._get_env_origins()

        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.robot_handles = []
        self.envs = []
        max_agg_bodies = 160
        max_agg_shapes = 160

        if hasattr(self.cfg.asset,'file_urdf'):
            # If the asset is a URDF, we need to copy some properties from the URDF to the MJCF asset.
            self.dof_props = self._build_dof_properties_from_urdf(asset_options, robot_asset)
            # self.dof_props["damping"] = torch.ones(len(self.dof_names), dtype=torch.float, device=self.device)
        else:
            self.dof_props = self.gym.get_asset_dof_properties(robot_asset)
            if self.drive_mode == gymapi.DOF_MODE_EFFORT:
                self.dof_props["damping"] = torch.ones(len(self.dof_names), dtype=torch.float, device=self.device)
                self.dof_props["stiffness"] = torch.zeros(len(self.dof_names), dtype=torch.float, device=self.device)
                self.dof_props['effort'] = torch.tensor(self.cfg.control.limit, dtype=torch.float, device=self.device)
                self.dof_props["velocity"] = torch.tensor(self.cfg.control.velocity_limit, dtype=torch.int32,
                                                          device=self.device)
            else:
            #     # self.dof_props['stiffness'] = self.dof_props['stiffness'] * self.cfg.control.pd_scale
            #     # self.dof_props['damping'] = self.dof_props['damping'] * self.cfg.control.pd_scale
                self.stiffness = self.dof_props['stiffness'].tolist()
                self.damping = self.dof_props['damping'].tolist()
            #     self.dof_props['stiffness'] = torch.tensor(self.stiffness, dtype=torch.float, device=self.device)
            #     self.dof_props['damping'] = torch.tensor(self.damping, dtype=torch.float, device=self.device)
        self.dof_props['driveMode'] = torch.tensor([self.cfg.asset.default_dof_drive_mode] * self.num_dofs,
                                                   dtype=torch.int32, device=self.device)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.gym.begin_aggregate(env_handle, max_agg_bodies, max_agg_shapes, True)
            self._build_env(i, env_handle, robot_asset)
            self.gym.end_aggregate(env_handle)
            self.envs.append(env_handle)

        # regularization?
        self._build_pd_action_offset_scale()
        self.p_gains = torch.tensor(self.stiffness, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.tensor(self.damping,  dtype=torch.float, device=self.device, requires_grad=False)

        # Hacky for trained smpl model
        if hasattr(self.cfg.control, "action_scale"):
            self.pd_action_scale = self.cfg.control.action_scale * torch.ones_like(self.dof_limits_lower).to(self.device)

        self.body_ids = torch.arange(len(self.body_names), device=self.device, dtype=torch.long)


        # Test
        body_props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.robot_handles[0])  # 获取刚体属性
        total_mass = 0.0
        for i, prop in enumerate(body_props):
            # print(f"Body {i} mass:", prop.mass)
            total_mass += prop.mass
        print("Total mass of the robot:", total_mass)

    def _build_env(self, env_id, env_ptr, robot_asset):
        col_group = env_id

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*(self.base_init_state[:3] + self.env_origins[env_id]))
        start_pose.r = gymapi.Quat(*self.base_init_state[3:7])

        # here is the instance of the humanoid asset
        robot_handle = self.gym.create_actor(env_ptr, robot_asset, start_pose, self.cfg.asset.name, col_group, 1,
                                             0)
        self.robot_handles.append(robot_handle)

        # if hasattr(self.cfg.rewards.scales, 'dof_force'):
        self.gym.enable_actor_dof_force_sensors(env_ptr, robot_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, robot_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        self.gym.set_actor_dof_properties(env_ptr, robot_handle, self.dof_props)

        return

    def _build_dof_properties_from_urdf(self, asset_options, robot_asset):
        urdf_path = self.cfg.asset.file_urdf.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        ref_asset = self.gym.load_asset(self.sim,
                                        os.path.dirname(urdf_path),
                                        os.path.basename(urdf_path),
                                        asset_options)

        ref_dof_names = self.gym.get_asset_dof_names(ref_asset)
        ref_dof_props = self.gym.get_asset_dof_properties(ref_asset)

        # 建立名字 → index 映射
        name2idx_ref = {n: i for i, n in enumerate(ref_dof_names)}

        dof_props = self.gym.get_asset_dof_properties(robot_asset).copy()

        fields_to_copy = ["velocity", "effort"]  # 想同步的字段

        for i_mjcf, name in enumerate(self.dof_names):
            if name not in name2idx_ref:
                continue
            j_ref = name2idx_ref[name]
            for field in fields_to_copy:
                dof_props[field][i_mjcf] = ref_dof_props[field][j_ref]

        dof_props['armature'] = torch.tensor([0.02] * self.num_dofs, dtype=torch.float, device=self.device)

        return dof_props

    def render(self):
        if not self.headless and hasattr(self, "ref_body_pos"):
            max_vis_envs = self.num_envs
            T = gymapi.Transform()
            for i in range(max_vis_envs):
                env_ptr = self.envs[i]  # 当前环境的指针
                pts = self.ref_body_pos[i].cpu().numpy()  # (K,3)
                for p in pts:
                    T.p.x, T.p.y, T.p.z = map(float, p)
                    gymutil.draw_lines(self._sphere_geom,
                                       self.gym,
                                       self.viewer,
                                       env_ptr,
                                       T)

                # Draw arrows at feet that just landed
            if hasattr(self, "just_landed_event"):
                feet_pos = self._rigid_body_state_reshaped[..., self.feet_indices, 0:3]  # [num_envs, num_feet, 3]

                # 创建一个蓝色球体
                sphere = gymutil.WireframeSphereGeometry(
                    radius=0.04,
                    num_lats=10,
                    num_lons=10,
                    color=(0.2, 0.2, 1.0)  # 蓝色
                )

                env_ids, foot_ids = torch.nonzero(self.just_landed_event, as_tuple=True)
                for env_id, foot_id in zip(env_ids.tolist(), foot_ids.tolist()):
                    env_ptr = self.envs[env_id]
                    foot_pos = feet_pos[env_id, foot_id].cpu().numpy()

                    T = gymapi.Transform()
                    T.p.x, T.p.y, T.p.z = map(float, foot_pos)

                    # gymutil.draw_lines(sphere, self.gym, self.viewer, env_ptr, T)
        super().render()


    # TODO: 重写 reset dof 和 load motion， 把 root 和 dof 和在一起
    def _load_motion(self, motion_file):

        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self.dof_body_ids,
                                     dof_offsets=self.dof_offsets,
                                     key_body_ids=self.body_ids,
                                     rotate_motion=self.cfg.motion.rotate_motion,
                                     device=self.device)


    def _load_gt_motion(self, motion_file):

        self._motion_lib_gt = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self.dof_body_ids,
                                     dof_offsets=self.dof_offsets,
                                     key_body_ids=self.body_ids,
                                     rotate_motion=self.cfg.motion.rotate_motion,
                                     device=self.device)

    def reset_idx(self, env_ids):
        # TODO: 需要更新 task obs!!!!
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        self.gym.clear_lines(self.viewer)
        self._reset_robot(env_ids)
        self._reset_env_tensors(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        # self.feet_air_time[env_ids] = 0.  # good idea!
        self.episode_length_buf[env_ids] = 0
        self._too_far_count[env_ids] = 0
        # TODO： 修改了這裡，从 1 改成了 0
        self.reset_buf[env_ids] = 1
        # self.early_termination_buf[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        self.extras["early_termination_buf"] = self.early_termination_buf
        self.extras["recorded_data"] = [[] for _ in range(self.num_envs)]

        mask = (torch.rand(len(env_ids), 1, device=self.device) < self.visual_prob).float()
        self.env_visual_mask[env_ids] = mask


        if hasattr(self.envs, "reward_subterm_sums"):
            for k, buf in self.reward_subterm_sums.items():
                # 仅对这批完成 episode 的 env_ids 做平均
                if buf is None or buf.numel() == 0:
                    continue
                mean_val = buf[env_ids].mean()
                # 用“完整英文路径名”，不缩写：
                # 约定所有子项都以 "rew_sub/" 前缀进入 infos['episode']
                self.extras["episode"][f"rew_sub/{k}"] = mean_val.detach()

            # 清零这些 env 的累计，避免跨 episode 污染
            for k in self.reward_subterm_sums:
                self.reward_subterm_sums[k][env_ids] = 0.0
        for key in self.episode_sums.keys():
            # 获取这批结束环境的累计奖励
            reward_tensor = self.episode_sums[key][env_ids]
            mean_reward = torch.mean(reward_tensor)

            # --- 核心改动：只有非零才记录 ---
            if mean_reward.item() != 0:
                self.extras["episode"]['rew_' + key] = mean_reward
            # ---------------------------

            # 无论是否打印，都必须清空这批 env 的累计值，防止下一个 Episode 污染
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self._refresh_sim_tensors()

    def _reset_robot(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if (self._state_init == 'default'):
            self._reset_default(env_ids)
        elif (self._state_init == 'start'
              or self._state_init == 'random'):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == 'hybrid'):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == 'physical'):
            if not self.physics_buffer_initialized:
                self._init_phys_init_state_buffer()
                self.physics_buffer_initialized = True
            self._reset_phys_state_init(env_ids)

        self.motion_lengths = self._motion_lib.get_motion_length(self._sampled_motion_ids[env_ids])/ self.dt
        # from 0, therefore the real length is (int(motion_lengths) + 1)


    def _reset_default(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.dof_vel[env_ids] = 0.
        self._reset_default_env_ids = env_ids

        # base position
        self.robot_states[env_ids] = self.base_init_state
        self.robot_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.robot_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        # no reference motion, reset tracking buffers
        self._sampled_motion_ids[env_ids] = 0
        self._motion_start_times[env_ids] = 0.0

    def sample_motions(self, env_ids):
        motion_ids = self._motion_lib.sample_motions(len(env_ids))
        self._sampled_motion_ids[env_ids] = motion_ids

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        self.sample_motions(env_ids)

        if self._state_init == 'random':
            motion_times = self._motion_lib.sample_time(self._sampled_motion_ids[env_ids])
        elif self._state_init == 'start':
            motion_times = torch.zeros(num_envs, device=self.device)
        elif self._state_init == 'hybrid':
            start_prob = 0.05
            start_mask = torch.rand(num_envs, device=self.device) < start_prob
            motion_times = self._motion_lib.sample_time(self._sampled_motion_ids[env_ids])
            motion_times[start_mask] = 0.0

        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        if self.eval_mode:
            motion_times = torch.zeros_like(motion_times)

        motion_state = self._motion_lib.get_motion_state(self._sampled_motion_ids[env_ids], motion_times)

        self._set_env_state(env_ids=env_ids,
                            root_pos=motion_state["root_pos"] + self.pos_offset['root'][env_ids],
                            root_rot=motion_state["root_rot"],
                            dof_pos=motion_state["dof_pos"],
                            root_vel=motion_state["root_vel"],
                            root_ang_vel=motion_state["root_ang_vel"],
                            dof_vel=motion_state["dof_vel"],
                            key_pos=motion_state["key_pos"] + self.pos_offset['body'][env_ids],
                            key_rot= motion_state["key_rot"],
                            key_vel=motion_state["key_vel"],
                            key_ang_vel=motion_state["key_ang_vel"]
                            )

        self._motion_start_times[env_ids] = motion_times


    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,
                       key_pos, key_rot, key_vel, key_ang_vel
                       ):
        self.robot_states[env_ids, 0:3] = root_pos
        self.robot_states[env_ids, 3:7] = root_rot
        self.robot_states[env_ids, 7:10] = root_vel
        self.robot_states[env_ids, 10:13] = root_ang_vel

        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        # self.body_pos, self.body_rot, self.body_vel, self.body_ang_vel,
        self.body_pos[env_ids,:] = key_pos
        self.body_rot[env_ids,:] = key_rot
        self.body_vel[env_ids,:] = key_vel
        self.body_ang_vel[env_ids,:] = key_ang_vel

        return

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # fall = torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)  # raw pitch yaw

        # 1) 时间相关
        time_out = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        progress = (self.episode_length_buf.to(torch.float) + 1) * self.dt  # progress is the current ref_motion
        motion_lens = self._motion_lib.get_motion_length(self._sampled_motion_ids)
        ref_out = (progress + self._motion_start_times )> motion_lens

        use_gt = bool(self.eval_mode) and hasattr(self, "_motion_lib_gt")

        if use_gt:
            ref_body_pos_used = self.ref_body_pos_gt
            ref_root_pos_used = self.ref_root_pos_gt
        else:
            ref_body_pos_used = self.ref_body_pos
            ref_root_pos_used = self.ref_root_pos

        # TODO： 这是原版，只要任何一个关键点 > 0.5 m 就触发
        body_delta_sq = torch.sum((self.body_pos[:,self.reset_body_id]
                                   - self.ref_body_pos[:, self.reset_body_id]) ** 2, dim=2)  # → ℝ[num_envs, K]
        body_too_far_now = torch.any(body_delta_sq > self.early_termination_distance[self.reset_body_id], dim=1)
        if self.eval_mode:
            body_too_far_now = (body_delta_sq.mean(dim=1) > self.early_termination_distance[0])  # → ℝ[num_envs]

        # TODO： 这是原版平均距离且无视 gravity axis 的版本，注意现在不论是 eval 还是 train 都应用了平均逻辑（考虑到数据很 noisy)
        # root_pos_wo_h = self.base_pos.clone()
        # root_pos_wo_h[:, 0:2] = 0.0
        # ref_rot_pos_wo_h = ref_root_pos_used.clone()
        # ref_rot_pos_wo_h[:, 0:2] = 0.0
        # body_delta_sq = torch.sum((self.body_pos[:, self.reset_body_id] - root_pos_wo_h.unsqueeze(1)
        #                         + ref_rot_pos_wo_h.unsqueeze(1) - ref_body_pos_used[:, self.reset_body_id]) ** 2, dim=2)
        #
        # body_too_far_now = (body_delta_sq.mean(dim=1) > self.early_termination_distance[0])

        # if self.eval_mode: #use_gt:
        #     # 可以设置对比试验，eval 恒定用 方案 1, 训练时对比 方案 1 和 方案2 看一下区别
        #     # 方案 1 ： 连续多帧达标才视为真的 early termination
        #     valid_step = (self.episode_length_buf > 1)
        #     inc_mask = torch.logical_and(body_too_far_now, valid_step)
        #
        #     # 计数累加/清零（逐 env）
        #     self._too_far_count[inc_mask] += 1
        #     self._too_far_count[~inc_mask] = 0
        #
        #     # 连续 15 帧达标才视为真的 early termination
        #     body_too_far = (self._too_far_count >= 15)
        # else:
            # 方案 2： 只要当前帧达标就视为 early termination
            # 训练或无 GT 时，仍按“当前帧超过阈值 + 步数>1”判定
        body_too_far = torch.logical_and(body_too_far_now, (self.episode_length_buf > 1))

        self.reset_buf = ref_out | body_too_far | time_out
        self.time_out_buf = time_out | ref_out
        self.early_termination_buf = body_too_far

        if not self.early_termination:
            self.reset_buf = time_out | ref_out
            self.time_out_buf = time_out | ref_out

        # Eval 模式下沿用你原先的覆盖逻辑
        if self.eval_mode:
            self.reset_buf = ref_out | body_too_far
            if not self.early_termination:
                self.reset_buf = ref_out
            self.time_out_buf = time_out | ref_out

    def _reset_env_tensors(self, env_ids):
        # here dof_pos and dof_vel is view of dof_state
        env_ids_int32 = self.robot_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # TODO: 有点好奇这个重不重要？
        if self.drive_mode == gymapi.DOF_MODE_POS:
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_pos.contiguous()),
                                                            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def land_event_detection(self):   # 还有必要升级
        # foot_floor_contact detect
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # 更新 contact history 缓冲（滑动窗口）
        self.contact_history = torch.roll(self.contact_history, shifts=-1, dims=0)
        self.contact_history[-1] = contact
        # 连续 3 帧都接地
        all_on = torch.all(self.contact_history, dim=0)  # [num_envs, num_feet]
        # 检测“离地 → 接地”事件（刚刚完成转变）
        self.just_landed_event = torch.logical_and(all_on, ~self.last_landing_mask)  # [num_envs, num_feet]

        self.last_contacts = contact
        if torch.any(self.just_landed_event):
            print('contact detected! contact force:', self.contact_forces[:, self.feet_indices])

        self.last_landing_mask = all_on.clone()

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def compute_observations(self):
        # 应该在这里更新 ref
        ## 有 2 个 大的 obs 分别是 smpl 的自身感知 humanoid obs 和 task obs，其中 humanoid obs 参考 compute_humanoid_observations
        ## 此外还有一个 actions 的 obs
        # TODO： 不过有一个事情是！这个 humanoid_obs 其实非常难以获得。所以如果目的是实机的话，obs 是要重写的
        ## task obs 参考 compute_imitation_observations

        progress = (self.episode_length_buf.to(torch.float) + 1) * self.dt
        motion_times = progress + self._motion_start_times
        # motion_lens = self._motion_lib.get_motion_length(self._sampled_motion_ids)
        # motion_times = torch.fmod(motion_times, motion_lens)
        motion_state = self._motion_lib.get_motion_state(self._sampled_motion_ids, motion_times)

        self.ref_root_pos[:] = motion_state["root_pos"] + self.pos_offset['root']
        self.ref_root_rot[:] = motion_state["root_rot"]
        self.ref_dof_pos[:] = motion_state["dof_pos"]
        self.ref_body_pos = motion_state["key_pos"] + self.pos_offset['body']
        self.ref_body_rot = motion_state["key_rot"]
        self.ref_body_vel = motion_state["key_vel"]
        self.ref_body_ang_vel = motion_state["key_ang_vel"]

        humanoid_obs = self.compute_humanoid_observations()

        task_obs = self.compute_mimic_observations()

        # 2. 获取历史观测 (h_t)
        # 注意：要在 update 之前 get，这样拿到的才是过去的，不包含当前的
        # history_obs_flat = self._get_strided_history()
        #
        # # 3. 更新 Buffer (为下一帧做准备)
        # self._update_history_buf(humanoid_obs)

        self.obs_buf = torch.cat((humanoid_obs, task_obs, self.actions), dim=-1)

        if self.eval_mode and hasattr(self, '_motion_lib_gt'):
            motion_state_gt = self._motion_lib_gt.get_motion_state(self._sampled_motion_ids, motion_times)

            self.ref_root_pos_gt = motion_state_gt["root_pos"] + self.pos_offset['root']
            self.ref_root_rot_gt = motion_state_gt["root_rot"]
            self.ref_dof_pos_gt = motion_state_gt["dof_pos"]
            self.ref_body_pos_gt = motion_state_gt["key_pos"] + self.pos_offset['body']
            self.ref_body_rot_gt = motion_state_gt["key_rot"]
            # self.ref_body_vel_gt = motion_state_gt["key_vel"]
            # self.ref_body_ang_vel_gt = motion_state_gt["key_ang_vel"]

        # self.obs_buf = torch.cat((humanoid_obs, task_obs), dim=-1)

    def compute_humanoid_observations(self):
        return compute_humanoid_observations_jit(self.base_pos, self.base_quat,
                                self.body_pos, self.body_rot, self.body_vel, self.body_ang_vel)

    # def compute_humanoid_observations(self):
    #     root_h = self.base_pos[:, 2:3]
    #     heading_rot_inv = calc_heading_quat_inv(self.base_quat)
    #     heading_rot_inv_expand = heading_rot_inv.unsqueeze(1).expand(-1, self.body_pos.shape[1], -1)
    #     root_base_expand = self.base_pos.unsqueeze(1).expand(-1, self.body_pos.shape[1], -1)  # [N, K, 3]
    #     local_body_pos = quat_apply(heading_rot_inv_expand, self.body_pos - root_base_expand)[:,1:].view(self.num_envs, -1)
    #     local_body_rot = quat_mul(heading_rot_inv_expand, self.body_rot).view(self.num_envs, -1)
    #     if self.activate_quat_to_tan_norm:
    #         local_body_rot = quat_to_tan_norm(local_body_rot.view(-1, 4)).view(self.num_envs, -1) # [N, K, 4]
    #     local_body_vel = quat_apply(heading_rot_inv_expand, self.body_vel).view(self.num_envs, -1)
    #     local_body_ang_vel = quat_apply(heading_rot_inv_expand, self.body_ang_vel).view(self.num_envs, -1)
    #
    #     # 1 +  3 * (K - 1) + 4 * K + 3 * K + 3 * K = 1 + 3K + 4K + 3K + 3K - 3 = 13 * 24 - 2  # 310
    #     obs = torch.cat((root_h, local_body_pos, local_body_rot, local_body_vel, local_body_ang_vel), dim=-1)
    #     return obs

    def compute_mimic_observations(self):
        task_obs = compute_mimic_observations_jit(self.base_pos, self.base_quat,
                               self.body_pos, self.body_rot, self.body_vel, self.body_ang_vel,
                               self.ref_body_pos, self.ref_body_rot, self.ref_body_vel, self.ref_body_ang_vel)

        return task_obs

    def _build_key_body_ids_tensor(self, key_body_names):
        body_ids = [self.body_names.index(name) for name in key_body_names]
        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _reward_imitation(self):
        # reward 是不需要 heading 归一化 的！
        """
        Computes the imitation reward based on the difference between the current and reference body positions and rotations.
        The reward is computed in the heading frame of the root body.
        """
        pos_err = torch.mean(torch.square(self.body_pos - self.ref_body_pos), dim=1).mean(-1)
        # pos_err = torch.norm(self.body_pos - self.ref_body_pos, p=2, dim=-1).mean(dim=1)
        rot_diff = quat_mul(self.ref_body_rot, quat_conjugate(self.body_rot))
        diff_global_body_angle = quat_to_angle_axis(rot_diff)[0]
        rot_err = (diff_global_body_angle ** 2).mean(dim=-1)
        vel_err = torch.mean(torch.square(self.body_vel - self.ref_body_vel), dim=1).mean(-1)
        ang_vel_err = torch.mean(torch.square(self.body_ang_vel - self.ref_body_ang_vel), dim=1).mean(-1)

        # Compute the reward as a weighted sum of the errors
        reward_pos = self.cfg.rewards.task_w.w_pos * torch.exp(-self.cfg.rewards.task_w.k_pos * pos_err)
        reward_rot = self.cfg.rewards.task_w.w_rot * torch.exp(-self.cfg.rewards.task_w.k_rot * rot_err)
        reward_vel = self.cfg.rewards.task_w.w_vel * torch.exp(-self.cfg.rewards.task_w.k_vel * vel_err)
        reward_ang_vel = self.cfg.rewards.task_w.w_ang_vel * torch.exp(-self.cfg.rewards.task_w.k_ang_vel * ang_vel_err)

        reward = reward_pos + reward_rot + reward_vel + reward_ang_vel
        self.extras['reward_pos'] = reward_pos
        self.extras['reward_rot'] = reward_rot
        self.extras['reward_vel'] = reward_vel
        self.extras['reward_ang_vel'] = reward_ang_vel
        self.extras['pos_err'] = pos_err
        self.extras['rot_err'] = rot_err
        self.extras['vel_err'] = vel_err
        self.extras['ang_vel_err'] = ang_vel_err

        return reward

    def _reward_imitation_mul(self):
        # reward 是不需要 heading 归一化 的！
        """
        Computes the imitation reward based on the difference between the current and reference body positions and rotations.
        The reward is computed in the heading frame of the root body.
        """
        pos_err = torch.mean(torch.square(self.body_pos - self.ref_body_pos), dim=1).mean(-1)
        # pos_err = torch.norm(self.body_pos - self.ref_body_pos, p=2, dim=-1).mean(dim=1)
        # rot_err = torch.mean(torch.square(self.ref_body_rot - self.body_rot), dim=1).mean(-1)
        rot_diff = quat_mul(self.ref_body_rot, quat_conjugate(self.body_rot))
        diff_global_body_angle = quat_to_angle_axis(rot_diff)[0]
        rot_err = (diff_global_body_angle ** 2).mean(dim=-1)
        vel_err = torch.mean(torch.square(self.body_vel - self.ref_body_vel), dim=1).mean(-1)
        ang_vel_err = torch.mean(torch.square(self.body_ang_vel - self.ref_body_ang_vel), dim=1).mean(-1)

        # Compute the reward as a weighted sum of the errors
        reward_pos = torch.exp(-self.cfg.rewards.task_w.k_pos * pos_err)
        reward_rot = torch.exp(-self.cfg.rewards.task_w.k_rot * rot_err)
        reward_vel = torch.exp(-self.cfg.rewards.task_w.k_vel * vel_err)
        reward_ang_vel = torch.exp(-self.cfg.rewards.task_w.k_ang_vel * ang_vel_err)

        reward = reward_pos * reward_rot * reward_ang_vel
        self.extras['reward_pos'] = reward_pos
        self.extras['reward_rot'] = reward_rot
        self.extras['reward_vel'] = reward_vel
        self.extras['reward_ang_vel'] = reward_ang_vel
        self.extras['pos_err'] = pos_err
        self.extras['rot_err'] = rot_err
        self.extras['vel_err'] = vel_err
        self.extras['ang_vel_err'] = ang_vel_err

        return reward

    def _reward_imitation_rela(self):
        # reward 是不需要 heading 归一化 的！
        """
        Computes the imitation reward based on the difference between the current and reference body positions and rotations.
        The reward is computed in the heading frame of the root body.
        """

        cur_root_h = self.base_pos.clone()
        ref_root_h = self.ref_root_pos.clone()
        cur_root_h[:, 0:2] = 0.0
        ref_root_h[:, 0:2] = 0.0

        rel_pos = self.body_pos[...] - cur_root_h.unsqueeze(1)
        ref_rel_pos = self.ref_body_pos[...] - ref_root_h.unsqueeze(1)

        pos_err = torch.mean(torch.square(rel_pos - ref_rel_pos), dim=1).mean(-1)
        rot_diff = quat_mul(self.ref_body_rot, quat_conjugate(self.body_rot))
        diff_global_body_angle = quat_to_angle_axis(rot_diff)[0]
        rot_err = (diff_global_body_angle ** 2).mean(dim=-1)
        vel_err = torch.mean(torch.square(self.body_vel - self.ref_body_vel), dim=1).mean(-1)
        ang_vel_err = torch.mean(torch.square(self.body_ang_vel - self.ref_body_ang_vel), dim=1).mean(-1)

        # Compute the reward as a weighted sum of the errors
        reward_pos = self.cfg.rewards.task_w.w_pos * torch.exp(-self.cfg.rewards.task_w.k_pos * pos_err)
        reward_rot = self.cfg.rewards.task_w.w_rot * torch.exp(-self.cfg.rewards.task_w.k_rot * rot_err)
        reward_vel = self.cfg.rewards.task_w.w_vel * torch.exp(-self.cfg.rewards.task_w.k_vel * vel_err)
        reward_ang_vel = self.cfg.rewards.task_w.w_ang_vel * torch.exp(-self.cfg.rewards.task_w.k_ang_vel * ang_vel_err)

        reward = reward_pos + reward_rot + reward_vel + reward_ang_vel
        self.extras['reward_pos'] = reward_pos
        self.extras['reward_rot'] = reward_rot
        self.extras['reward_vel'] = reward_vel
        self.extras['reward_ang_vel'] = reward_ang_vel
        self.extras['pos_err'] = pos_err
        self.extras['rot_err'] = rot_err
        self.extras['vel_err'] = vel_err
        self.extras['ang_vel_err'] = ang_vel_err

        return reward

    def _reward_dof_force(self):
        # reward 是不需要 heading 归一化 的！
        """
        Computes the imitation reward based on the difference between the current and reference key body positions and rotations.
        The reward is computed in the heading frame of the root body.
        """
        reward = torch.abs(torch.multiply(self.dof_force_tensor, self.dof_vel)).sum(dim=-1)
        reward[self.episode_length_buf <= 3] = 0
        return reward


    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self._refresh_sim_tensors()
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.robot_states[:, 0:3]
        self.base_quat[:] = self.robot_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.robot_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.robot_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.body_pos[:] = self._rigid_body_state_reshaped[..., self.body_ids, 0:3]
        self.body_rot[:] = self._rigid_body_state_reshaped[..., self.body_ids, 3:7]
        self.body_vel[:] = self._rigid_body_state_reshaped[..., self.body_ids, 7:10]
        self.body_ang_vel[:] = self._rigid_body_state_reshaped[..., self.body_ids, 10:13]

        # compute observations, rewards, resets, ...

        self.compute_reward()  # both reward and terminationare done with the last reference motion
        self.check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()


        if self.is_recording_data:
            body_pos_cpu = (self.body_pos - self.pos_offset['body']).detach().cpu().numpy()
            ref_body_pos_cpu = (self.ref_body_pos - self.pos_offset['body']).detach().cpu().numpy()
            body_rot_cpu = self.body_rot.detach().cpu().numpy()
            ref_body_rot_cpu = self.ref_body_rot.detach().cpu().numpy()
            dof_pos_cpu = self.dof_pos.detach().cpu().numpy()
            ref_dof_pos_cpu = self.ref_dof_pos.detach().cpu().numpy()
            done_flags_cpu = self.done_flags.detach().cpu().numpy()  # 先转为 CPU 上的 bool 数组

            actions_cpu = self.actions.detach().cpu().numpy()

            if hasattr(self, 'ref_body_pos_gt'):
                ref_body_pos_gt_cpu = (self.ref_body_pos_gt - self.pos_offset['body']).detach().cpu().numpy()
                ref_body_rot_gt_cpu = self.ref_body_rot_gt.detach().cpu().numpy()
                ref_dof_pos_gt_cpu = self.ref_dof_pos_gt.detach().cpu().numpy()

            # 取所有 still-alive 的环境索引
            alive_ids = np.where(done_flags_cpu == False)[0]

            # 批量记录，不用 for-loop 判断
            for env_id in alive_ids:
                self.recorded_data[env_id].append({
                    'body_pos': body_pos_cpu[env_id].copy(),  # 以初始位置为基准
                    'ref_body_pos': ref_body_pos_cpu[env_id].copy(),
                    'body_rot': body_rot_cpu[env_id].copy(),
                    'ref_body_rot': ref_body_rot_cpu[env_id].copy(),
                    'dof_pos': dof_pos_cpu[env_id].copy(),
                    'ref_dof_pos': ref_dof_pos_cpu[env_id].copy(),
                    'actions': actions_cpu[env_id].copy(),
                    'ref_body_pos_gt': ref_body_pos_gt_cpu[env_id].copy() if hasattr(self, 'ref_body_pos_gt') else None,
                    'ref_body_rot_gt': ref_body_rot_gt_cpu[env_id].copy() if hasattr(self, 'ref_body_rot_gt') else None,
                    'ref_dof_pos_gt': ref_dof_pos_gt_cpu[env_id].copy() if hasattr(self, 'ref_dof_pos_gt') else None,
                })

        self.reset_idx(env_ids)
        self.extras['early_termination_buf'] = self.early_termination_buf
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        if self.cfg.env.land_event_detect:
            self.land_event_detection()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.robot_states[:, 7:13]

    def export_recorded_data(self, output_dir, motion_ids):
        os.makedirs(output_dir, exist_ok=True)
        for env_id, motion_id in enumerate(motion_ids):
            data = self.recorded_data[env_id]
            body_pos = np.stack([f['body_pos'] for f in data], axis=0)
            ref_pos = np.stack([f['ref_body_pos'] for f in data], axis=0)
            body_rot = np.stack([f['body_rot'] for f in data], axis=0)
            ref_rot = np.stack([f['ref_body_rot'] for f in data], axis=0)
            dof_pos = np.stack([f['dof_pos'] for f in data], axis=0)
            ref_dof_pos = np.stack([f['ref_dof_pos'] for f in data], axis=0)

            filename = f"motion_{motion_id}_env_{env_id}.npz"
            filepath = os.path.join(output_dir, filename)
            np.savez_compressed(filepath,
                                pred_pos=body_pos,
                                gt_pos=ref_pos,
                                pred_rot=body_rot,
                                gt_rot=ref_rot)

    def reset_with_motion_ids(self, motion_ids, random = False):
        """ Reset all environments with given motion ids. (For Evaluation)
            This method is used to reset the environment with specific motion ids, e.g. in the training stage.
        Args:
            motion_ids (torch.Tensor): Tensor of shape [num_envs] containing motion ids to reset the environments with.
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._sampled_motion_ids = motion_ids.detach()
        self._motion_start_times = torch.zeros(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        motion_state = self._motion_lib.get_motion_state(self._sampled_motion_ids[env_ids], self._motion_start_times[env_ids])

        self._set_env_state(env_ids=env_ids,
                            root_pos=motion_state["root_pos"] + self.pos_offset['root'][env_ids],
                            root_rot=motion_state["root_rot"],
                            dof_pos=motion_state["dof_pos"],
                            root_vel=motion_state["root_vel"],
                            root_ang_vel=motion_state["root_ang_vel"],
                            dof_vel=motion_state["dof_vel"],
                            key_pos=motion_state["key_pos"] + self.pos_offset['body'][env_ids],
                            key_rot=motion_state["key_rot"],
                            key_vel=motion_state["key_vel"],
                            key_ang_vel=motion_state["key_ang_vel"]
                            )
        self._reset_env_tensors(env_ids)
        self.gym.clear_lines(self.viewer)

        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        # self.feet_air_time[env_ids] = 0.  # good idea!
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids])
            # self.extras["episode"]['rew_' + key] = torch.mean(
            #     self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self._refresh_sim_tensors()

        motion_lens = self._motion_lib._motion_lengths[motion_ids]
        self.extras["motion_length"] = motion_lens.clone()
        self.compute_observations()

        # obs, _, _, _, _ = self.step(
        #     torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return self.obs_buf

    def begin_eval(self, motion_ids):
        """Initialize a simple linear iterator over motion_ids for eval."""
        if motion_ids is None:
            motion_ids = list(range(self._motion_lib.num_motions()))
        motion_ids = torch.as_tensor(motion_ids, device=self.device, dtype=torch.long)
        self._eval_motion_ids = motion_ids
        self._eval_cursor = 0

    def next_eval_batch_ids(self):
        """Return a tensor of shape [num_envs] for the next eval batch, using random padding (no overrun)."""
        n = int(self.num_envs)
        start = int(self._eval_cursor)
        end = min(start + n, len(self._eval_motion_ids))
        core = self._eval_motion_ids[start:end]

        # 若不满，随机从已有样本中补齐（而不是固定取前面的）
        if end - start < n:
            num_pad = n - (end - start)
            pad_idx = torch.randint(low=0, high=len(self._eval_motion_ids), size=(num_pad,), device=self.device)
            pad = self._eval_motion_ids[pad_idx]
            batch = torch.cat([core, pad], dim=0)
        else:
            batch = core

        # 更新游标（只向前推进真实未评估部分）
        self._eval_cursor = end
        done = (self._eval_cursor >= len(self._eval_motion_ids))

        return batch.to(self.device), done

    def enable_data_recording(self, enable: bool = True):
        self.is_recording_data = enable
        if enable:
            self.recorded_data = [[] for _ in range(self.num_envs)]

    def disable_data_recording(self):
        self.is_recording_data = False

        self.recorded_data = [[] for _ in range(self.num_envs)]




    # def init_pd_from_mass_matrix(self, zeta: float = 0.8):
    #     def infer_wn(name: str) -> float:
    #         if "hip_" in name: return 20.0
    #         if "knee" in name: return 18.0
    #         if "ankle" in name: return 14.0
    #         if "waist" in name: return 16.0
    #         if "shoulder" in name: return 12.0
    #         if "elbow" in name: return 10.0
    #         if "wrist" in name or "head" in name or "neck" in name:
    #             return 8.0
    #         raise ValueError(f"undefine dof name: {name}")
    #
    #     wn = torch.tensor([infer_wn(n) for n in self.dof_names],
    #                       device=self.device)
    #
    #     J_eff = torch.as_tensor(self.cfg.control.J_eff,
    #                             dtype=torch.float32,
    #                             device=self.device)
    #
    #     # ---- 3. 计算 PD 增益 ----
    #     self.p_gains = (J_eff * wn ** 2).detach()
    #     self.d_gains = (2 * zeta * J_eff * wn).detach()
    #
    #     self.p_gains.requires_grad = False
    #     self.d_gains.requires_grad = False


@torch.jit.script
def compute_humanoid_observations_jit(
    base_pos: Tensor,
    base_quat: Tensor,
    body_pos: Tensor,
    body_rot: Tensor,
    body_vel: Tensor,
    body_ang_vel: Tensor,
    activate_quat_to_tan_norm: bool = True
) -> Tensor:
    root_h = base_pos[:, 2:3]
    heading_rot_inv = calc_heading_quat_inv(base_quat)
    heading_rot_inv_expand = heading_rot_inv.unsqueeze(1).expand(-1, body_pos.shape[1], -1)
    root_base_expand = base_pos.unsqueeze(1).expand(-1, body_pos.shape[1], -1)
    local_body_pos = quat_apply(heading_rot_inv_expand, body_pos - root_base_expand)[:, 1:].reshape(base_pos.shape[0], -1)
    local_body_rot = quat_mul(heading_rot_inv_expand, body_rot).reshape(base_pos.shape[0], -1)
    local_body_rot = quat_to_tan_norm(local_body_rot.view(-1, 4)).view(base_pos.shape[0], -1)
    local_body_vel = quat_apply(heading_rot_inv_expand, body_vel).reshape(base_pos.shape[0], -1)
    local_body_ang_vel = quat_apply(heading_rot_inv_expand, body_ang_vel).reshape(base_pos.shape[0], -1)
    return torch.cat((root_h, local_body_pos, local_body_rot, local_body_vel, local_body_ang_vel), dim=-1)


@torch.jit.script
def compute_amp_observations_jit(
    base_pos: Tensor,
    base_quat: Tensor,
    base_lin_vel: Tensor,
    base_ang_vel: Tensor,
    dof_pos: Tensor,
    dof_vel: Tensor,
    key_pos: Tensor,
) -> Tensor:
    #     obs_list += [root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos]
    #     # 1? + 6 + 3 + 3 + 114 + 57 + 12
    root_h = base_pos[:, 2:3]
    heading_rot_inv = calc_heading_quat_inv(base_quat)
    heading_rot_inv_expand = heading_rot_inv.unsqueeze(1).expand(-1, key_pos.shape[1], -1)

    root_base_expand = base_pos.unsqueeze(1).expand(-1, key_pos.shape[1], -1)
    local_key_pos = quat_apply(heading_rot_inv_expand, key_pos - root_base_expand).reshape(base_pos.shape[0], -1)

    local_root_rot = quat_mul(heading_rot_inv, base_quat).reshape(base_pos.shape[0], -1)
    local_root_rot = quat_to_tan_norm(local_root_rot.view(-1, 4)).view(base_pos.shape[0], -1)

    local_root_vel = quat_apply(heading_rot_inv, base_lin_vel).reshape(base_pos.shape[0], -1)
    local_root_ang_vel = quat_apply(heading_rot_inv, base_ang_vel).reshape(base_pos.shape[0], -1)

    dof_pos = quat_to_tan_norm(exp_map_to_quat(dof_pos.reshape(-1, 3))).reshape(base_pos.shape[0], -1)

    obs_list = []
    obs_list += [root_h, local_root_rot, local_root_vel, local_root_ang_vel, dof_pos, dof_vel, local_key_pos]
    # 1 + 6 + 3 + 3 + 69 * 2 + 69 + 4 * 3 = 1 + 6 + 3 + 3 + 138 + 69 + 12 = 13 + 81 + 138 =  219 + 13 = 232
    obs = torch.cat(obs_list, dim=-1)
    return obs

@torch.jit.script
def compute_mimic_observations_jit(
    base_pos: Tensor,             # [N, 3]
    base_quat: Tensor,            # [N, 4]
    body_pos: Tensor,              # [N, K, 3]
    body_rot: Tensor,              # [N, K, 4]
    body_vel: Tensor,              # [N, K, 3]
    body_ang_vel: Tensor,          # [N, K, 3]
    ref_body_pos: Tensor,          # [N, K, 3]
    ref_body_rot: Tensor,          # [N, K, 4]
    ref_body_vel: Tensor,          # [N, K, 3]
    ref_body_ang_vel: Tensor,      # [N, K, 3]
    activate_quat_to_tan_norm: bool = True
) -> Tensor:
    # 引用 jit-safe 工具函数
    heading_rot_inv = calc_heading_quat_inv(base_quat)         # [N, 4]
    heading_rot = calc_heading_quat(base_quat)                 # [N, 4]
    N, K, _ = body_pos.shape

    heading_rot_expand = heading_rot.unsqueeze(1).expand(N, K, 4)
    heading_rot_inv_expand = heading_rot_inv.unsqueeze(1).expand(N, K, 4)

    diff_global_body_pos = ref_body_pos - body_pos
    diff_local_body_pos_flat = quat_apply(heading_rot_inv_expand, diff_global_body_pos).reshape(N, -1)

    diff_global_body_rot = quat_mul(ref_body_rot, quat_conjugate(body_rot))
    diff_local_body_rot_flat = quat_mul(
        quat_mul(heading_rot_inv_expand, diff_global_body_rot),
        heading_rot_expand
    ).reshape(N, -1)
    if activate_quat_to_tan_norm:
        diff_local_body_rot_flat = quat_to_tan_norm(diff_local_body_rot_flat.view(-1, 4)).view(N, -1)

    diff_global_body_vel = ref_body_vel - body_vel
    diff_local_body_vel_flat = quat_apply(heading_rot_inv_expand, diff_global_body_vel).reshape(N, -1)

    diff_global_body_ang_vel = ref_body_ang_vel - body_ang_vel
    diff_local_body_ang_vel_flat = quat_apply(heading_rot_inv_expand, diff_global_body_ang_vel).reshape(N, -1)

    local_ref_body_pos = ref_body_pos - base_pos.unsqueeze(1).expand(N, K, 3)
    local_ref_body_pos = quat_apply(heading_rot_inv_expand, local_ref_body_pos).reshape(N, -1)

    local_ref_body_rot = quat_mul(heading_rot_inv_expand, ref_body_rot).reshape(N, -1)
    if activate_quat_to_tan_norm:
        local_ref_body_rot = quat_to_tan_norm(local_ref_body_rot.view(-1, 4)).view(N, -1)

    obs = torch.cat([
        diff_local_body_pos_flat,
        diff_local_body_rot_flat,
        diff_local_body_vel_flat,
        diff_local_body_ang_vel_flat,
        local_ref_body_pos,
        local_ref_body_rot
    ], dim=-1)

    return obs

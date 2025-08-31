
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from videoskills import LEGGED_GYM_ROOT_DIR
from videoskills.envs.base.legged_robot_config import LeggedRobotCfg
from videoskills.envs.base.legged_robot import LeggedRobot
from videoskills.utils.poselib.skeleton.skeleton3d import SkeletonTree
from videoskills.utils.motionlib.motion_lib import MotionLib
from videoskills.utils.torch_utils import to_torch, quat_mul, quat_conjugate, quat_to_angle_axis
from videoskills.utils.torch_utils import calc_heading_quat_inv, calc_heading_quat, quat_apply, quat_to_tan_norm
from videoskills.utils.torch_utils import exp_map_to_quat
from torch import Tensor
from videoskills.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor


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

        self.activate_amp = self.cfg.amp.activate
        if self.activate_amp:
            self.num_amp_obs_steps = self.cfg.amp.num_amp_obs_steps
            self.num_amp_obs_per_step = self.cfg.amp.num_amp_obs
            self._amp_obs_buf = torch.zeros((self.cfg.env.num_envs, self.num_amp_obs_steps, self.num_amp_obs_per_step),
                                            device=sim_device, dtype=torch.float)
            self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
            self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

            self._amp_obs_demo_buf = None

        self._parse_cfg(self.cfg)

        self.early_termination_buf = torch.zeros(self.cfg.env.num_envs, device=sim_device, dtype=torch.bool)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if isinstance(self.cfg.motion.file, list):
            motion_file = self.cfg.motion.file
        else:
            motion_file = self.cfg.motion.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self._load_motion(motion_file)
        self._initialize_motion_offsets()

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

        if self.activate_amp:
            key_body_ids = []
            for body_name in self.cfg.motion.key_bodies:
                body_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], body_name)
                assert (body_id != -1)
                key_body_ids.append(body_id)

            self.key_body_ids = key_body_ids

        # Test
        body_props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.robot_handles[0])  # 获取刚体属性
        total_mass = 0.0
        for i, prop in enumerate(body_props):
            # print(f"Body {i} mass:", prop.mass)
            total_mass += prop.mass
        print("Total mass of the robot:", total_mass)

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
            self.gym.clear_lines(self.viewer)
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

                    gymutil.draw_lines(sphere, self.gym, self.viewer, env_ptr, T)
        super().render()


    # TODO: 重写 reset dof 和 load motion， 把 root 和 dof 和在一起
    def _load_motion(self, motion_file):

        self._motion_lib = MotionLib(motion_file=motion_file,
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
        # reset robot states
        self._reset_robot(env_ids)
        self._reset_env_tensors(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        # self.feet_air_time[env_ids] = 0.  # good idea!
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        self.extras["early_termination_buf"] = self.early_termination_buf
        self.extras["recorded_data"] = [[] for _ in range(self.num_envs)]
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids])
            # self.extras["episode"]['rew_' + key] = torch.mean(
            #     self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self._refresh_sim_tensors()

        if self.activate_amp:
            self._init_amp_obs_ref(env_ids)

    def _init_amp_obs_ref(self, env_ids):
        dt = self.dt

        time_steps = -dt * (torch.arange(0, self.num_amp_obs_steps - 1, device=self.device) + 1)
        expanded_motion_ids = torch.tile(self._sampled_motion_ids[env_ids].unsqueeze(1),
                                         (1, self.num_amp_obs_steps - 1)).reshape(-1)

        motion_times = self._motion_start_times[env_ids].view(-1, 1) + time_steps.view(1, -1)
        motion_times = motion_times.view(-1)

        motion_state = self._motion_lib.get_motion_state(expanded_motion_ids, motion_times)

        key_pos = motion_state["key_pos"][:, self.key_body_ids, :]
        amp_obs_demo = compute_amp_observations_jit(motion_state["root_pos"], motion_state["root_rot"],
                                                    motion_state["root_vel"], motion_state["root_ang_vel"],
                                                    motion_state["dof_pos"], motion_state["dof_vel"],
                                                    key_pos)

        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)


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
            self._reset_hybrid_state_init(env_ids)

        self.motion_lengths = self._motion_lib.get_motion_length(self._sampled_motion_ids[env_ids])/ self.dt
        # from 0, therefore the real length is (int(motion_lengths) + 1)


    def _reset_default(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.dof_vel[env_ids] = 0.
        self._reset_default_env_ids = env_ids

        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        # no reference motion, reset tracking buffers
        self._sampled_motion_ids[env_ids] = 0
        self._motion_start_times[env_ids] = 0.0

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_lib.sample_motions(num_envs)
        self._sampled_motion_ids[env_ids] = motion_ids

        if (self._state_init == 'random'
                or self._state_init == 'hybrid'):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == 'start'):
            motion_times = torch.zeros(num_envs, device=self.device)
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
                            key_ang_vel=motion_state["key_ang_vel"])

        self._motion_start_times[env_ids] = motion_times

    # def _reward_action_rate(self):
    #     diff = self.actions - self.last_actions
    #     return -self.cfg.rewards.w_act_rate * torch.sum(diff ** 2, dim=1)

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,
                       key_pos, key_rot, key_vel, key_ang_vel):
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:10] = root_vel
        self.root_states[env_ids, 10:13] = root_ang_vel

        # self.base_pos[:] = self.root_states[:, 0:3]
        # self.base_quat[:] = self.root_states[:, 3:7]

        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        # self.body_pos, self.body_rot, self.body_vel, self.body_ang_vel,
        self.body_pos[env_ids,:] = key_pos
        self.body_rot[env_ids,:] = key_rot
        self.body_vel[env_ids,:] = key_vel
        self.body_ang_vel[env_ids,:] = key_ang_vel

        return

    def check_termination(self):
        # 这里并没有取分 early termination 和 time out
        """ Check if environments need to be reset
        """
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        # fall = torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)  # raw pitch yaw

        time_out = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        progress = (self.episode_length_buf.to(torch.float) + 1) * self.dt  # progress is the current ref_motion
        motion_lens = self._motion_lib.get_motion_length(self._sampled_motion_ids)
        ref_out = (progress + self._motion_start_times )> motion_lens
        body_delta_sq = torch.sum((self.body_pos[:,self.reset_body_id]
                                   - self.ref_body_pos[:, self.reset_body_id]) ** 2, dim=2)  # → ℝ[num_envs, K]
        # 只要任何一个关键点 > 0.5 m 就触发
        body_too_far = torch.any(body_delta_sq > self.early_termination_distance[self.reset_body_id], dim=1)  # → ℝ[num_envs]
        body_too_far *= (self.episode_length_buf > 1)
        # --------- ③ 汇总三个条件 ----------
        # self.reset_buf = fall | time_out | body_too_far
        self.reset_buf = ref_out | body_too_far | time_out
        self.time_out_buf = time_out | ref_out
        self.early_termination_buf = body_too_far

        if not self.early_termination:
            self.reset_buf = time_out | ref_out
            self.time_out_buf = time_out | ref_out

        if self.eval_mode:
            self.reset_buf = ref_out | body_too_far
            self.time_out_buf = time_out | ref_out

    def _reset_env_tensors(self, env_ids):
        # here dof_pos and dof_vel is view of dof_state
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # if self.drive_mode == gymapi.DOF_MODE_POS:
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
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

        humanoid_obs = compute_humanoid_observations_jit(self.base_pos, self.base_quat,
                                self.body_pos, self.body_rot, self.body_vel, self.body_ang_vel,
                                activate_quat_to_tan_norm=self.activate_quat_to_tan_norm)

        task_obs = compute_task_observations_jit(self.base_pos, self.base_quat,
                               self.body_pos, self.body_rot, self.body_vel, self.body_ang_vel,
                               self.ref_body_pos, self.ref_body_rot, self.ref_body_vel, self.ref_body_ang_vel,
                               activate_quat_to_tan_norm=self.activate_quat_to_tan_norm)

        self.obs_buf = torch.cat((humanoid_obs, task_obs, self.actions), dim=-1)
        # self.obs_buf = torch.cat((humanoid_obs, task_obs), dim=-1)

        if self.activate_amp:
            key_body_pos = self.body_pos[:, self.key_body_ids, :]
            self._curr_amp_obs_buf[:] = compute_amp_observations_jit(self.base_pos, self.base_quat, self.base_lin_vel,
                              self.base_ang_vel, self.dof_pos, self.dof_vel, key_body_pos)
            self.extras["amp_state"] = self._amp_obs_buf.clone().reshape(self.num_envs, -1)

            self._update_hist_amp_obs()


    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            try:
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0:(self.num_amp_obs_steps - 1)]
            except:
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0:(self.num_amp_obs_steps - 1)].clone()
        else:
            self._hist_amp_obs_buf[env_ids] = self._amp_obs_buf[env_ids, 0:(self.num_amp_obs_steps - 1)]
        return

    def fetch_amp_obs_demo(self, num_samples):
        # Creates the reference motion amp obs. For discrinminiator

        dt = self.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)
        motion_times = self._motion_lib.sample_time(motion_ids)

        time_steps = -dt * (torch.arange(0, self.num_amp_obs_steps, device=self.device))
        expanded_motion_ids = torch.tile(motion_ids,(1, self.num_amp_obs_steps)).reshape(-1)

        motion_times = motion_times.view(-1, 1) + time_steps.view(1, -1)
        motion_times = motion_times.view(-1)

        motion_state = self._motion_lib.get_motion_state(expanded_motion_ids, motion_times)

        key_pos = motion_state["key_pos"][:, self.key_body_ids, :]
        amp_obs_demo = compute_amp_observations_jit(motion_state["root_pos"], motion_state["root_rot"],
                                                    motion_state["root_vel"], motion_state["root_ang_vel"],
                                                    motion_state["dof_pos"], motion_state["dof_vel"],
                                                    key_pos)

        amp_obs_demo_flat = amp_obs_demo.view(num_samples, -1)

        return amp_obs_demo_flat

    def compute_humanoid_observations(self):
        root_h = self.base_pos[:, 2:3]
        heading_rot_inv = calc_heading_quat_inv(self.base_quat)
        heading_rot_inv_expand = heading_rot_inv.unsqueeze(1).expand(-1, self.body_pos.shape[1], -1)
        root_base_expand = self.base_pos.unsqueeze(1).expand(-1, self.body_pos.shape[1], -1)  # [N, K, 3]
        local_body_pos = quat_apply(heading_rot_inv_expand, self.body_pos - root_base_expand)[:,1:].view(self.num_envs, -1)
        local_body_rot = quat_mul(heading_rot_inv_expand, self.body_rot).view(self.num_envs, -1)
        if self.activate_quat_to_tan_norm:
            local_body_rot = quat_to_tan_norm(local_body_rot.view(-1, 4)).view(self.num_envs, -1) # [N, K, 4]
        local_body_vel = quat_apply(heading_rot_inv_expand, self.body_vel).view(self.num_envs, -1)
        local_body_ang_vel = quat_apply(heading_rot_inv_expand, self.body_ang_vel).view(self.num_envs, -1)

        # 1 +  3 * (K - 1) + 4 * K + 3 * K + 3 * K = 1 + 3K + 4K + 3K + 3K - 3 = 13 * 24 - 2  # 310
        obs = torch.cat((root_h, local_body_pos, local_body_rot, local_body_vel, local_body_ang_vel), dim=-1)
        return obs

    def compute_task_observations(self):
        obs = []
        heading_rot_inv = calc_heading_quat_inv(self.base_quat)
        heading_rot = calc_heading_quat(self.base_quat)
        heading_rot_expand = heading_rot.unsqueeze(1).expand(-1, self.body_pos.shape[1], -1)
        heading_rot_inv_expand = heading_rot_inv.unsqueeze(1).expand(-1, self.body_pos.shape[1], -1)

        diff_global_body_pos = self.ref_body_pos - self.body_pos
        diff_local_body_pos_flat = quat_apply(heading_rot_inv_expand, diff_global_body_pos).view(self.num_envs, -1)

        # 这个应该相同才对吧，目前不同，反倒是 diff_global_body_rot 相同 # 解决了！
        diff_global_body_rot = quat_mul(self.ref_body_rot, quat_conjugate(
            self.body_rot))
        diff_local_body_rot_flat =quat_mul(
            quat_mul(heading_rot_inv_expand, diff_global_body_rot),
            heading_rot_expand).view(self.num_envs, -1)
        if self.activate_quat_to_tan_norm:
            diff_local_body_rot_flat = quat_to_tan_norm(diff_local_body_rot_flat.view(-1, 4)).view(self.num_envs, -1)

        diff_global_body_vel = self.ref_body_vel - self.body_vel
        diff_local_body_vel_flat = quat_apply(heading_rot_inv_expand, diff_global_body_vel).view(self.num_envs, -1)

        diff_global_body_ang_vel = self.ref_body_ang_vel - self.body_ang_vel
        diff_local_body_ang_vel_flat = quat_apply(heading_rot_inv_expand, diff_global_body_ang_vel).view(self.num_envs, -1)

        local_ref_body_pos = self.ref_body_pos - self.base_pos.unsqueeze(1).expand(-1, self.ref_body_pos.shape[1], -1)
        local_ref_body_pos = quat_apply(heading_rot_inv_expand, local_ref_body_pos).view(self.num_envs, -1)
        # 这里的 local_ref_body_rot 是在 heading frame 下的，而不是相对父节点的！
        local_ref_body_rot = quat_mul(heading_rot_inv_expand, self.ref_body_rot).view(
            self.num_envs, -1)
        if self.activate_quat_to_tan_norm:
            local_ref_body_rot = quat_to_tan_norm(local_ref_body_rot.view(-1, 4)).view(self.num_envs, -1)

        obs.append(diff_local_body_pos_flat)  # 3
        obs.append(diff_local_body_rot_flat)  # 4
        obs.append(diff_local_body_vel_flat)  # 3
        obs.append(diff_local_body_ang_vel_flat) # 3
        obs.append(local_ref_body_pos) # 3
        obs.append(local_ref_body_rot) # 4

        obs = torch.cat(obs, dim=-1)

        return obs

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


    # def _reward_torques(self):
    #     # Penalize torques
    #     return torch.log(1 + self.cfg.rewards.alpha_torques * torch.sum(torch.square(self.torques), dim=1))


    def _initialize_motion_offsets(self):
        """
        For each environment, generate a random heading rotation and compute:
        - rotation_offset: quaternion of shape [num_envs, 4]
        - pos_offset: [env_origins, env_key_pos_origins]
        """
        # Generate random heading angles in [-pi, pi]
        heading_angles = torch_rand_float(-np.pi, np.pi, (self.num_envs, 1), device=self.device)
        self.rotation_offset = torch.zeros(self.num_envs, 4, device=self.device)

        # Heading quaternion (rotation around up-axis)
        axis = torch.zeros(self.num_envs, 3, device=self.device)
        axis[:, self.up_axis_idx] = 1.0
        sin_half = torch.sin(heading_angles * 0.5)
        cos_half = torch.cos(heading_angles * 0.5)
        self.rotation_offset[:, :3] = axis * sin_half
        self.rotation_offset[:, 3] = cos_half.squeeze()

        # Positional offsets (env origins & key_pos origins already expanded properly)
        self.pos_offset = {
            "root": self.env_origins.clone(),
            "body": self.env_origins.unsqueeze(1).expand(-1, self.body_pos.shape[1], -1),
        }

    def reset_with_motion_ids(self, motion_ids):
        """ Reset all environments with given motion ids. (For Evaluation)
            This method is used to reset the environment with specific motion ids, e.g. in the training stage.
        Args:
            motion_ids (torch.Tensor): Tensor of shape [num_envs] containing motion ids to reset the environments with.
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._sampled_motion_ids = motion_ids
        self._motion_start_times = torch.zeros(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        self.gym.clear_lines(self.viewer)
        # reset robot states
        motion_times = torch.zeros(self.num_envs, device=self.device)
        motion_state = self._motion_lib.get_motion_state(self._sampled_motion_ids[env_ids], motion_times)

        self._set_env_state(env_ids=env_ids,
                            root_pos=motion_state["root_pos"] +  self.pos_offset['root'][env_ids],
                            root_rot=motion_state["root_rot"],
                            dof_pos=motion_state["dof_pos"],
                            root_vel=motion_state["root_vel"],
                            root_ang_vel=motion_state["root_ang_vel"],
                            dof_vel=motion_state["dof_vel"],
                            key_pos=motion_state["key_pos"] +  self.pos_offset['body'][env_ids],
                            key_rot=motion_state["key_rot"],
                            key_vel=motion_state["key_vel"],
                            key_ang_vel=motion_state["key_ang_vel"])

        self._motion_start_times[env_ids] = motion_times
        self._reset_env_tensors(env_ids)
        # self._resample_commands(env_ids)

        # reset buffers
        # if self.actions.is_inference():  # PyTorch ≥2.2
        #     self.actions = self.actions.clone()
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
    activate_quat_to_tan_norm: bool = False
) -> Tensor:
    root_h = base_pos[:, 2:3]
    heading_rot_inv = calc_heading_quat_inv(base_quat)
    heading_rot_inv_expand = heading_rot_inv.unsqueeze(1).expand(-1, body_pos.shape[1], -1)
    root_base_expand = base_pos.unsqueeze(1).expand(-1, body_pos.shape[1], -1)
    local_body_pos = quat_apply(heading_rot_inv_expand, body_pos - root_base_expand)[:, 1:].reshape(base_pos.shape[0], -1)
    local_body_rot = quat_mul(heading_rot_inv_expand, body_rot).reshape(base_pos.shape[0], -1)
    if activate_quat_to_tan_norm:
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
def compute_task_observations_jit(
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
    activate_quat_to_tan_norm: bool = False
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
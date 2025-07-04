from videoskills.envs.base.legged_robot import LeggedRobot
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from videoskills import LEGGED_GYM_ROOT_DIR
from videoskills.envs.base.legged_robot_config import LeggedRobotCfg
from scripts.poselib.skeleton.skeleton3d import SkeletonTree
from videoskills.utils.motionlib.motion_lib import MotionLib
from videoskills.utils.torch_utils import to_torch, quat_mul, quat_conjugate, quat_to_angle_axis, get_axis_params
from videoskills.utils.torch_utils import calc_heading_quat_inv, calc_heading_quat, quat_apply, quat_to_tan_norm
from videoskills.utils.torch_utils import exp_map_to_quat
from torch import Tensor
import xml.etree.ElementTree as ET
import glob

class LeggedRobotImi(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        if self.cfg.dev:
            self.cfg.env.num_envs = 16
            headless = False
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False

        # quat_to_tan_norm ablation study
        self.activate_quat_to_tan_norm = True
        if self.activate_quat_to_tan_norm:
            self.cfg.env.num_observations = 358 + 576 + 69

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

        self.early_termination_distance = torch.tensor(self.cfg.early_termination.distance, device=self.device) ** 2

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
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # asset is a xml resource
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file , asset_options)

        # marker
        if not self.headless:
            self._sphere_geom = gymutil.WireframeSphereGeometry(
                radius=0.03,
                num_lats=12,  # “纬线”数
                num_lons=12,  # “经线”数
                color=(1.0, 0.2, 0.2))

        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.skeleton_tree = SkeletonTree.from_mjcf(asset_path)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_dofs = len(self.dof_names)
        self.dof_body_ids = np.arange(1, len(self.body_names)).tolist()
        self.dof_offsets = np.linspace(0, len(self.dof_names), len(self.body_names)).astype(int)
        self.cfg.init_state.default_joint_angles = {dof_name: 0.0 for dof_name in self.dof_names}

        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()

        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.robot_handles = []
        self.envs = []
        max_agg_bodies = 160
        max_agg_shapes = 160
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.gym.begin_aggregate(env_handle, max_agg_bodies, max_agg_shapes, True)
            self._build_env(i, env_handle, robot_asset)
            self.gym.end_aggregate(env_handle)
            self.envs.append(env_handle)

        # regularization?
        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.robot_handles[0])
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        for j in range(self.num_dofs):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            elif dof_prop['lower'][j] == dof_prop['upper'][j]:
                print("Warning: DOF limits are the same")
                if dof_prop['lower'][j] == 0:
                    self.dof_limits_lower.append(-np.pi)
                    self.dof_limits_upper.append(np.pi)
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.dof_pos_limits = torch.stack([self.dof_limits_lower, self.dof_limits_upper], dim=-1)
        self.torque_limits = to_torch(dof_prop['effort'], device=self.device)

        key_body_ids = []
        for body_name in self.cfg.motion.key_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], body_name)
            assert(body_id != -1)
            key_body_ids.append(body_id)

        self.body_ids = torch.arange(len(self.cfg.motion.bodies), device=self.device, dtype=torch.long)
        self.key_body_ids = key_body_ids

        # TODO： need to be test
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.robot_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.robot_handles[0],
                                                                                        termination_contact_names[i])

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



    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self.cfg.asset.self_collisions # Setting the collision filter to 0 will enable collisions between all shapes in the actor.

        start_pose = gymapi.Transform()
        char_h = 0.89
        pos = torch.tensor(get_axis_params(char_h, self.up_axis_idx)).to(self.device)
        pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(
            1)
        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # here is the instance of the humanoid asset
        robot_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter,
                                                0)
        self.gym.enable_actor_dof_force_sensors(env_ptr, robot_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, robot_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        # configure PD control method
        dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
        for i, dof_name in enumerate(self.dof_names):
            self.cfg.control.stiffness[dof_name] = torch.tensor(dof_prop['stiffness'][i] * self.cfg.asset.pd_scale, dtype=torch.float, device=self.device)
            self.cfg.control.damping[dof_name] =  torch.tensor(dof_prop['damping'][i] * self.cfg.asset.pd_scale, dtype=torch.float, device=self.device)

        self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_prop)

        filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        props = self.gym.get_actor_rigid_shape_properties(env_ptr, robot_handle)
        assert (len(filter_ints) == len(props))

        for p_idx in range(len(props)):
            props[p_idx].filter = filter_ints[p_idx]
        self.gym.set_actor_rigid_shape_properties(env_ptr, robot_handle, props)

        self.robot_handles.append(robot_handle)

        return

    # TODO: 重写 reset dof 和 load motion， 把 root 和 dof 和在一起
    def _load_motion(self, motion_file):

        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self.dof_body_ids,
                                     dof_offsets=self.dof_offsets,
                                     key_body_ids=self.body_ids,
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
        self.feet_air_time[env_ids] = 0.  # good idea!
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        self.extras["recorded_data"] = [[] for _ in range(self.num_envs)]
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
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


    def _refresh_sim_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

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


    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.body_pos = self._rigid_body_state_reshaped[..., self.body_ids, 0:3]   #3
        self.body_rot = self._rigid_body_state_reshaped[..., self.body_ids, 3:7]   #4
        self.body_vel = self._rigid_body_state_reshaped[..., self.body_ids, 7:10]   #3
        self.body_ang_vel = self._rigid_body_state_reshaped[..., self.body_ids, 10:13]  #3

        self.contact_history = torch.zeros((3, self.num_envs, self.feet_indices.shape[0]),
                                           dtype=torch.bool, device=self.device)

        self.last_landing_mask = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
                                             device=self.device)

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
                            root_pos=motion_state["root_pos"] + self.env_origins[env_ids],
                            root_rot=motion_state["root_rot"],
                            dof_pos=motion_state["dof_pos"],
                            root_vel=motion_state["root_vel"],
                            root_ang_vel=motion_state["root_ang_vel"],
                            dof_vel=motion_state["dof_vel"])

        self._reset_ref_env_ids = env_ids
        self._motion_start_times[env_ids] = motion_times


    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:10] = root_vel
        self.root_states[env_ids, 10:13] = root_ang_vel

        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel
        return

    def check_termination(self):
        # 这里并没有取分 early termination 和 time out
        """ Check if environments need to be reset
        """
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        # fall = torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)  # raw pitch yaw

        time_out = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        progress = (self.episode_length_buf.to(torch.float) + 1) * self.dt  # 在这种情况下已经找不到 ref motion 了！
        motion_lens = self._motion_lib.get_motion_length(self._sampled_motion_ids)
        ref_out = (progress + self._motion_start_times )>= motion_lens
        body_delta_sq = torch.sum((self.body_pos - self.ref_body_pos) ** 2, dim=2)  # → ℝ[num_envs, K]
        # 只要任何一个关键点 > 0.5 m 就触发
        body_too_far = torch.any(body_delta_sq > self.early_termination_distance, dim=1)  # → ℝ[num_envs]

        # --------- ③ 汇总三个条件 ----------
        # self.reset_buf = fall | time_out | body_too_far
        self.reset_buf = ref_out | body_too_far | time_out
        self.time_out_buf = time_out

        if self.eval_mode:
            progress = self.episode_length_buf.to(torch.float) * self.dt
            motion_lens = self._motion_lib.get_motion_length(self._sampled_motion_ids)
            ref_out = progress >= motion_lens
            self.reset_buf = ref_out | body_too_far

    def _reset_env_tensors(self, env_ids):
        # here dof_pos and dof_vel is view of dof_state
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def post_physics_step(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.body_pos[:] = self._rigid_body_state_reshaped[..., self.body_ids, 0:3]
        self.body_rot[:] = self._rigid_body_state_reshaped[..., self.body_ids, 3:7]
        self.body_vel[:] = self._rigid_body_state_reshaped[..., self.body_ids, 7:10]
        self.body_ang_vel[:] = self._rigid_body_state_reshaped[..., self.body_ids, 10:13]

        # self.gym.refresh_force_sensor_tensor(self.sim)
        # tau_cmd = self.dof_force_tensor
        # tau_react = self.torques
        # torque_gap = tau_cmd - tau_react

        super().post_physics_step()

        if self.cfg.env.land_event_detect:
            self.land_event_detection()


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
        motion_lens = self._motion_lib.get_motion_length(self._sampled_motion_ids)
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
                                self.body_pos, self.body_rot,self.body_vel, self.body_ang_vel,
                                activate_quat_to_tan_norm=self.activate_quat_to_tan_norm)

        task_obs = compute_task_observations_jit(self.base_pos, self.base_quat,
                               self.body_pos, self.body_rot, self.body_vel, self.body_ang_vel,
                               self.ref_body_pos, self.ref_body_rot, self.ref_body_vel, self.ref_body_ang_vel,
                               activate_quat_to_tan_norm=self.activate_quat_to_tan_norm)

        self.obs_buf = torch.cat((humanoid_obs, task_obs, self.actions), dim=-1)

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

    def _reward_imitation(self):
        # reward 是不需要 heading 归一化 的！
        """
        Computes the imitation reward based on the difference between the current and reference body positions and rotations.
        The reward is computed in the heading frame of the root body.
        """
        pos_err = torch.mean(torch.square(self.body_pos- self.ref_body_pos), dim=1).mean(-1)
        rot_diff = quat_mul(self.ref_body_rot, quat_conjugate(self.body_rot))
        diff_global_body_angle = quat_to_angle_axis(rot_diff)[0]
        rot_err = (diff_global_body_angle ** 2).mean(dim=-1)
        vel_err = torch.mean(torch.square(self.body_vel- self.ref_body_vel), dim=1).mean(-1)
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

    def _reward_power(self):
        # reward 是不需要 heading 归一化 的！
        """
        Computes the imitation reward based on the difference between the current and reference key body positions and rotations.
        The reward is computed in the heading frame of the root body.
        """
        reward = torch.abs(torch.multiply(self.torques, self.dof_vel)).sum(dim=-1)
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
                            root_pos=motion_state["root_pos"] + self.env_origins[env_ids],
                            root_rot=motion_state["root_rot"],
                            dof_pos=motion_state["dof_pos"],
                            root_vel=motion_state["root_vel"],
                            root_ang_vel=motion_state["root_ang_vel"],
                            dof_vel=motion_state["dof_vel"])

        self._reset_ref_env_ids = env_ids
        self._motion_start_times[env_ids] = motion_times
        self._reset_env_tensors(env_ids)
        self._resample_commands(env_ids)

        # reset buffers
        # if self.actions.is_inference():  # PyTorch ≥2.2
        #     self.actions = self.actions.clone()
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.  # good idea!
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
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
from videoskills.envs.base.legged_robot import LeggedRobot
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from videoskills import LEGGED_GYM_ROOT_DIR
from videoskills.envs.base.legged_robot_config import LeggedRobotCfg
from videoskills.utils.motionlib.motion_lib_smpl import MotionLibSMPL
from scripts.poselib.skeleton.skeleton3d import SkeletonTree
from videoskills.utils.motionlib.motion_lib import MotionLib

class SMPLRobot(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        # define motionlib and motion sampling buf
        motion_file = self.cfg.motion.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self._load_motion(motion_file)

        self._motion_start_times = torch.zeros(self.num_envs).to(self.device)
        self._sampled_motion_ids = torch.zeros(self.num_envs).long().to(self.device)
        self._state_init = self.cfg.init_state.type


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
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.skeleton_tree = SkeletonTree.from_mjcf(asset_path)


        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        self.dof_body_ids = np.arange(1, len(self.body_names))
        self.dof_offsets = np.linspace(0, len(self.dof_names) * 3, len(self.body_names)).astype(int)


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
        self.actor_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self._build_env(i, env_handle, self.robot_asset)
            self.gym.end_aggregate(env_handle)
            self.envs.append(env_handle)

        # regularization?
        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        for j in range(self.num_dof):

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
        self.dof_limits = torch.stack([self.dof_limits_lower, self.dof_limits_upper], dim=-1)
        self.torque_limits = to_torch(dof_prop['effort'], device=self.device)

        key_body_ids = []
        for body_name in self.cfg.motion.keybodys:
            body_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], body_name)
            assert(body_id != -1)
            key_body_ids.append(body_id)

        self.key_body_ids = np.array(key_body_ids, dtype=np.int32)

        # TODO： need to be test
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        self.humanoid_handles = []
        col_group = env_id
        col_filter = 0 # Setting the collision filter to 0 will enable collisions between all shapes in the actor.

        start_pose = gymapi.Transform()
        char_h = 0.89
        pos = torch.tensor(get_axis_params(char_h, self.up_axis_idx)).to(self.device)
        pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(
            1)
        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # here is the instance of the humanoid asset
        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter,
                                                0)
        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)

        # control mode is isaac_pd
        self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
        assert (len(filter_ints) == len(props))

        for p_idx in range(len(props)):
            props[p_idx].filter = filter_ints[p_idx]
        self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

        self.humanoid_handles.append(humanoid_handle)

        return

    # TODO: 重写 reset dof 和 load motion， 把 root 和 dof 和在一起
    def _load_motion(self, motion_file):
        self._dof_body_ids = self.cfg.motion

        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self.dof_body_ids,
                                     dof_offsets=self.dof_offsets,
                                     key_body_ids=self.key_body_ids,
                                     device=self.device)


    def reset_idx(self, env_ids):
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

        # reset robot states
        self._reset_robot(env_ids)
        self._reset_env_tensors(env_ids)
        self._resample_commands(env_ids)

        # reset buffers
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


    def _reset_default(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.dof_vel[env_ids] = 0.
        self._reset_default_env_ids = env_ids

        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)

        if (self._state_init == 'random'
                or self._state_init == 'hybrid'):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == 'start'):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=dof_pos,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:10] = root_vel
        self.root_states[env_ids, 10:13] = root_ang_vel

        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel
        return


    def _reset_env_tensors(self, env_ids):
        # here dof_pos and dof_vel is view of dof_state
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


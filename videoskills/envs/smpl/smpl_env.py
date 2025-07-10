from isaacgym import gymapi
import torch
from videoskills.envs.base.legged_robot_imi import LeggedRobotImi


class SMPLRobot(LeggedRobotImi):
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self.cfg.asset.self_collisions # Setting the collision filter to 0 will enable collisions between all shapes in the actor.

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*(self.base_init_state[:3] + self.env_origins[env_id]))
        start_pose.r = gymapi.Quat(*self.base_init_state[3:6])

        # here is the instance of the humanoid asset
        robot_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter,
                                                0)
        self.gym.enable_actor_dof_force_sensors(env_ptr, robot_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, robot_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        self.cfg.control.stiffness = {}
        self.cfg.control.damping = {}
        # configure PD control method
        dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
        for i, dof_name in enumerate(self.dof_names):
            self.cfg.control.stiffness[dof_name] = torch.tensor(dof_prop['stiffness'][i] * self.cfg.control.pd_scale, dtype=torch.float, device=self.device)
            self.cfg.control.damping[dof_name] =  torch.tensor(dof_prop['damping'][i] * self.cfg.control.pd_scale, dtype=torch.float, device=self.device)

        self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_prop)

        filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        props = self.gym.get_actor_rigid_shape_properties(env_ptr, robot_handle)

        assert (len(filter_ints) == len(props))

        for p_idx in range(len(props)):
            props[p_idx].filter = filter_ints[p_idx]
        self.gym.set_actor_rigid_shape_properties(env_ptr, robot_handle, props)

        self.robot_handles.append(robot_handle)

        return



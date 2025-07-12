from isaacgym import gymapi
import torch
from videoskills.envs.base.legged_robot_imi import LeggedRobotImi


class G1Robot(LeggedRobotImi):
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self.cfg.asset.self_collisions # Setting the collision filter to 0 will enable collisions between all shapes in the actor.

        start_pose = gymapi.Transform()
        # 检查一下是不是这里有问题
        start_pose.p = gymapi.Vec3(*(self.base_init_state[:3] + self.env_origins[env_id]))
        start_pose.r = gymapi.Quat(*self.base_init_state[3:7])

        # here is the instance of the humanoid asset
        robot_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter,
                                                0)

        self.gym.enable_actor_dof_force_sensors(env_ptr, robot_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, robot_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)

        # dof_prop["stiffness"] = torch.tensor(self.stiffness, dtype=torch.float, device=self.device)
        # dof_prop["damping"] =  torch.tensor(self.damping, dtype=torch.float, device=self.device)

        # self.cfg.control.stiffness = {}
        # self.cfg.control.damping = {}
        # for i, dof_name in enumerate(self.dof_names):
        #     self.cfg.control.stiffness[dof_name] = torch.tensor(dof_prop['stiffness'][i] * self.cfg.control.pd_scale, dtype=torch.float, device=self.device)
        #     self.cfg.control.damping[dof_name] =  torch.tensor(dof_prop['damping'][i] * self.cfg.control.pd_scale, dtype=torch.float, device=self.device)

        self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_prop)

        props = self.gym.get_actor_rigid_shape_properties(env_ptr, robot_handle)
        body2shape = self.gym.get_actor_rigid_body_shape_indices(env_ptr, robot_handle)
        body_names = self.body_names

        # 遍历每个 body
        for body_id, index_range in enumerate(body2shape):
            name = body_names[body_id]

            for shape_idx in range(index_range.start, index_range.start + index_range.count):
                if 'right' in name:
                    if 'ankle' in name:
                        props[shape_idx].filter = 2
                    elif 'knee' in name:
                        props[shape_idx].filter = 6
                    elif 'hip' in name:
                        props[shape_idx].filter = 12
                if 'left' in name:
                    if 'ankle' in name:
                        props[shape_idx].filter = 16
                    elif 'knee' in name:
                        props[shape_idx].filter = 48
                    elif 'hip' in name:
                        props[shape_idx].filter = 96

        self.gym.set_actor_rigid_shape_properties(env_ptr, robot_handle, props)

        self.robot_handles.append(robot_handle)

        return

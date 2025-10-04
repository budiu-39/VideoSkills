from isaacgym import gymapi
import torch
from videoskills.envs.base.legged_robot_imi import LeggedRobotImi
from videoskills.envs.base.legged_robot_hoi import LeggedRobotHoi


class SMPLRobot(LeggedRobotImi):

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        if self.cfg.asset.self_collisions:
            robot_handle = self.robot_handles[env_id]
            filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            props = self.gym.get_actor_rigid_shape_properties(env_ptr, robot_handle)

            assert (len(filter_ints) == len(props))

            for p_idx in range(len(props)):
                props[p_idx].filter = filter_ints[p_idx]
            self.gym.set_actor_rigid_shape_properties(env_ptr, robot_handle, props)

        return



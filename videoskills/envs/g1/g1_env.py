from isaacgym import gymapi
import torch
from videoskills.envs.base.legged_robot_imi import LeggedRobotImi
import os
import numpy as np
from isaacgym import gymapi, gymutil
from videoskills.utils.torch_utils import to_torch
from videoskills import LEGGED_GYM_ROOT_DIR


class G1Robot(LeggedRobotImi):
    def _build_env(self, env_id, env_ptr, robot_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        robot_handle = self.robot_handles[env_id]
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


        return

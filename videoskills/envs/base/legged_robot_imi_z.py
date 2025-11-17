
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from videoskills import LEGGED_GYM_ROOT_DIR
from videoskills.envs.base.legged_robot_config import LeggedRobotCfg
from videoskills.envs.base.legged_robot import LeggedRobot
from videoskills.utils.motion_lib import MotionLib
from videoskills.utils.torch_utils import to_torch, quat_mul, quat_conjugate, quat_to_angle_axis
from videoskills.utils.torch_utils import calc_heading_quat_inv, calc_heading_quat, quat_apply, quat_to_tan_norm
from videoskills.utils.torch_utils import exp_map_to_quat
from videoskills.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from torch import Tensor
from videoskills.envs.base.legged_robot_imi import LeggedRobotImi

class LeggedRobotImiZ(LeggedRobotImi):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._z_provider = None

    def set_z_provider(self, fn):
        self._z_provider = fn



from videoskills import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from .h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from .h1.h1_env import H1Robot
from .g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from .g1.g1_env import G1Robot
from .smpl.smpl_config_attention import SMPLRobotCfg, SMPLRoughCfgPPO
from .smpl.smpl_env import SMPLRobot
from .base.legged_robot import LeggedRobot
from .smplx.smplx_config import SMPLXRobotCfg, SMPLXRoughCfgPPO
from .smplx.smplx_env import SMPLXRobot

from videoskills.utils.task_registry import task_registry

task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register( "smpl", SMPLRobot, SMPLRobotCfg(), SMPLRoughCfgPPO())
task_registry.register( "smplx", SMPLXRobot, SMPLXRobotCfg(), SMPLXRoughCfgPPO())

from videoskills import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from videoskills.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from videoskills.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from videoskills.envs.h1.h1_env import H1Robot
from videoskills.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from videoskills.envs.h1_2.h1_2_env import H1_2Robot
from videoskills.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from videoskills.envs.g1.g1_env import G1Robot
from videoskills.envs.smpl.smpl_config import SMPLRobotCfg, SMPLRoughCfgPPO
from videoskills.envs.smpl.smpl_env import SMPLRobot
from .base.legged_robot import LeggedRobot


from videoskills.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register( "smpl", SMPLRobot, SMPLRobotCfg(), SMPLRoughCfgPPO())

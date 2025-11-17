
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
        self._proprio_dim = 358

    def set_z_decoder(self, fn):
        self._z_decoder = fn

    def set_z_prior(self, fn):
        self._z_prior = fn

    @torch.no_grad()
    def compute_z_action(
        self,
        z_res: torch.Tensor,
        use_prior: bool = True,
        sample: bool = True,
    ):
        """
        obs: [B, obs_dim] —— PPO policy 的输入 obs
        z_res: [B, z_dim] —— actor_critic 输出的 residual latent
        返回: act, extra
        """
        assert self._z_decoder is not None, "z_decoder 未设置"
        proprio = self.obs_buf[:, :self._proprio_dim]  # [B, proprio_dim]

        mu_p = lv_p = None
        if use_prior:
            assert self._z_prior is not None, "use_prior=True 但 z_prior 未设置"
            mu_p, lv_p = self._z_prior(proprio)      # [B, z_dim], [B, z_dim]
            z = mu_p + z_res
        else:
            z = z_res

        # decoder: (proprio, z) -> (mu_a, log_std_a)
        decoder_obs = torch.cat([proprio, z], dim=-1)
        mu_a, log_std_a = self._z_decoder(decoder_obs)
        if sample:
            std_a = log_std_a.exp()
            eps = torch.randn_like(std_a)
            act = mu_a + eps * std_a
        else:
            act = mu_a

        extra = {
            "mu_p": mu_p,
            "lv_p": lv_p,
            "z": z,
            "mu_a": mu_a,
            "log_std_a": log_std_a,
        }
        return act, extra







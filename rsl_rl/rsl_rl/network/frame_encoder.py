import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameEncoderMLP(nn.Module):
    """
    q(z | obs_full) —— 帧级编码器，输入整段 obs_buf，输出 z 的高斯参数 (mu_q, logvar_q) 和采样 z。
    """
    def __init__(self, in_dim: int, d_z: int = 64, hidden=(512, 512)):
        super().__init__()
        self.d_z = d_z
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.GELU()]
            last = h
        self.mlp = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, d_z)
        self.lv_head = nn.Linear(last, d_z)

    def forward(self, obs_full: torch.Tensor):
        x = self.mlp(obs_full)
        mu_q = self.mu_head(x)
        lv_q = self.lv_head(x).clamp(min=-10.0, max=10.0)
        std_q = torch.exp(0.5 * lv_q)
        # reparameterization
        eps = torch.randn_like(std_q)
        z = mu_q + eps * std_q
        return z, mu_q, lv_q


class PriorMLP(nn.Module):
    """
    p(z | humanoid_obs) —— 先验网络，仅吃 humanoid_obs。
    输出 (mu_p, logvar_p)。
    """
    def __init__(self, in_dim: int, d_z: int = 64, hidden=(256, 256)):
        super().__init__()
        self.d_z = d_z
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.GELU()]
            last = h
        self.mlp = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, d_z)
        self.lv_head = nn.Linear(last, d_z)

    def forward(self, humanoid_obs: torch.Tensor):
        x = self.mlp(humanoid_obs)
        mu_p = self.mu_head(x)
        lv_p = self.lv_head(x).clamp(min=-10.0, max=10.0)
        return mu_p, lv_p


def kl_normal(mu_q, lv_q, mu_p, lv_p):
    """ KL( q || p ) for diagonal Gaussians """
    # 0.5 * sum( log(sig_p^2/sig_q^2) + (sig_q^2 + (mu_q - mu_p)^2)/sig_p^2 - 1 )
    var_q = torch.exp(lv_q)
    var_p = torch.exp(lv_p)
    term = (lv_p - lv_q) + (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8) - 1.0
    return 0.5 * term.sum(dim=-1)

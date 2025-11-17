import torch
import torch.nn as nn
from rsl_rl.utils.running_mean_std import RunningMeanStd

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


# 需要考虑 rms 的 训练/推理设置
class FramePrior(nn.Module):
    def __init__(self, in_dim: int, d_z: int = 32, hidden=(256, 256), use_rms=True):
        super().__init__()
        self.d_z = d_z
        self.use_rms = use_rms

        if use_rms:
            self.obs_rms = RunningMeanStd((in_dim,))
        else:
            self.obs_rms = None

        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.GELU()]
            last = h
        self.mlp = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, d_z)
        self.lv_head = nn.Linear(last, d_z)

    def forward(self, humanoid_obs: torch.Tensor):
        if self.obs_rms is not None:
            _ = self.obs_rms(humanoid_obs)
            obs = self.obs_rms(humanoid_obs)
        else:
            obs = humanoid_obs

        x = self.mlp(obs)
        mu_p = self.mu_head(x)
        lv_p = self.lv_head(x).clamp(min=-10.0, max=10.0)
        return mu_p, lv_p

# 这个可以当作是临时方案，确定了 decoder 结构后可以改掉
class FrameDecoder(nn.Module):
    def __init__(self, backbone, d_model, action_dim):
        super().__init__()
        self.backbone = backbone
        self.mu_head  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, action_dim))
        self.logstd_head = nn.Linear(d_model, action_dim)
        # optional initialization（可以和 decoder 共享风格）
        nn.init.uniform_(self.mu_head[-1].weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.mu_head[-1].bias)
        nn.init.uniform_(self.logstd_head.weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.logstd_head.bias)

    def forward(self, x):
        # x = torch.cat([proprio, z], dim=-1)
        feats = self.backbone(x)
        mu_pre = self.mu_head(feats)
        mu = torch.tanh(mu_pre)
        log_std = self.logstd_head(feats).clamp(min=-10.0, max=2.0)
        return mu, log_std



def kl_normal(mu_q, lv_q, mu_p, lv_p):
    """ KL( q || p ) for diagonal Gaussians """
    # 0.5 * sum( log(sig_p^2/sig_q^2) + (sig_q^2 + (mu_q - mu_p)^2)/sig_p^2 - 1 )
    var_q = torch.exp(lv_q)
    var_p = torch.exp(lv_p)
    term = (lv_p - lv_q) + (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8) - 1.0
    return 0.5 * term.sum(dim=-1)

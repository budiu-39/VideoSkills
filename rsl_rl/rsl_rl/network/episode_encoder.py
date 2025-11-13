# episode_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 工具：KL[q||p]（对角高斯）----
def kl_normal(mu_q, logvar_q, mu_p=None, logvar_p=None):
    """
    mu_q, logvar_q: (..., D)
    mu_p, logvar_p: (..., D)  若为 None，默认 p = N(0, I)
    return: KL per-sample (...,)
    """
    if mu_p is None:    mu_p = torch.zeros_like(mu_q)
    if logvar_p is None: logvar_p = torch.zeros_like(logvar_q)
    # KL = 0.5 * ( tr(Sigma_p^-1 Sigma_q) + (mu_p - mu_q)^T Sigma_p^-1 (mu_p - mu_q) - k + log(|Sigma_p|/|Sigma_q|) )
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (
        (var_q / var_p).sum(dim=-1) +
        ((mu_p - mu_q)**2 / var_p).sum(dim=-1) -
        mu_q.shape[-1] +
        (logvar_p - logvar_q).sum(dim=-1)
    )
    return kl

# ---- Episode Encoder：K 帧上下文 -> z_global（对角高斯）（支持 mask）----
class EpisodeEncoder(nn.Module):
    """
    输入:
      x:    [B, C, K]
      mask: [B, K]  0/1，有效帧=1（可为 None）
    输出:
      z ~ N(mu, diag(exp(logvar)))，维度 d_z
    结构:
      Conv1d 堆叠 -> masked mean pool (time) -> 2x Linear
    """
    def __init__(self, in_channels: int, d_model: int = 128, d_z: int = 32, n_layers: int = 3):
        super().__init__()
        convs = []
        c = in_channels
        for _ in range(n_layers):
            convs += [
                nn.Conv1d(c, d_model, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            c = d_model
        self.backbone = nn.Sequential(*convs)  # [B, d_model, K]
        self.head_mu = nn.Linear(d_model, d_z)
        self.head_lv = nn.Linear(d_model, d_z)
        self.d_z = d_z

    @staticmethod
    def _masked_mean_time(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h:    [B, D, K]
        mask: [B, K] or None
        return: [B, D]
        """
        if mask is None:
            return h.mean(dim=-1)
        w = mask.unsqueeze(1)                      # [B,1,K]
        num = (h * w).sum(dim=-1)                  # [B,D]
        den = w.sum(dim=-1).clamp_min(1e-6)        # [B,1]  防止除 0
        return num / den

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x:    [B, C, K]
        mask: [B, K]  (optional)
        return: z, mu, logvar
        """
        # （可选）把 pad 帧先置零，避免卷积看到非零垃圾值
        if mask is not None:
            x = x * mask.unsqueeze(1)              # 广播到 [B,C,K]

        h = self.backbone(x)                       # [B, d_model, K]
        h = self._masked_mean_time(h, mask)        # [B, d_model]

        mu = self.head_mu(h)                       # [B, d_z]
        logvar = torch.clamp(self.head_lv(h), min=-8.0, max=8.0)
        std = (0.5 * logvar).exp()

        # reparameterization
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


# ---- 条件先验：基于“简短统计”给出 p(z|stats)（可选，开始可不用）----
class PriorNet(nn.Module):
    """
    输入: stats 向量 [B, S]（例如：初始本体状态统计 + 片段的速度/转向统计）
    输出: prior 的 (mu_p, logvar_p)
    """
    def __init__(self, in_dim: int, d_z: int = 32, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )
        self.head_mu = nn.Linear(hidden, d_z)
        self.head_lv = nn.Linear(hidden, d_z)

    def forward(self, stats):
        h = self.mlp(stats)
        mu_p = self.head_mu(h)
        logvar_p = torch.clamp(self.head_lv(h), min=-8.0, max=8.0)
        return mu_p, logvar_p

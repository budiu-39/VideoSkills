import torch
import torch.nn as nn
from rsl_rl.backbone import TCN

class TCNEncoder(nn.Module):
    """
    Encoder:
      输入: state_seq [B, T, D_state]
      输出: latent μ, logvar [B, latent_dim]
    """
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tcn = TCN(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, state_seq):
        # state_seq: [B, T, D_state]
        x = state_seq.transpose(1, 2)  # [B, D_state, T]
        h = self.tcn(x)                # [B, H, T]

        # global pooling over time (average)
        h_global = h.mean(dim=-1)      # [B, H]

        mu = self.fc_mu(h_global)      # [B, latent_dim]
        logvar = self.fc_logvar(h_global)
        return mu, logvar


class TCNDecoder(nn.Module):
    """
    Decoder:
      输入: latent z [B, latent_dim]
      输出: state_hat [B, T, D_state], action_hat [B, T, D_action]
    使用一个 TCN backbone + 两个输出头
    """
    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        action_dim: int,
        window_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_pos_encoding = use_pos_encoding

        # 我们先把 z 映射到 hidden_dim，再展开到每个时间步
        self.fc_z = nn.Linear(latent_dim, hidden_dim)

        # 可选: 简单的 1D 位置编码（线性坐标）
        if use_pos_encoding:
            self.pos_embed = nn.Parameter(torch.randn(window_size, hidden_dim))
        else:
            self.pos_embed = None

        # TCN backbone
        # 输入维度就是 hidden_dim
        self.tcn = TCN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # 两个输出头
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z):
        # z: [B, latent_dim]
        B = z.size(0)
        T = self.window_size

        # [B, H]
        h0 = self.fc_z(z)

        # 扩展到每个时间步: [B, H, T]
        h = h0.unsqueeze(-1).expand(B, h0.size(1), T)

        # 加位置编码 (broadcast) -> 依然 [B, H, T]
        if self.pos_embed is not None:
            # pos_embed: [T, H] -> [1, H, T]
            pos = self.pos_embed.transpose(0, 1).unsqueeze(0)  # [1, H, T]
            h = h + pos

        # TCN 处理
        h_dec = self.tcn(h)  # [B, H, T]

        # 转回 [B, T, H]
        h_dec = h_dec.transpose(1, 2)  # [B, T, H]

        state_hat = self.state_head(h_dec)   # [B, T, D_state]
        action_hat = self.action_head(h_dec) # [B, T, D_action]
        return state_hat, action_hat


import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- utils: FeedForward 保留 --------
class FeedForward(nn.Module):
    def __init__(self, d_model, mlp_ratio=4.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
    def forward(self, x):  # [B, L, D]
        return self.net(x)

# -------- Decoder Block：Self-Attn + Cross-Attn(z) + FFN --------
class ZDecoderBlock(nn.Module):
    """
    非自回归 Transformer Decoder block：
    - 对 [CLS + body_tokens] 做 self-attention
    - 再对 z_tokens 做 cross-attention（z 作为条件）
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.d_model = d_model

        # self-attention over [CLS + body_tokens]
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )

        # cross-attention: Q = [CLS + body], K,V = z_tokens
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )

        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_ratio=mlp_ratio)

    def forward(self, x_q: torch.Tensor, z_tokens: torch.Tensor) -> torch.Tensor:
        """
        x_q:      [B, Lq, D]  (CLS + body tokens)
        z_tokens: [B, Lz, D]  (通常 Lz = 1)
        """
        # 1) self-attention on query tokens
        x = x_q + self.self_attn(
            self.norm1(x_q),
            self.norm1(x_q),
            self.norm1(x_q),
            need_weights=False,
        )[0]

        # 2) cross-attention: Q = x, K/V = z_tokens
        x = x + self.cross_attn(
            self.norm2(x),          # Q
            self.norm2(z_tokens),   # K
            self.norm2(z_tokens),   # V
            need_weights=False,
        )[0]

        # 3) FFN
        x = x + self.ff(self.norm3(x))
        return x

# -------- Decoder-Style BodyGraphZ --------
class BodyGraphZ(nn.Module):
    """
    Decoder 版 BodyGraphZ：
    - proprio → body tokens（作为 Q 的一部分）
    - z       → z_token（作为 K/V）
    - [CLS, body_tokens] 先 self-attn，再 cross-attn 到 z
    - 输出 CLS 作为全局表征
    """
    def __init__(
        self,
        obs_dim: int,          # proprio 维度
        z_dim: int,            # latent 维度
        num_bodies: int = 24,
        d_model: int = 256,
        n_heads: int = 4,
        depth: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.num_bodies = num_bodies
        self.d_model = d_model

        # === tokenizer: proprio → body tokens ===
        self.tokenizer = nn.Sequential(
            nn.Linear(obs_dim, num_bodies * d_model),
            nn.SiLU()
        )

        # === decoder blocks ===
        self.blocks = nn.ModuleList([
            ZDecoderBlock(d_model, n_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # === learnable CLS token（作为 Q 的一部分）===
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        # === z token projection（作为 K/V）===
        self.z_proj = nn.Linear(z_dim, d_model)
        nn.init.xavier_uniform_(self.z_proj.weight)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, obs_dim + z_dim]
          - proprio: obs[:, :obs_dim]
          - z:       obs[:, obs_dim: obs_dim + z_dim]
        """
        proprio = obs[:, :self.obs_dim]
        z = obs[:, self.obs_dim:self.obs_dim + self.z_dim]
        B = proprio.size(0)

        # ---- Q: [CLS + body_tokens] ----
        body_tokens = self.tokenizer(proprio).view(
            B, self.num_bodies, self.d_model
        )  # [B, N, D]

        cls_token = self.cls.expand(B, 1, self.d_model)  # [B, 1, D]
        x_q = torch.cat([cls_token, body_tokens], dim=1)  # [B, 1+N, D]

        # ---- K,V: z_tokens ----
        z_tokens = self.z_proj(z).unsqueeze(1)  # [B, 1, D]

        # ---- decoder stack ----
        x = x_q
        for blk in self.blocks:
            x = blk(x_q=x, z_tokens=z_tokens)

        x = self.final_norm(x)
        global_feat = x[:, 0]  # 取 CLS 作为全局表征
        return global_feat

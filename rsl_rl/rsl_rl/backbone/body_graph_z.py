# videoskills/policies/body_graph_policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 如果后续做时序状态，可从 rsl_rl.modules import ActorCriticRecurrent

# -------- utils: Multi-Head Graph Attention (mask 邻接可选) --------
class GraphAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # x: [B, N, D]
        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, mlp_ratio=4.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
    def forward(self, x):  # [B,N,D]
        return self.net(x)

class SpatialBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.gattn = GraphAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
    def forward(self, x):  # [B,N,D]
        x = x + self.gattn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# -------- Backbone：Tokenizer + 多层 SpatialBlock + Pool --------
class BodyGraphZ(nn.Module):
    """
    BodyGraphBackbone 的 z 版本：
    - proprio → body tokens
    - z       → global token
    - [CLS, z_token, body_tokens] 经多层图注意力融合
    输出全局表征 [B, D]
    """
    def __init__(
        self,
        obs_dim: int,          # proprio 维度
        z_dim: int,            # latent 维度
        num_bodies: int = 24,
        d_model: int = 256,
        n_heads: int = 4,
        depth: int = 4,
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

        self.blocks = nn.ModuleList([SpatialBlock(d_model, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)

        # === learnable CLS token ===
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        # === z token projection ===
        self.z_proj = nn.Linear(z_dim, d_model)
        nn.init.xavier_uniform_(self.z_proj.weight)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        proprio: [B, obs_dim]
        z:       [B, z_dim]
        """
        proprio = obs[:, :self.obs_dim]
        z = obs[:, self.obs_dim:self.obs_dim + self.z_dim]
        B = proprio.size(0)
        x_body = self.tokenizer(proprio).view(B, self.num_bodies, self.d_model)  # [B,N,D]

        z_token = self.z_proj(z).unsqueeze(1)      # [B,1,D]
        cls_token = self.cls.expand(B, -1, -1)     # [B,1,D]

        x = torch.cat([cls_token, z_token, x_body], dim=1)  # [B,1+1+N,D]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        global_feat = x[:, 0]  # CLS token 作为全局表征
        return global_feat

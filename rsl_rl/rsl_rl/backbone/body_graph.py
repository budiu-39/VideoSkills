# videoskills/policies/body_graph_policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules import ActorCritic  # 直接替换默认 ActorCritic
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
class BodyGraphBackbone(nn.Module):
    def __init__(self, obs_dim, num_bodies=24, d_model=256, n_heads=4, depth=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_bodies = num_bodies
        self.d_model = d_model
        # 简单 tokenizer：把扁平 obs → [B, num_bodies, d_model]
        self.tokenizer = nn.Sequential(
            nn.Linear(obs_dim, num_bodies * d_model),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([SpatialBlock(d_model, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)

        # 可选：learnable [CLS]，用于全局聚合（也可用 mean-pool）
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, obs):  # obs: [B, obs_dim]
        B = obs.size(0)
        x = self.tokenizer(obs).view(B, self.num_bodies, self.d_model)  # [B,N,D]
        # 追加 CLS
        cls = self.cls.expand(B, -1, -1)  # [B,1,D]
        x = torch.cat([cls, x], dim=1)    # [B,1+N,D]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # 取 CLS 作为全局表征（也可 mean-pool）
        global_feat = x[:, 0]  # [B,D]
        return global_feat

# -------- Actor-Critic 封装：保持接口兼容 OnPolicyRunner --------
class ActorCritic_BodyGraph(ActorCritic):
    def __init__(self, num_obs, num_actions, **kwargs):
        # 忽略父类 MLP 构造，自己搭头
        nn.Module.__init__(self)
        self.num_obs = num_obs
        self.num_actions = num_actions

        # 这些超参可来自 cfg["policy"].extra_xxx
        d_model   = kwargs.get("d_model", 256)
        depth     = kwargs.get("depth", 4)
        n_heads   = kwargs.get("n_heads", 4)
        num_bodies= kwargs.get("num_bodies", 24)

        self.backbone = BodyGraphBackbone(
            obs_dim=num_obs, num_bodies=num_bodies,
            d_model=d_model, n_heads=n_heads, depth=depth
        )

        # Actor / Critic 头
        self.actor_head  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_actions))
        self.critic_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        # 动作标准差（若 fixed_std=True）
        self.log_std = nn.Parameter(torch.zeros(num_actions), requires_grad=True)

    # --- 对齐 rsl_rl ActorCritic 接口 ---
    def act(self, obs):
        feat = self.backbone(obs)                # [B,D]
        mu   = self.actor_head(feat)             # [B,A]
        std  = self.log_std.exp().expand_as(mu)  # [B,A]
        return mu, std

    def evaluate(self, obs, actions):
        feat = self.backbone(obs)
        mu   = self.actor_head(feat)
        std  = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy  = dist.entropy().sum(dim=-1, keepdim=True)
        value    = self.critic_head(feat)
        return value, log_prob, entropy

    def get_value(self, obs):
        feat = self.backbone(obs)
        return self.critic_head(feat)

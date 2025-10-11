import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic
from rsl_rl.utils.body_graph import BodyGraphBackbone

# 关键点：模仿 MLP ActorCritic 的风格（distribution + properties）
class ActorCritic_Attention(ActorCritic):
    is_recurrent = False

    def __init__(self, num_obs, num_critic_obs, num_actions, **kwargs):
        nn.Module.__init__(self)
        self.num_obs = num_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions

        d_model    = kwargs.get("d_model", 256)
        depth      = kwargs.get("depth", 4)
        n_heads    = kwargs.get("n_heads", 4)
        num_bodies = kwargs.get("num_bodies", 24)

        fixed_std      = kwargs.get("fixed_std", False)
        init_noise_std = kwargs.get("init_noise_std", 0.055)

        # --- backbones ---
        self.actor_backbone = BodyGraphBackbone(
            obs_dim=num_obs, num_bodies=num_bodies, d_model=d_model, n_heads=n_heads, depth=depth
        )
        self.critic_backbone = BodyGraphBackbone(
            obs_dim=num_critic_obs, num_bodies=num_bodies, d_model=d_model, n_heads=n_heads, depth=depth
        )

        # --- heads ---
        self.actor_head  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_actions))
        self.critic_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        # --- 模仿 MLP 版：直接维护 std 参数 + distribution 对象 ---
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=not fixed_std)
        self.distribution = None  # 正态分布缓存（与 MLP 版一致）
        Normal.set_default_validate_args = False  # 性能

    # ====== 模仿 MLP 版的接口 ======
    def update_distribution(self, observations):
        feat = self.actor_backbone(observations)     # [B, D]
        mean = self.actor_head(feat)                 # [B, A]
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, **_):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        # 和 MLP 版一致：按动作维度求和
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        with torch.no_grad():
            feat = self.actor_backbone(observations)
            actions_mean = self.actor_head(feat)
        return actions_mean

    def evaluate(self, critic_observations, **_):
        feat = self.critic_backbone(critic_observations)
        return self.critic_head(feat)

    def reset(self, dones=None):
        pass

    # ====== 只读属性：与 MLP 版一致 ======
    @property
    def action_mean(self):
        return self.distribution.mean if self.distribution is not None else None

    @property
    def action_std(self):
        return self.distribution.stddev if self.distribution is not None else None

    @property
    def entropy(self):
        if self.distribution is None:
            return None
        return self.distribution.entropy().sum(dim=-1)


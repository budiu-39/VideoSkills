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
        # action_scale = kwargs.get("action_scale", 0.25)
        use_obs_rms = kwargs.get("use_obs_rms", True)
        fixed_std      = kwargs.get("fixed_std", False)
        init_noise_std = kwargs.get("init_noise_std", 0.055)

        # self.action_scale = torch.tensor(action_scale, dtype=torch.float32)
        # --- backbones ---
        self.actor_backbone = BodyGraphBackbone(
            obs_dim=num_obs, num_bodies=num_bodies, d_model=d_model, n_heads=n_heads, depth=depth
        )
        self.critic_backbone = BodyGraphBackbone(
            obs_dim=num_critic_obs, num_bodies=num_bodies, d_model=d_model, n_heads=n_heads, depth=depth
        )

        if use_obs_rms:
            from rsl_rl.utils.running_mean_std import RunningMeanStd
            self.actor_obs_rms = RunningMeanStd((num_obs,))
            self.critic_obs_rms = RunningMeanStd((num_critic_obs,))
            self._update_rms = True
        else:
            self.actor_obs_rms = self.critic_obs_rms = None
            self._update_rms = False

        # --- heads ---
        self.actor_head  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_actions))
        self.critic_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        nn.init.uniform_(self.actor_head[-1].weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.actor_head[-1].bias)
        nn.init.uniform_(self.critic_head[-1].weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.critic_head[-1].bias)

        # self.action_scale = torch.tensor(action_scale, dtype=torch.float32)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=not fixed_std)

        Normal.set_default_validate_args = False  # 性能

    def _norm_actor_obs(self, obs):
        if self.actor_obs_rms is None:
            return obs
        if self._update_rms:
            _ = self.actor_obs_rms(obs)
        return self.actor_obs_rms(obs)

    def _norm_critic_obs(self, obs):
        if self.critic_obs_rms is None:
            return obs
        if self._update_rms:
            _ = self.critic_obs_rms(obs)
        return self.critic_obs_rms(obs)

    def update_distribution(self, observations):
        obs = self._norm_actor_obs(observations)
        feat = self.actor_backbone(obs)
        pre = self.actor_head(feat)  # 未压缩
        mean = torch.tanh(pre) # * self.action_scale
        std = torch.clamp(self.std, 1e-4, 1.0)  # 避免过大/过小
        self.distribution = Normal(mean, mean * 0. + std)

    def act(self, observations, **_):
        self.update_distribution(observations)
        a = self.distribution.sample()
        # 保险起见再裁剪一次
        return a

    def get_actions_log_prob(self, actions):
        # 和 MLP 版一致：按动作维度求和
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        with torch.no_grad():
            obs = self._norm_actor_obs(observations)
            feat = self.actor_backbone(obs)
            pre = self.actor_head(feat)
            mean = torch.tanh(pre) # * self.action_scale
        return mean

    def evaluate(self, critic_observations, **_):
        obs = self._norm_critic_obs(critic_observations)
        feat = self.critic_backbone(obs)
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


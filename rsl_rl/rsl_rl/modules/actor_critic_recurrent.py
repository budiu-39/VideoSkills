import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic  # 导入基类
from rsl_rl.utils.running_mean_std import RunningMeanStd

# 方案二：循环学生 (Recurrent Student for Windowed Z)
class ActorCritic_Recurrent_Z(ActorCritic):
    is_recurrent = True  # 关键：告诉 PPO 这是一个循环模型

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 num_state_obs,  # s_t 的维度
                 num_latent_z,  # z 的维度
                 rnn_hidden_size=256,
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):

        # 不要调用 super().__init__()，因为它会构建 MLP
        nn.Module.__init__(self)

        # 验证输入维度
        assert num_actor_obs == num_state_obs + num_latent_z
        assert num_critic_obs == num_state_obs + num_latent_z

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.rnn_hidden_size = rnn_hidden_size
        act = self.get_activation(activation)  #

        # --- Actor ---
        # 1. 编码器 (MLP pre-processor)
        self.actor_encoder = nn.Sequential(
            nn.Linear(num_actor_obs, rnn_hidden_size), act,
            nn.Linear(rnn_hidden_size, rnn_hidden_size), act
        )
        # 2. 循环核心 (GRU)
        self.actor_gru = nn.GRU(rnn_hidden_size, rnn_hidden_size)
        # 3. 动作头 (MLP head)
        self.actor_head = nn.Linear(rnn_hidden_size, num_actions)

        # --- Critic ---
        self.critic_encoder = nn.Sequential(
            nn.Linear(num_critic_obs, rnn_hidden_size), act,
            nn.Linear(rnn_hidden_size, rnn_hidden_size), act
        )
        self.critic_gru = nn.GRU(rnn_hidden_size, rnn_hidden_size)
        self.critic_head = nn.Linear(rnn_hidden_size, 1)

        # --- 动作分布 (与 actor_critic.py 一致) ---
        fixed_std = kwargs.get("fixed_std", False)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=not fixed_std)
        self.distribution = None
        Normal.set_default_validate_args = False

        # --- 隐藏状态 (用于 PPO  rollout) ---
        # ppo.py 会调用 get_hidden_states()
        self.actor_hidden_states = None
        self.critic_hidden_states = None

        # RMS (可选, 但推荐)
        self.actor_obs_rms = RunningMeanStd((num_actor_obs,))
        self.critic_obs_rms = RunningMeanStd((num_critic_obs,))
        self._update_rms = True

    def set_update_rms(self, flag: bool):
        self._update_rms = flag
        if self.actor_obs_rms: self.actor_obs_rms.train(flag)
        if self.critic_obs_rms: self.critic_obs_rms.train(flag)

    def reset(self, dones=None):
        # PPO 在 dones 时调用 reset
        if dones is None: return
        if self.actor_hidden_states is not None:
            self.actor_hidden_states[dones] = 0.
        if self.critic_hidden_states is not None:
            self.critic_hidden_states[dones] = 0.

    def get_hidden_states(self):
        # PPO 在 act 之前调用
        return self.actor_hidden_states, self.critic_hidden_states

    def _norm_actor_obs(self, obs):
        if self._update_rms: _ = self.actor_obs_rms(obs)
        return self.actor_obs_rms(obs)

    def _norm_critic_obs(self, obs):
        if self._update_rms: _ = self.critic_obs_rms(obs)
        return self.critic_obs_rms(obs)

    def _run_gru(self, x, gru_layer, hidden_states):
        # GRU 需要 (seq_len, batch, input_size)
        x = x.unsqueeze(0)
        if hidden_states is None:
            # 第一次初始化
            hidden_states = torch.zeros((1, x.shape[1], self.rnn_hidden_size), device=x.device, dtype=x.dtype)

        # hidden_states 需要 (num_layers, batch, hidden_size)
        gru_out, new_hidden_states = gru_layer(x, hidden_states)

        # 输出 (batch, hidden_size) 和 (num_layers, batch, hidden_size)
        return gru_out.squeeze(0), new_hidden_states

    def update_distribution(self, observations, **kwargs):
        # PPO 在 update 循环中会传入 'hidden_states' kwarg
        hidden_states = kwargs.get('hidden_states')
        if hidden_states is None:
            # 如果在 rollout (act) 期间调用，使用缓冲区
            hidden_states = self.actor_hidden_states

        obs_norm = self._norm_actor_obs(observations)
        x_encoded = self.actor_encoder(obs_norm)

        gru_out, new_hidden = self._run_gru(x_encoded, self.actor_gru, hidden_states)

        # 更新 rollout 缓冲区（分离梯度）
        self.actor_hidden_states = new_hidden.detach()

        mean = self.actor_head(gru_out)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def evaluate(self, critic_observations, **kwargs):
        # PPO 在 update 循环中会传入 'hidden_states' kwarg
        hidden_states = kwargs.get('hidden_states')
        if hidden_states is None:
            # 如果在 rollout (act) 期间调用，使用缓冲区
            hidden_states = self.critic_hidden_states

        obs_norm = self._norm_critic_obs(critic_observations)
        x_encoded = self.critic_encoder(obs_norm)

        gru_out, new_hidden = self._run_gru(x_encoded, self.critic_gru, hidden_states)

        # 更新 rollout 缓冲区（分离梯度）
        self.critic_hidden_states = new_hidden.detach()

        return self.critic_head(gru_out)

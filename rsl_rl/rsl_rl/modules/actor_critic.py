# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils.running_mean_std import RunningMeanStd
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self,
                 num_actor_obs: int,
                 num_critic_obs: int,
                 num_actions: int,
                 d_model = None,
                 actor_backbone='MLP',
                 critic_backbone='MLP',
                 init_noise_std: float = 1.0,
                 fixed_std: bool = False,
                 use_obs_rms: bool = True,
                 use_embedding: bool = False,
                 num_embeddings: int = 138,  # 动作总数
                 embedding_dim: int = 32,  # 编码后的向量维度
                 actor_hidden_dims = [256, 256],
                 critic_hidden_dims = [256, 256],
                 **kwargs
                 ):
        super().__init__()
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions

        if actor_backbone == 'MLP':
            self.actor_network = self.build_mlp(
                input_dim=num_actor_obs,
                hidden_dims=actor_hidden_dims,
                activation=nn.SiLU
            )

        if critic_backbone == 'MLP':
            self.critic_network = self.build_mlp(
                input_dim=num_critic_obs,
                hidden_dims=critic_hidden_dims,
                activation=nn.SiLU
            )

        # 如果为 None 就取最后一层隐藏层的维度

        d_model = actor_hidden_dims[-1]

        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
            # 如果使用了 embedding，RMS 归一化的维度应该是总维度减去 ID 的 1 维
            # 假设 ID 位于 obs 的最后一列
            rms_actor_dim = num_actor_obs - 32
            # Critic 是否包含 ID 取决于你的具体实现，这里假设 Critic 不需要 Embedding 或者单独处理
            # 简单起见，这里假设 Critic 也是同样的结构
            rms_critic_dim = num_critic_obs - 32
        else:
            rms_actor_dim = num_actor_obs
            rms_critic_dim = num_critic_obs

        self._update_rms = use_obs_rms
        if use_obs_rms:
            self.actor_obs_rms = RunningMeanStd((rms_actor_dim,))
            self.critic_obs_rms = RunningMeanStd((rms_critic_dim,))
        else:
            self.actor_obs_rms = self.critic_obs_rms = None

        # Policy heads
        self.actor_head = nn.Sequential(nn.Linear(d_model, num_actions))
        self.logvar_head = nn.Sequential(nn.Linear(d_model, num_actions))
        self.critic_head = nn.Sequential(nn.Linear(d_model, 1))

        nn.init.uniform_(self.actor_head[-1].weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.actor_head[-1].bias)
        nn.init.uniform_(self.critic_head[-1].weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.critic_head[-1].bias)

        # 标准 PPO 的 state-independent std
        self.policy_std = nn.Parameter(
            init_noise_std * torch.ones(num_actions),
            requires_grad=not fixed_std
        )

        # --- gSDE 配置: state-dependent std ---
        # self.use_gsde = use_gsde
        # self.gsde_eps = 1e-6
        #
        # if self.use_gsde:
        #     # 让 E[std] 大概在 init_noise_std 附近
        #     # Heuristics: logσ = log(init_noise_std) - 0.5*log(d)
        #     scale_correction = 0.5 * np.log(d_model)
        #     initial_log_val = float(np.log(init_noise_std) - scale_correction)
        #
        #     # [D, A] 的 log σ 参数矩阵
        #     self.log_std_sde = nn.Parameter(
        #         torch.ones(d_model, num_actions) * initial_log_val
        #     )
        # else:
        #     self.log_std_sde = None

        # 这些变量由 act()/evaluate() 写入，被 PPO 使用
        self.distribution = None
        self._latent_mu = None
        self._latent_lv = None
        Normal.set_default_validate_args = False
        self.is_recurrent = False

    # --------------------------------------------------------------------- #
    #                              Forward (policy)
    # --------------------------------------------------------------------- #

    def act(self, observations):
        # [修改] 使用新的预处理函数
        obs = self._preprocess_obs(observations, self.actor_obs_rms)

        feat = self.actor_network(obs)  # [B, D]
        mu = self.actor_head(feat)  # [B, A]

        std_policy = torch.clamp(self.policy_std, 1e-4, 1.0)  # [A]
        if std_policy.dim() == 1 and mu.dim() == 2:
            std_policy = std_policy.unsqueeze(0).expand_as(mu)  # [B, A]

        self.distribution = Normal(mu, std_policy)
        # 用 reparameterization 采样，保证梯度正确
        actions = self.distribution.rsample()
        return actions

    # --------------------------------------------------------------------- #
    #                          公共工具 / 规范化
    # --------------------------------------------------------------------- #
    def set_update_rms(self, flag: bool):
        self._update_rms = flag
        if self.actor_obs_rms is not None:
            self.actor_obs_rms.train(flag)
            self.critic_obs_rms.train(flag)

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

    # --------------------------------------------------------------------- #
    #                     latent 分布（你的 KL_lat / distill 用）
    # --------------------------------------------------------------------- #
    def update_latent_distribution(self, observations):
        # [修改] 使用新的预处理函数
        obs = self._preprocess_obs(observations, self.actor_obs_rms)

        feat = self.actor_network(obs)

        mu_q = self.actor_head(feat)
        lv_q = self.logvar_head(feat).clamp(min=-10.0, max=2.0)
        std_q = torch.exp(0.5 * lv_q)

        self._latent_mu = mu_q
        self._latent_lv = lv_q
        return mu_q, lv_q, std_q

    @torch.no_grad()
    def act_inference(self, observations):
        # [修改] 使用新的预处理函数
        obs = self._preprocess_obs(observations, self.actor_obs_rms)

        feat = self.actor_network(obs)
        act = self.actor_head(feat)
        return act

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations):
        # [修改] Critic 也使用预处理函数
        # 注意：如果 Critic 输入不包含 ID，这里需要单独逻辑。
        # 假设 Critic 输入结构与 Actor 类似（包含 ID），则复用逻辑
        obs = self._preprocess_obs(critic_observations, self.critic_obs_rms)

        feat = self.critic_network(obs)
        return self.critic_head(feat)

    def reset(self, dones=None):
        # 如果以后要做 recurrent，可以在这里 reset hidden states
        pass

    @property
    def action_mean(self):
        return self.distribution.mean if self.distribution is not None else None

    @property
    def action_std(self):
        return self.distribution.stddev if self.distribution is not None else None

    # === latent 相关接口 ===
    @property
    def latent_mu(self):
        return self._latent_mu

    @property
    def latent_logvar(self):
        return self._latent_lv

    @property
    def entropy(self):
        if self.distribution is None:
            return None
        return self.distribution.entropy().sum(dim=-1)

    # # --------------------------------------------------------------------- #
    # #                           gSDE: std 计算
    # # --------------------------------------------------------------------- #
    # def _get_gsde_std_matrix(self) -> torch.Tensor:
    #     """
    #     返回 [D, A] 的 σ 参数矩阵。
    #     """
    #     return torch.exp(self.log_std_sde)
    #
    # def _compute_gsde_std(self, feat: torch.Tensor) -> torch.Tensor:
    #     """
    #     给定 feature φ(s)=[B,D]，计算每个维度的 state-dependent std:
    #         var(s) = (φ(s)^2) @ (σ^2)
    #         std(s) = sqrt(var(s) + eps)
    #     返回: [B, A]
    #     """
    #     sigma_mat = self._get_gsde_std_matrix()      # [D, A]
    #     phi_sq = feat.pow(2)                         # [B, D]
    #     sigma_sq = sigma_mat.pow(2)                  # [D, A]
    #
    #     var = phi_sq @ sigma_sq                      # [B, A]
    #     raw_std = torch.sqrt(var + self.gsde_eps)
    #
    #     # 保守一点做个 clamp，避免 std 过小或过大
    #     MIN_LIMIT = 1e-3
    #     MAX_LIMIT = 5.0
    #     std = torch.clamp(raw_std, min=MIN_LIMIT, max=MAX_LIMIT)
    #     return std

    # --------------------------------------------------------------------- #
    #                          核心处理逻辑 (新增)
    # --------------------------------------------------------------------- #

    def _preprocess_obs(self, obs, rms_module):
        """
        处理输入观测：
        1. 如果启用 Embedding，将最后一维(ID)分离出来。
        2. 对剩余的连续观测进行 RMS 归一化。
        3. 对 ID 进行 Embedding 查表。
        4. 拼接并返回。
        """
        if not self.use_embedding:
            # 原有逻辑：直接归一化所有
            if rms_module is None:
                return obs
            if self._update_rms:
                rms_module(obs)  # update stats
            return rms_module(obs)

        else:
            # 假设 ID 是最后一维
            continuous_obs = obs[:, :-1]
            motion_id = obs[:, -1].long()

            # 1. 归一化连续部分
            if rms_module is not None:
                continuous_obs = rms_module(continuous_obs)

            # 2. Embedding 离散部分
            motion_embed = self.embedding_layer(motion_id)

            # 3. 拼接 [Norm_State, Embedding]
            return torch.cat([continuous_obs, motion_embed], dim=-1)

    def build_mlp(self, input_dim, hidden_dims, activation=nn.SiLU):
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            prev_dim = h_dim
        return nn.Sequential(*layers)


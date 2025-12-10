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
                 actor_network: nn.Module,
                 critic_network: nn.Module,
                 d_model: int,
                 init_noise_std: float = 1.0,
                 fixed_std: bool = False,
                 use_obs_rms: bool = True,
                 use_gsde: bool = False,
                 **kwargs):
        super().__init__()
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions

        self.actor_network = actor_network
        self.critic_network = critic_network
        self.d_model = d_model

        self._update_rms = use_obs_rms
        if use_obs_rms:
            self.actor_obs_rms = RunningMeanStd((num_actor_obs,))
            self.critic_obs_rms = RunningMeanStd((num_critic_obs,))
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
        self.use_gsde = use_gsde
        self.gsde_eps = 1e-6

        if self.use_gsde:
            # 让 E[std] 大概在 init_noise_std 附近
            # Heuristics: logσ = log(init_noise_std) - 0.5*log(d)
            scale_correction = 0.5 * np.log(d_model)
            initial_log_val = float(np.log(init_noise_std) - scale_correction)

            # [D, A] 的 log σ 参数矩阵
            self.log_std_sde = nn.Parameter(
                torch.ones(d_model, num_actions) * initial_log_val
            )
        else:
            self.log_std_sde = None

        # 这些变量由 act()/evaluate() 写入，被 PPO 使用
        self.distribution = None
        self._latent_mu = None
        self._latent_lv = None
        Normal.set_default_validate_args = False
        self.is_recurrent = False

    # --------------------------------------------------------------------- #
    #                              Forward (policy)
    # --------------------------------------------------------------------- #


    def act(self, observations, **kwargs):
        """
        统一的动作生成接口：
        - 若 use_gsde=False:  π(a|s) = N(μ(s), σ) (state-independent)
        - 若 use_gsde=True :  π(a|s) = N(μ(s), σ(s))，σ(s) 由 gSDE 公式计算
        无论哪种情况：
        - action 都是从 self.distribution.rsample() 采样出来的
        - PPO 的 log_prob / KL 都是数学一致的
        """
        obs = self._norm_actor_obs(observations)
        feat = self.actor_network(obs)          # [B, D]
        mu = self.actor_head(feat)              # [B, A]

        if self.use_gsde and self.log_std_sde is not None:
            # state-dependent std
            std_policy = self._compute_gsde_std(feat)   # [B, A]
        else:
            # 原始 PPO：state-independent std
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
        """latent q(z|obs) = N(mu_q, std_q)，只用 actor 的 feature。"""
        obs = self._norm_actor_obs(observations)
        feat = self.actor_network(obs)

        mu_q = self.actor_head(feat)
        lv_q = self.logvar_head(feat).clamp(min=-10.0, max=2.0)
        std_q = torch.exp(0.5 * lv_q)

        self._latent_mu = mu_q
        self._latent_lv = lv_q
        return mu_q, lv_q, std_q

    @torch.no_grad()
    def act_inference(self, observations):
        """评估 / 测试时用的纯 deterministic 动作。"""
        obs = self._norm_actor_obs(observations)
        feat = self.actor_network(obs)
        act = self.actor_head(feat)
        return act

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        obs = self._norm_critic_obs(critic_observations)
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

    # --------------------------------------------------------------------- #
    #                           gSDE: std 计算
    # --------------------------------------------------------------------- #
    def _get_gsde_std_matrix(self) -> torch.Tensor:
        """
        返回 [D, A] 的 σ 参数矩阵。
        """
        return torch.exp(self.log_std_sde)

    def _compute_gsde_std(self, feat: torch.Tensor) -> torch.Tensor:
        """
        给定 feature φ(s)=[B,D]，计算每个维度的 state-dependent std:
            var(s) = (φ(s)^2) @ (σ^2)
            std(s) = sqrt(var(s) + eps)
        返回: [B, A]
        """
        sigma_mat = self._get_gsde_std_matrix()      # [D, A]
        phi_sq = feat.pow(2)                         # [B, D]
        sigma_sq = sigma_mat.pow(2)                  # [D, A]

        var = phi_sq @ sigma_sq                      # [B, A]
        raw_std = torch.sqrt(var + self.gsde_eps)

        # 保守一点做个 clamp，避免 std 过小或过大
        MIN_LIMIT = 1e-3
        MAX_LIMIT = 5.0
        std = torch.clamp(raw_std, min=MIN_LIMIT, max=MAX_LIMIT)
        return std

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
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils.running_mean_std import RunningMeanStd
import copy



class ActorCritic(nn.Module):

    def __init__(self,
                 num_actor_obs: int,
                 num_critic_obs: int,
                 num_actions: int,
                 actor_network: nn.Module,
                 critic_network: nn.Module,
                 d_model: int,
                 init_noise_std: float = 0.055,
                 fixed_std: bool = False,
                 use_obs_rms: bool = True,
                 **kwargs):
        super().__init__()
        self.num_actor_obs  = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions    = num_actions

        self.actor_network  = actor_network
        self.critic_network = critic_network
        self.d_model = d_model

        self._update_rms = use_obs_rms
        if use_obs_rms:
            self.actor_obs_rms  = RunningMeanStd((num_actor_obs,))
            self.critic_obs_rms = RunningMeanStd((num_critic_obs,))
        else:
            self.actor_obs_rms = self.critic_obs_rms = None

        # Policy
        self.actor_head  = nn.Linear(d_model, num_actions)
        self.logvar_head = nn.Linear(d_model, num_actions)
        self.critic_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        nn.init.uniform_(self.actor_head.weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.uniform_(self.critic_head[-1].weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.critic_head[-1].bias)
        nn.init.uniform_(self.logvar_head.weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.logvar_head.bias)

        self.policy_std = nn.Parameter(
            init_noise_std * torch.ones(num_actions),
            requires_grad=not fixed_std
        )

        self.distribution = None
        self._latent_mu = None            # μ_q
        self._latent_lv = None            # logvar_q
        Normal.set_default_validate_args = False
        self.is_recurrent = False

    # ---------- 公共工具 ----------
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

    def update_policy_distribution(self, observations):
        """PPO 用的动作分布：Normal(mu_policy, policy_std)"""
        obs  = self._norm_actor_obs(observations)
        feat = self.actor_network(obs)
        mu_policy  = self.actor_head(feat)
        # mu_policy = torch.tanh(pre)                       # PPO 动作均值
        std_policy = torch.clamp(self.policy_std, 1e-4, 1.0)

        self.distribution = Normal(mu_policy, std_policy)
        return mu_policy, std_policy

    def update_latent_distribution(self, observations):
        """latent q(z|obs) 的分布：Normal(mu_q, std_q) —— 用于 KL_lat / 采样 z"""
        obs  = self._norm_actor_obs(observations)
        feat = self.actor_network(obs)
        mu_q  = self.actor_head(feat)
        # mu_q = torch.tanh(pre)

        lv_q = self.logvar_head(feat).clamp(min=-10.0, max=2.0)
        std_q = torch.exp(0.5 * lv_q)

        self._latent_mu = mu_q
        self._latent_lv = lv_q
        return mu_q, lv_q, std_q

    def act(self, observations, **kwargs):
        self.update_policy_distribution(observations)
        return self.distribution.sample()

    @torch.no_grad()
    def act_inference(self, observations):
        obs  = self._norm_actor_obs(observations)
        feat = self.actor_network(obs)
        act  = self.actor_head(feat)
        return act
        # return torch.tanh(pre)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        obs  = self._norm_critic_obs(critic_observations)
        feat = self.critic_network(obs)
        return self.critic_head(feat)

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        # 这是 PPO 动作 mean
        return self.distribution.mean if self.distribution is not None else None

    @property
    def action_std(self):
        # 这是 PPO 动作 std
        return self.distribution.stddev if self.distribution is not None else None

    # === latent 相关接口：给 distill / KL_lat 用 ===
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


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
                 use_obs_rms: bool = True):
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
        self.actor_head  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_actions))
        self.critic_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        nn.init.uniform_(self.actor_head[-1].weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.actor_head[-1].bias)
        nn.init.uniform_(self.critic_head[-1].weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.critic_head[-1].bias)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions),
                                requires_grad=not fixed_std)
        self.distribution = None
        Normal.set_default_validate_args = False

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

    def update_distribution(self, observations):
        obs  = self._norm_actor_obs(observations)
        feat = self.actor_network(obs)                  # [B,d_model]
        pre  = self.actor_head(feat)                     # [B,A]
        mean = torch.tanh(pre)                           # 与注意力版一致
        std  = torch.clamp(self.std, 1e-4, 1.0)
        self.distribution = Normal(mean, mean*0. + std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    @torch.no_grad()
    def act_inference(self, observations):
        obs  = self._norm_actor_obs(observations)
        feat = self.actor_network(obs)
        pre  = self.actor_head(feat)
        return torch.tanh(pre)

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
        return self.distribution.mean if self.distribution is not None else None

    @property
    def action_std(self):
        return self.distribution.stddev if self.distribution is not None else None

    @property
    def entropy(self):
        if self.distribution is None:
            return None
        return self.distribution.entropy().sum(dim=-1)
    def get_activation(self, act_name):
        if act_name == "elu":
            return nn.ELU()
        elif act_name == "selu":
            return nn.SELU()
        elif act_name == "relu":
            return nn.ReLU()
        elif act_name == "crelu":
            return nn.ReLU()
        elif act_name == "lrelu":
            return nn.LeakyReLU()
        elif act_name == "tanh":
            return nn.Tanh()
        elif act_name == "sigmoid":
            return nn.Sigmoid()
        elif act_name == "silu":
            return nn.SiLU()
        else:
            print("invalid activation function!")
            return None

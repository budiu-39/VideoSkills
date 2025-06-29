"""amp_discriminator.py  (re‑write)

A *stand‑alone*, light‑dependency AMP discriminator that expects exactly the
same `amp_obs` tensor your current env already outputs (shape `[B, feat_dim]`).

Key points
==========
1. **No window logic inside** – the runner is free to feed single‑frame or
   concatenated multi‑frame observations.
2. **Pluggable architecture** – hidden layer sizes, activation, and LayerNorm
   toggled via constructor kwargs.
3. **Minimal public API**
   • `forward(x) -> logits`  (real vs fake)
   • `compute_reward(x) -> r` ( −log(1−D) )
   • `train_step(fake, real, n_updates)`
4. **Optional Normalisation** – RunningMeanStd on inputs (env already gives
   amp‑obs; enable if needed).
"""
from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class RunningMeanStd(nn.Module):
    """Very small RMS implementation (per‑feature)."""

    def __init__(self, dim: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("count", torch.tensor(eps))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))

    @torch.no_grad()
    def update(self, x: torch.Tensor):  # x: [B, D]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean[:] = new_mean
        self.var[:] = new_var
        self.count += batch_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


def make_mlp(dims: Sequence[int], ln: bool = True) -> nn.Sequential:
    layers = []
    for in_d, out_d in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_d, out_d))
        if ln:
            layers.append(nn.LayerNorm(out_d))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers.pop()  # drop last activation
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------
class AMPDiscriminator(nn.Module):
    """Binary classifier D(s) with helper methods for AMP reward & training."""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int] = (512, 256),
        normalize_input: bool = False,
        lr: float = 3e-4,
        weight_decay: float = 1e-6,
        grad_penalty_coef: float = 1.0,
        logit_l2_coef: float = 1e-5,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model = make_mlp([state_dim, *hidden_dims, 1]).to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
        self.bce = nn.BCEWithLogitsLoss()
        self.grad_penalty_coef = grad_penalty_coef
        self.logit_l2_coef = logit_l2_coef
        self.norm = RunningMeanStd(state_dim).to(self.device) if normalize_input else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,D] -> [B]
        if self.norm is not None:
            self.norm.update(x.detach())
            x = self.norm(x)
        return self.model(x).squeeze(-1)

    @torch.inference_mode()
    def compute_reward(self, amp_obs): # ✅ 再保险一层
        logits = self.model(amp_obs)
        probs = torch.sigmoid(logits)
        reward = -torch.log(1.0 - probs + 1e-6)
        return reward

    def _logit_weight_l2(self) -> torch.Tensor:
        return sum((p ** 2).sum() for n, p in self.named_parameters() if "weight" in n and p.ndim > 1)

    def _grad_penalty(self, real: torch.Tensor) -> torch.Tensor:
        real = real.detach().requires_grad_(True)
        logit_real = self.forward(real)
        grad = torch.autograd.grad(logit_real.sum(), real, create_graph=True)[0]
        return (grad.pow(2).sum(dim=-1)).mean()

    def train_step(self, fake: torch.Tensor, real: torch.Tensor, n_updates: int = 1) -> float:
        self.train()
        fake, real = fake.to(self.device), real.to(self.device)
        total_loss = 0.0

        for _ in range(n_updates):
            logit_fake = self.forward(fake.detach())
            logit_real = self.forward(real.detach())
            loss_fake = self.bce(logit_fake, torch.zeros_like(logit_fake))
            loss_real = self.bce(logit_real, torch.ones_like(logit_real))
            loss = loss_fake + loss_real

            # Add gradient penalty
            if self.grad_penalty_coef > 0:
                gp = self._grad_penalty(real)
                loss += self.grad_penalty_coef * gp

            # Add logit weight regularization
            if self.logit_l2_coef > 0:
                loss += self.logit_l2_coef * self._logit_weight_l2()

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)
            self.opt.step()
            total_loss += loss.item()

        return total_loss / n_updates

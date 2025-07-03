from __future__ import annotations
from typing import Sequence
import torch, torch.nn as nn

class AMPDiscriminator(nn.Module):
    """
    D(s) —— 判别器
    - 结构：MLP( state_dim → 1024 → 512 ) + logits(512→1)
    - API: forward / compute_reward / train_step
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        state_dim: int,                     # e.g. 1960
        hidden_dims: Sequence[int] = (1024, 512),
        lr: float = 3e-4,
        weight_decay: float = 1e-6,
        grad_penalty_coef: float = 1.0,
        logit_l2_coef: float = 1e-5,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.grad_penalty_coef = grad_penalty_coef
        self.logit_l2_coef   = logit_l2_coef

        # --------------------- 网络 -------------------------------
        layers: list[nn.Module] = []
        in_dim = state_dim
        for out_dim in hidden_dims:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)]
            in_dim = out_dim
        self._disc_mlp    = nn.Sequential(*layers)
        self._disc_logits = nn.Linear(in_dim, 1)

        # ------------------- 优化器 & 损失 -------------------------
        self.to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr,
                                    betas=(0.5, 0.999),
                                    weight_decay=weight_decay)
        self.bce = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------ #
    # forward / reward
    def forward(self, x: torch.Tensor) -> torch.Tensor:        # [B,D]→[B]
        feat   = self._disc_mlp(x)
        logits = self._disc_logits(feat).squeeze(-1)
        return logits                                           # 无 sigmoid

    @torch.inference_mode()
    def compute_reward(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """
        r(s) = -log(1 - D(s))   —— AMP paper 的伪造者奖励
        """
        logits = self.forward(amp_obs)
        probs  = torch.sigmoid(logits)
        return -torch.log1p(-probs + 1e-6)                      # log1p(x) = log(1+x)

    # ------------------------------------------------------------------ #
    # helpers for regularisation
    def _logit_weight_l2(self) -> torch.Tensor:
        return sum((p ** 2).sum()
                   for n, p in self.named_parameters()
                   if "weight" in n and p.ndim > 1)

    def _grad_penalty(self, real: torch.Tensor) -> torch.Tensor:
        real = real.detach().requires_grad_(True)
        logits_real = self.forward(real)
        grad = torch.autograd.grad(logits_real.sum(), real,
                                   create_graph=True)[0]
        return grad.pow(2).sum(dim=-1).mean()

    # ------------------------------------------------------------------ #
    # 训练一步（与旧版完全一致）
    def train_step(
        self, fake: torch.Tensor, real: torch.Tensor, n_updates: int = 1
    ) -> float:
        self.train()
        fake, real = fake.to(self.device), real.to(self.device)
        total_loss = 0.0

        for _ in range(n_updates):
            logits_f = self.forward(fake.detach())
            logits_r = self.forward(real.detach())

            loss = ( self.bce(logits_f, torch.zeros_like(logits_f))
                   + self.bce(logits_r, torch.ones_like(logits_r)) )

            if self.grad_penalty_coef:
                loss += self.grad_penalty_coef * self._grad_penalty(real)
            if self.logit_l2_coef:
                loss += self.logit_l2_coef   * self._logit_weight_l2()

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)
            self.opt.step()

            total_loss += loss.item()

        return total_loss / n_updates

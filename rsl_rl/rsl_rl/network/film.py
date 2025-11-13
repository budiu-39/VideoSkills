import torch
import torch.nn as nn
from typing import Tuple, Optional

# ---------- 1) FiLM 残差块：只用 z 生成 γ/β ----------
class FiLMBlock(nn.Module):
    def __init__(self, hid: int, z_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hid, hid),
            nn.SiLU(),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(hid, hid),
        )
        self.ln  = nn.LayerNorm(hid)
        self.gam = nn.Linear(z_dim, hid)   # γ(z)
        self.bet = nn.Linear(z_dim, hid)   # β(z)
        # 更稳的起步：γ/β 从 0 开始，相当于先学“恒等”
        nn.init.zeros_(self.gam.weight); nn.init.zeros_(self.gam.bias)
        nn.init.zeros_(self.bet.weight); nn.init.zeros_(self.bet.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x  = self.ff(h)
        x  = self.ln(x)
        gam = self.gam(z)
        bet = self.bet(z)
        x   = gam * x + bet                # FiLM: γ·x + β
        return h + x                       # 残差

# ---------- 2) 简单 MLP 后端（也可换 Graph/Transformer） ----------
class MLPBackbone(nn.Module):
    def __init__(self, in_dim: int, hid: int = 512, depth: int = 3):
        super().__init__()
        layers = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hid), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid, hid), nn.SiLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)    # [B, hid]

# ---------- 3) 统一 obs 输入的 FiLM 网络 ----------
class FiLMNetwork(nn.Module):
    """
    期望 obs 的拼接顺序为: [proprio | act_tail | phase(sin,cos or Fourier) | z]
    如果你的顺序不同，改 _slice() 里的切片即可。

    Args:
        dims: (P, A, F, Z) = (proprio_dim, act_tail_dim, phase_dim, z_dim)
        hid:  隐藏特征维度（需与上层 ActorCritic 的 d_model 对齐）
        depth_film:  FiLM 残差块层数
        phase_gate:  是否使用相位门控（轻量增强，默认 False）
    """
    def __init__(
        self,
        dims: Tuple[int, int, int, int],
        hid: int = 512,
        depth_film: int = 3,
        phase_gate: bool = False,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.proprio_dim, self.act_tail_dim, self.phase_dim, self.z_dim = dims
        self.hid = hid
        in_base = self.proprio_dim + self.act_tail_dim + self.phase_dim

        self.backbone = backbone if backbone is not None else MLPBackbone(in_dim=in_base, hid=hid, depth=3)
        self.pre_ln = nn.LayerNorm(hid)
        self.film_blocks = nn.ModuleList([FiLMBlock(hid, self.z_dim) for _ in range(depth_film)])

        # 可选：phase 门控（不进入 FiLM，只做轻量缩放）
        self.phase_gate = phase_gate
        if phase_gate:
            self.gate = nn.Sequential(
                nn.Linear(self.phase_dim, hid),
                nn.SiLU(),
                nn.Linear(hid, hid),
                nn.Sigmoid(),  # 输出 (0,1)
            )

    # ---- 根据 (P, A, F, Z) 维度切片 ----
    def _slice(self, x: torch.Tensor):
        B, D = x.shape
        P, A, F, Z = self.proprio_dim, self.act_tail_dim, self.phase_dim, self.z_dim
        assert D == P + A + F + Z, f"obs 维度不匹配: got {D}, expect {P+A+F+Z}"

        i0 = 0
        proprio = x[:, i0:i0+P]; i0 += P
        act_tail = x[:, i0:i0+A]; i0 += A
        phase    = x[:, i0:i0+F]; i0 += F
        z        = x[:, i0:i0+Z]
        return proprio, act_tail, phase, z

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprio, act_tail, phase, z = self._slice(obs)             # 统一 obs 内部切片
        x_base = torch.cat([proprio, act_tail, phase], dim=-1)     # phase 作为 base 输入参与编码

        h = self.backbone(x_base)                                  # [B, H]
        h = self.pre_ln(h)

        # 可选：相位门控（轻量增强）
        if self.phase_gate:
            gate = self.gate(phase)                                # [B, H] in (0,1)
            h = h * (0.5 + gate)                                   # 稳一点，避免全关

        # FiLM：仅用 z 产生 γ/β 调制“整段风格”
        for blk in self.film_blocks:
            h = blk(h, z)

        return h   # [B, hid] —— 交给上层 ActorCritic 的 actor_head / critic_head

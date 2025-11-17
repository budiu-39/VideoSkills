import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBackbone(nn.Module):
    """
    通用 MLP backbone:
    in_dim -> hidden[0] -> ... -> hidden[-1]
    可选 LayerNorm 和激活函数类型。
    """
    def __init__(
        self,
        in_dim: int,
        hidden=(512, 512),
        activation=nn.SiLU,
        use_layernorm: bool = False,
    ):
        super().__init__()
        layers = []
        last = in_dim

        for h in hidden:
            layers.append(nn.Linear(last, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation())
            last = h

        self.out_dim = last
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
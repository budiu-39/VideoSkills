import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2   # SAME padding

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # x: [B, C_in, T]
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return out + self.downsample(x)

class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size=3, dropout=0.0):
        super().__init__()
        layers = []
        in_ch = input_dim

        for i in range(num_layers):
            dilation = 2 ** i
            out_ch = hidden_dim
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # 输入: [B, C, T]，输出: [B, hidden_dim, T]
        return self.network(x)


class TCNEncoder(nn.Module):
    """
    Encoder:
      输入: state_seq [B, T, D_state]
      输出: latent μ, logvar [B, latent_dim]
    """
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tcn = TCN(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=False,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, state_seq):
        # state_seq: [B, T, D_state]
        x = state_seq.transpose(1, 2)  # [B, D_state, T]
        h = self.tcn(x)                # [B, H, T]

        # global pooling over time (average)
        h_global = h.mean(dim=-1)      # [B, H]

        mu = self.fc_mu(h_global)      # [B, latent_dim]
        logvar = self.fc_logvar(h_global)
        return mu, logvar


import torch
import torch.nn as nn


# --- 基础组件 ---

class NonLinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)  # Swish / SiLU


class ResConv1DBlock(nn.Module):
    """
    非因果 ResNet Block。
    使用 'same' padding，让卷积核同时利用前后上下文信息。
    """

    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None):
        super().__init__()

        # Kernel size 3, padding = dilation 保证了输入输出长度一致 (Same Padding)
        padding = dilation
        self.norm_type = norm

        # 定义 Norm 层
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(n_in)
            self.norm2 = nn.BatchNorm1d(n_in)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # 定义 Activation
        if activation == "relu":
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif activation == "silu":
            self.act1 = NonLinearity()
            self.act2 = NonLinearity()
        elif activation == "gelu":
            self.act1 = nn.GELU()
            self.act2 = nn.GELU()

        # 核心卷积：普通 Conv1d，左右均匀填充
        self.conv1 = nn.Conv1d(n_in, n_state, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_orig = x

        # Block 1
        if self.norm_type == "LN":
            x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
        else:
            x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)

        # Block 2
        if self.norm_type == "LN":
            x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
        else:
            x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)

        return x + x_orig


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, activation='relu', norm=None):
        super().__init__()
        blocks = []
        # 堆叠 ResBlocks，dilation 指数增长以扩大感受野
        for depth in range(n_depth):
            dilation = dilation_growth_rate ** depth
            blocks.append(ResConv1DBlock(n_in, n_in, dilation=dilation, activation=activation, norm=norm))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# --- 核心网络 ---
class WindowEncoder(nn.Module):
    def __init__(self, input_dim=272, window_size=32, hidden_size=1024, latent_dim=16,
                 down_t=3, stride=2, depth=3, dilation_growth_rate=3, activation='relu', norm='BN'):
        super().__init__()

        self.window_size = window_size
        blocks = []

        # 1. 初始特征提取
        current_dim = input_dim
        blocks.append(nn.Conv1d(current_dim, hidden_size, kernel_size=3, padding=1))
        blocks.append(nn.ReLU())

        current_dim = hidden_size
        current_time = window_size

        # 2. 下采样 + ResNet 特征提取
        # 每次循环时间维度减半 (stride=2)
        for i in range(down_t):
            blocks.append(nn.Sequential(
                nn.Conv1d(current_dim, hidden_size, kernel_size=4, stride=stride, padding=1),
                # kernel=4, stride=2, pad=1 -> exact half
                Resnet1D(hidden_size, depth, dilation_growth_rate, activation=activation, norm=norm)
            ))
            current_time = current_time // stride

        self.feature_extractor = nn.Sequential(*blocks)

        # 3. 计算 Flatten 后的维度
        # [Batch, Hidden, T_final] -> Flatten -> [Batch, Hidden * T_final]
        self.flat_dim = hidden_size * current_time

        # 4. 映射到 Latent Space (Mean & LogVar)
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_var = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        # x: [B, Input_Dim, T]
        x = self.feature_extractor(x)  # [B, Hidden, T_small]
        x = x.flatten(1)  # [B, Hidden * T_small]

        mu = self.fc_mu(x)  # [B, Latent]
        logvar = self.fc_var(x)  # [B, Latent]
        return mu, logvar


class WindowDecoder(nn.Module):
    def __init__(self, output_dim=272, window_size=32, hidden_size=1024, latent_dim=16,
                 down_t=2, stride=2, depth=3, dilation_growth_rate=3, activation='relu', norm='BN'):
        super().__init__()

        self.hidden_size = hidden_size

        # 计算最底层的时序长度
        self.t_bottom = window_size // (stride ** down_t)
        self.flat_dim = hidden_size * self.t_bottom

        # 1. 从 Latent 映射回 Flatten 特征
        self.fc_proj = nn.Linear(latent_dim, self.flat_dim)

        # 2. 上采样 + ResNet 重建
        blocks = []

        # 镜像 Encoder 的结构
        for i in range(down_t):
            blocks.append(nn.Sequential(
                Resnet1D(hidden_size, depth, dilation_growth_rate, activation=activation, norm=norm),
                nn.Upsample(scale_factor=stride, mode='nearest'),  # 或者用 TransposeConv
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
            ))

        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(hidden_size, output_dim, kernel_size=3, padding=1))

        self.reconstructor = nn.Sequential(*blocks)

    def forward(self, z):
        # z: [B, Latent]
        x = self.fc_proj(z)  # [B, Flat_Dim]
        x = x.view(-1, self.hidden_size, self.t_bottom)  # [B, Hidden, T_bottom]
        x = self.reconstructor(x)  # [B, Output_Dim, T_original]
        return x
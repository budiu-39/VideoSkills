import torch
import torch.nn as nn
from motor_vae.cnn_models import WindowEncoder, WindowDecoder

class PolicySkillTreeVAE(nn.Module):
    def __init__(self,
                 state_dim=272,
                 action_dim=12,
                 window_size=32,
                 hidden_dim=512,
                 latent_dim=32):
        super().__init__()

        # 1. 输入维度修正：直接使用 state_dim
        # 你的 272 维数据已经包含了 Pos 和 Vel，Conv1d 会自动处理时序依赖推导动力学特征
        self.input_dim = state_dim
        self.state_dim = state_dim

        # 2. 输出：State + Action
        # Decoder 的任务：重建状态 (Pos+Vel) 并 预测对应的 Action
        self.output_dim = state_dim + action_dim

        # Encoder
        self.encoder = WindowEncoder(self.input_dim, window_size, hidden_dim, latent_dim)

        # Decoder
        self.decoder = WindowDecoder(self.output_dim, window_size, hidden_dim, latent_dim)

    def preprocess(self, x_state):
        """
        简化预处理：只进行维度置换，不再额外计算差分
        Input:  [B, T, D]
        Output: [B, D, T] (适配 Conv1d)
        """
        # 只需要转置维度，不需要额外计算 Vel/Acc
        return x_state.permute(0, 2, 1).float()

    def forward(self, x_state):
        # x_state: [B, T, 272]

        # 1. Encode
        x_in = self.preprocess(x_state)  # [B, 272, T]
        mu, logvar = self.encoder(x_in)
        z = self.reparameterize(mu, logvar)

        # 2. Decode
        out = self.decoder(z).permute(0, 2, 1)  # [B, T, 272 + Action_Dim]

        # 3. Split Output
        recon_state = out[..., :self.state_dim]
        pred_action = out[..., self.state_dim:]

        return recon_state, pred_action, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def get_skill_embedding(self, x_state):
        """
        Inference / Clustering
        """
        self.eval()
        with torch.no_grad():
            x_in = self.preprocess(x_state)
            mu, _ = self.encoder(x_in)
        return mu   # 这就是该样本在“技能树”上的坐标
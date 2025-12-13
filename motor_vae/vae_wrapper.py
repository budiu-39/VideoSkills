# videoskills/utils/vae_utils.py

import torch
import os
import yaml
from motor_vae import PolicySkillTreeVAE, StandardScaler


class VAEInterface:
    def __init__(self, config_path, model_ckpt_path, scaler_stats_path, device):
        self.device = device

        # 1. Load Config (这里简化处理，假设你知道维度)
        # 实际工程中建议用 yaml.safe_load 读取 config_path
        config = yaml.safe_load(open(config_path, 'r'))
        self.window_size = config['window_size']
        self.state_dim = config['state_dim']
        self.latent_dim = config['latent_dim']
        self.hidden_dim = config['hidden_dim']

        # 2. Load Scaler
        self.scaler = StandardScaler(device=device)
        stats = torch.load(scaler_stats_path, map_location=device)
        self.scaler.mean_state = stats["mean_state"].to(device)
        self.scaler.std_state = stats["std_state"].to(device)
        # Action stats phase 1.2 用不到，因为我们只重建 state

        # 3. Load Model
        self.model = PolicySkillTreeVAE(
            state_dim=self.state_dim,
            window_size=self.window_size,
            hidden_dim=self.hidden_dim ,  # 需与 config 一致
            latent_dim=self.latent_dim,
            down_t=config.get('down_t', 3)  # 需与 config 一致
        ).to(device)

        checkpoint = torch.load(model_ckpt_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode_motion(self, motion_window):
        """
        Input: [B, T, D] Raw Physics State
        Output: [B, Latent_Dim] (Mean of distribution)
        """
        # Normalize
        norm_state = self.scaler.transform_state(motion_window)
        # Encode
        x_in = self.model.preprocess(norm_state)
        mu, _ = self.model.encoder(x_in)
        return mu

    def decode_latent(self, z):
        """
        Input: [B, Latent_Dim]
        Output: [B, T, D] Raw Physics State (Reconstructed)
        """
        out = self.model.decoder(z).permute(0, 2, 1)
        recon_state_norm = out[..., :self.state_dim]
        # Denormalize
        recon_state = self.scaler.inverse_transform_state(recon_state_norm)
        return recon_state

    def process_full_trajectory(self, full_motion_tensor):
        """
        预处理整条轨迹：
        1. 将长轨迹 [L, D] 切分为窗口 [N, 4, D] (Stride=1 或 4)
        2. Encode 得到 Z [N, 32]
        3. Decode 得到 Ref [N, 4, D]

        为了 Phase 1.2 模仿训练，我们希望 Policy 每一帧都有一个 z 指导。
        策略：使用滑动窗口 (Stride=1)。
        对于第 t 帧，我们使用窗口 m[t : t+4] 生成的 z_t。
        """
        L, D = full_motion_tensor.shape
        if L < self.window_size:
            return None, None

        # Unfold creating sliding windows: [L-Window+1, Window, D]
        # 注意: Unfold dimension 0
        windows = full_motion_tensor.unfold(0, self.window_size, 1).permute(0, 2, 1)

        # Batch processing to avoid OOM
        batch_size = 1024
        z_list = []
        recon_list = []

        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = windows[i: i + batch_size]

                # 1. Encode
                z = self.encode_motion(batch)  # [B, 32]
                z_list.append(z)

                # 2. Decode (For Reward Calculation)
                # 我们需要 Policy 追踪 VAE 的重建结果，而不是原始数据
                recon = self.decode_latent(z)  # [B, 4, D]

                # 这里的 recon 是整个窗口。对于 t 时刻，我们只取窗口的第一帧作为 target?
                # 或者取窗口的中心帧？
                # 标准做法：Phase 1.2 中，为了对齐，我们通常取窗口的第一帧作为当前帧的 Target。
                recon_list.append(recon[:, 0, :])

        Z_full = torch.cat(z_list, dim=0)  # [L-3, 32]
        Recon_full = torch.cat(recon_list, dim=0)  # [L-3, D]

        # 补齐最后几帧 (简单复制最后一行)
        pad_len = self.window_size - 1
        Z_full = torch.cat([Z_full, Z_full[-1:].repeat(pad_len, 1)], dim=0)
        Recon_full = torch.cat([Recon_full, Recon_full[-1:].repeat(pad_len, 1)], dim=0)

        return Z_full, Recon_full
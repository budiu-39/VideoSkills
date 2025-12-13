import torch
from videoskills.utils.motion_lib import MotionLib
from motor_vae.vae_wrapper import VAEInterface  # 刚才定义的工具类


class MotionLibZ(MotionLib):
    def __init__(self, motion_file, dof_body_ids, dof_offsets, key_body_ids,
                 device, vae_cfg, rotate_motion=True):
        """
        Args:
            vae_cfg: 包含 VAE 配置的字典或对象 (ckpt_path, stats_path 等)
        """
        # 1. 调用父类加载原始运动数据 (Raw GT)
        # 这会填充 self._motion_data (通常包含 root_pos, root_rot, dof_pos 等)
        super().__init__(motion_file, dof_body_ids, dof_offsets,
                         key_body_ids, device, rotate_motion)

        # 2. 初始化 VAE 接口
        self.vae = VAEInterface(
            config_path=None,
            model_ckpt_path=vae_cfg.ckpt_path,
            scaler_stats_path=vae_cfg.stats_path,
            device=device
        )
        self.vae.model.eval()

        # 3. 执行预处理：Raw Motion -> Z & Reconstructed Motion
        self._latents = None  # 用于存储所有动作的 z
        self.preprocess_motions_with_vae()

    def preprocess_motions_with_vae(self):
        print(f"Processing {self.num_motions()} motions with VAE (Window Size: {self.vae.window_size})...")

        reconstructed_data_list = []
        latents_list = []

        # 遍历每条动作
        # 注意：MotionLib 内部通常把所有动作拼成了一个巨大的 tensor，并用 start_indices 索引
        # 我们最好还是按条处理

        for i in range(self.num_motions()):
            # A. 获取单条动作的原始数据 [T, D_raw]
            # 这里的 get_full_motion_tensor 需要你自己根据 MotionLib 的存储结构实现
            # 假设父类存储在 self._motion_data 且是展平的
            start = self._motion_start_indices[i]
            length = self._motion_lengths[i]
            # 注意：MotionLib 里的 data 通常是 [root_pos(3), root_rot(4), dof_pos(N), ...]
            # 你需要确保这里提取出的维度和 VAE 训练时的 state_dim 一致！
            # 如果不一致，你需要在 VAEInterface 里写一个 converter
            raw_motion = self._get_motion_raw_data(start, length)

            # B. 送入 VAE 计算 (Sliding Window)
            # z_seq: [T, 32], recon_seq: [T, State_Dim]
            z_seq, recon_seq = self.vae.process_full_trajectory(raw_motion)

            # C. 存起来
            latents_list.append(z_seq)
            reconstructed_data_list.append(recon_seq)

            # 打印进度
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} / {self.num_motions()} motions.")

        # 4. 拼接并覆盖
        # 将 Z 拼成一个大 Tensor (和 _motion_data 平行)
        self._latents = torch.cat(latents_list, dim=0).to(self.device)

        # 将 Reconstructed Motion 拼成大 Tensor
        # ★关键★：用 VAE 重建的动作覆盖原始 GT 数据
        # 这样 Environment 在计算 Reward 时，self.ref_body_pos 拿到的就是 VAE 认为“正确”的动作
        recon_flat = torch.cat(reconstructed_data_list, dim=0).to(self.device)

        # 确保维度匹配 (安全检查)
        if recon_flat.shape != self._motion_data.shape:
            print(f"[Warning] VAE recon shape {recon_flat.shape} != Original shape {self._motion_data.shape}")
            print(
                "请检查 VAE 的 State Dim 定义是否包含 root+dof。如果 VAE 只输出了 features，你需要在这里转换回 raw pose。")
            # 如果 VAE 输出的是 Feature (local pos等)，你需要 inverse kinematics 或 inverse feature extraction
            # 这里假设 VAE 训练时用的就是 MotionLib 的那种 Raw State 格式

        self._motion_data = recon_flat
        print("MotionLibZ: Preprocessing complete. GT replaced with VAE Reconstructions.")

    def get_motion_state(self, motion_ids, motion_times):
        """
        重写：除了返回物理状态，还要返回 z
        """
        # 调用父类获取物理状态 (已经是 Reconstruction 了)
        state = super().get_motion_state(motion_ids, motion_times)

        # 获取 z
        # 计算全局帧索引
        frame_indices = self.get_frame_indices(motion_ids, motion_times)  # 假设父类有这个辅助函数
        global_indices = self._motion_start_indices[motion_ids] + frame_indices

        # 从 _latents 查找
        z = self._latents[global_indices]

        # 将 z 放入返回字典，或者直接返回
        # 为了不破坏父类签名，我们可以把 z 塞进 state 字典
        state["z"] = z
        return state

    def _get_motion_raw_data(self, start_idx, length):
        """
        Helper: 从扁平的 _motion_data 中切片出一条动作
        """
        # 假设 _motion_data 是 [Total_Frames, Dim]
        # 且 dt 已经在 load 时处理好了，这里 index 就是 frame index
        # 注意: length 是 float (秒)，需要转成 int (帧数)
        # 如果 self._motion_lengths 存的是秒：
        num_frames = int(length / self._dt + 0.5)  # self._dt 来自父类

        # 安全切片
        return self._motion_data[start_idx: start_idx + num_frames]

    def compute_272_features(self):
        """
        利用现有的 Global Tensors (24 joints) 批量计算 272D 特征 (22 joints)。
        步骤：
        1. 筛选关节 (24 -> 22)
        2. 转换四元数格式 (xyzw -> wxyz)
        3. 去重定位 (Canonicalize)
        4. 拼接特征
        """
        device = self._device

        # =========================================================
        # 1. 关节筛选 (24 -> 22)
        # =========================================================
        # 原始 SMPL Sim 顺序 (24 joints):
        # 0-17: Body + L_Wrist
        # 18: L_Hand (Drop)
        # 19-22: R_Thorax + ... + R_Wrist
        # 23: R_Hand (Drop)

        # 需要保留的索引
        keep_idxs = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,  # 0-17
            19, 20, 21, 22  # 19-22
        ]
        # 转为 tensor 以便索引
        keep_idxs_tensor = torch.tensor(keep_idxs, device=device, dtype=torch.long)

        # 切片数据 (N, 24, 3) -> (N, 22, 3)
        pos = self.gts[:, keep_idxs_tensor, :]
        vel = self.gvs[:, keep_idxs_tensor, :]

        # 旋转数据 (N, 24, 4) -> (N, 22, 4)
        # 注意: MotionLib/IsaacGym 默认四元数格式是 [x, y, z, w]
        rot_xyzw = self.grs[:, keep_idxs_tensor, :]

        # 转换为 PyTorch3D 需要的 [w, x, y, z]
        rot_wxyz = rot_xyzw[..., [3, 0, 1, 2]]

        # 根节点位置 (用于归中)
        root_pos = pos[:, 0, :]  # (N, 3)

        # =========================================================
        # 2. 计算 Heading (基于 22 关节的 Hips)
        # =========================================================
        # 在 24 关节中，L_Hip=1, R_Hip=5
        # 在保留的 22 关节中，索引没有变 (因为 18 之前都被保留了)
        l_hip_idx, r_hip_idx = 1, 5

        across_vec = pos[:, r_hip_idx] - pos[:, l_hip_idx]
        across_vec = across_vec / torch.norm(across_vec, dim=-1, keepdim=True)

        up_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).expand_as(across_vec)
        forward_vec = torch.cross(up_axis, across_vec, dim=-1)
        forward_vec = forward_vec / torch.norm(forward_vec, dim=-1, keepdim=True)

        heading_angles = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])

        # =========================================================
        # 3. 构造去朝向矩阵 & 基础计算
        # =========================================================
        c = torch.cos(-heading_angles)
        s = torch.sin(-heading_angles)
        zeros = torch.zeros_like(c)
        ones = torch.ones_like(c)

        # R_inv: (N, 3, 3)
        r1 = torch.stack([c, -s, zeros], dim=-1)
        r2 = torch.stack([s, c, zeros], dim=-1)
        r3 = torch.stack([zeros, zeros, ones], dim=-1)
        R_inv = torch.stack([r1, r2, r3], dim=-2)

        # =========================================================
        # 4. 生成各部分特征
        # =========================================================

        # --- A. Root Linear Velocity (Indices 0-2) ---
        # 使用 MotionLib 计算好的 Global Root Velocity (N, 3)
        root_vel_global = self.grvs.clone()
        root_vel_local = torch.matmul(R_inv, root_vel_global.unsqueeze(-1)).squeeze(-1)
        feat_root_vel = root_vel_local[:, :2]  # X, Y

        # --- B. Root Angular Velocity (Indices 2-8) ---
        # 计算 Heading Diff
        heading_diff = heading_angles.clone()
        # Diff = Curr - Prev
        # 注意: MotionLib 是拼接的大 Tensor，直接减会导致不同 Motion 连接处出错
        # 但通常 RL 环境会在 episode 结束时 reset，所以这一帧的错误通常可接受
        # 或者你可以利用 self.length_starts 来 mask 掉连接处
        heading_diff[1:] = heading_angles[1:] - heading_angles[:-1]
        heading_diff[0] = 0

        start_idxs = self.length_starts.long()

        # 强制将所有动作第一帧的 Heading 变化设为 0
        # 否则这里会包含 (Current_Start - Prev_End) 的错误数值
        heading_diff[start_idxs] = 0
        # =========================================================

        # Wrap angle -pi to pi
        heading_diff = (heading_diff + torch.pi) % (2 * torch.pi) - torch.pi

        # Diff -> 6D Rot
        hd_c = torch.cos(heading_diff)
        hd_s = torch.sin(heading_diff)
        hd_r1 = torch.stack([hd_c, -hd_s, zeros], dim=-1)
        hd_r2 = torch.stack([hd_s, hd_c, zeros], dim=-1)
        hd_r3 = torch.stack([zeros, zeros, ones], dim=-1)
        R_diff = torch.stack([hd_r1, hd_r2, hd_r3], dim=-2)

        from pytorch3d.transforms import matrix_to_rotation_6d, quaternion_to_matrix
        feat_heading_vel = matrix_to_rotation_6d(R_diff)  # (N, 6)

        # --- C. Local Joint Positions (Indices 8-74) ---
        # 减去根节点水平位置
        root_xy_offset = root_pos.clone()
        root_xy_offset[:, 2] = 0
        pos_centered = pos - root_xy_offset.unsqueeze(1)

        # 旋转去朝向
        pos_local = torch.matmul(
            R_inv.unsqueeze(1),
            pos_centered.unsqueeze(-1)
        ).squeeze(-1)
        feat_pos = pos_local.reshape(pos_local.shape[0], -1)  # (N, 66)

        # --- D. Local Joint Velocities (Indices 74-140) ---
        # 旋转去朝向
        vel_local = torch.matmul(
            R_inv.unsqueeze(1),
            vel.unsqueeze(-1)
        ).squeeze(-1)
        feat_vel = vel_local.reshape(vel_local.shape[0], -1)  # (N, 66)

        # --- E. Joint Rotations (Indices 140-272) ---
        # Quat(wxyz) -> Matrix
        rot_mat = quaternion_to_matrix(rot_wxyz)

        # 旋转去朝向: R_local = R_inv @ R_global
        rot_local_mat = torch.matmul(
            R_inv.unsqueeze(1),
            rot_mat
        )
        feat_rot = matrix_to_rotation_6d(rot_local_mat).reshape(rot_local_mat.shape[0], -1)  # (N, 132)

        # =========================================================
        # 5. 拼接
        # =========================================================
        self.features_272 = torch.cat([
            feat_root_vel,  # 2
            feat_heading_vel,  # 6
            feat_pos,  # 66
            feat_vel,  # 66
            feat_rot  # 132
        ], dim=-1)

        print(f"[MotionLib] Computed 272D features from 24-joint source. Output Shape: {self.features_272.shape}")
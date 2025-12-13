
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from videoskills import LEGGED_GYM_ROOT_DIR
from videoskills.envs.base.legged_robot_config import LeggedRobotCfg
from videoskills.envs.base.legged_robot import LeggedRobot
from videoskills.utils.motion_lib import MotionLib
from videoskills.utils.torch_utils import to_torch, quat_mul, quat_conjugate, quat_to_angle_axis
from videoskills.utils.torch_utils import calc_heading_quat_inv, calc_heading_quat, quat_apply, quat_to_tan_norm
from videoskills.utils.torch_utils import exp_map_to_quat
from videoskills.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from torch import Tensor
from videoskills.envs.base.legged_robot_imi import LeggedRobotImi

class LeggedRobotImiZ(LeggedRobotImi):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.d_z = 64
        self.z_global_buf = torch.zeros(self.num_envs, self.d_z, device=self.device)
        self.current_ctx_key = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._z_provider = None
        self.local_phase_buf = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_win_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.ctx_center_frame = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.ctx_center_time = torch.zeros(self.num_envs, device=self.device)
        self.last_win_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.z_stride_frames = getattr(self.cfg.env, "z_stride_frames", 16)
        self.z_window_K = getattr(self.cfg.env, "z_window_K", 32)

    def set_z_provider(self, fn):
        self._z_provider = fn

    def _update_phase_buf(self, env_ids=None):
        """根据当前轨迹时间与长度，计算 [sin, cos] 并写入 phase_buf。"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # 当前参考时间 t_now 与长度 L
        progress = (self.episode_length_buf[env_ids].float() + 1) * self.dt
        t_now = progress + self._motion_start_times[env_ids]  # [B]
        L = self._motion_lib.get_motion_length(self._sampled_motion_ids[env_ids])  # [B]

        # 归一化到 [0,1)
        u = torch.remainder(t_now, L) / (L + 1e-8)  # [B]
        angle = 2.0 * torch.pi * u
        self.phase_buf[env_ids, 0] = torch.sin(angle)  # sin
        self.phase_buf[env_ids, 1] = torch.cos(angle)  # cos

    def _reset_ref_state_init(self, env_ids):
        super()._reset_ref_state_init(env_ids)
        if self._z_provider is not None and len(env_ids) > 0:
            z_vals, aux = self._z_provider(self, env_ids)  # 在线编码器前向
            # 形状: z_vals [len(env_ids), d_z]
            self.z_global_buf[env_ids] = z_vals.detach()
        return

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        # 重置窗口索引
        self.last_win_idx[env_ids] = -1

        if self._z_provider is not None and len(env_ids) > 0:
            z_vals, aux = self._z_provider(self, env_ids)
            self.z_global_buf[env_ids] = z_vals.detach()
            if 'ctx_keys' in aux:
                self.current_ctx_key[env_ids] = aux['ctx_keys'].to(self.device, dtype=torch.long, non_blocking=True)

            # 初始化中心（以当前帧所在窗口）
            dt = self.dt
            stride = self.z_stride_frames
            frame_idx_now = torch.floor(((self.episode_length_buf[env_ids].float() + 1) * dt +
                                         self._motion_start_times[env_ids]) / dt).long()
            center_frame, win_idx = self.compute_center_frame(frame_idx_now, stride)

            self.ctx_center_frame[env_ids] = center_frame
            self.ctx_center_time[env_ids] = center_frame.float() * dt + self._motion_start_times[env_ids]
            self.last_win_idx[env_ids] = win_idx

        # 初始 local phase
        self._update_local_phase_buf(env_ids)

    def compute_center_frame(self, frame_idx_now: torch.Tensor, stride: int) -> torch.Tensor:
        # frame_idx_now: [B]（motion 局部帧索引，已含 start_time 对齐）
        win_idx = (frame_idx_now // stride).clamp(min=0)
        center = win_idx * stride + (stride // 2)
        return center, win_idx

    def build_context_tensor(self, env_ids, use_center_window=True):
        """
        返回:
          ctx:         [B, C_ctx, K]  逐帧拼接后的上下文特征
          prior_stats: [B, 2*d_z]     没有先验就全 0（占位）
          mids:        List[int]      窗口级 key（建议: motion_id#window_idx）
          mask:        [B, K]         0/1，有需要可做 pad；若用clamp不pad则全 1
        """
        device = self.device
        dt = self.dt
        B = len(env_ids)
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        K = self.z_window_K

        # 1) 取当前帧对应的 “参考轨迹时间” t_now
        motion_ids = self._sampled_motion_ids[env_ids]  # [B]
        t_now = (self.episode_length_buf[env_ids].float() + 1) * dt + self._motion_start_times[env_ids]  # [B]
        motion_lens = self._motion_lib.get_motion_length(motion_ids)  # [B]

        # 2) 构造 K 帧时间窗口（中心对齐；如需因果，可改成全向后）
        if use_center_window:
            offsets = torch.arange(-(K // 2), K - (K // 2), device=device, dtype=torch.float32) * dt  # [K]
            times = t_now[:, None] + offsets[None, :]  # [B,K]
        else:
            # 纯过去窗口（因果）
            offsets = torch.arange(-K + 1, 1, device=device, dtype=torch.float32) * dt
            times = t_now[:, None] + offsets[None, :]

        # 3) clamp 到合法区间，或你也可以选择 pad + mask
        eps = 1e-6
        valid = (times >= 0.0) & (times < motion_lens.unsqueeze(1))  # [B,K] bool
        max_t = (motion_lens - eps).unsqueeze(1)  # [B,1]
        times = torch.clamp(times, min=torch.zeros_like(times),  # [B,K]
                            max=max_t)

        # 4) 展开成一维批量，批量查询 motion_state
        ids_exp = motion_ids[:, None].expand(B, K).reshape(-1)  # [B*K]
        t_flat = times.reshape(-1)  # [B*K]
        ms = self._motion_lib.get_motion_state(ids_exp, t_flat)  # dict of [B*K, ...]

        # 还原为 [B,K,...]
        def BK(x, last_dim):
            return x.view(B, K, last_dim)

        root_pos = BK(ms['root_pos'], 3)
        root_rot = BK(ms['root_rot'], 4)
        root_vel = BK(ms['root_vel'], 3)
        root_ang_vel = BK(ms['root_ang_vel'], 3)
        dof_pos = BK(ms['dof_pos'], self.dof_pos.shape[1])
        dof_vel = BK(ms['dof_vel'], self.dof_vel.shape[1])
        key_pos = BK(ms['key_pos'].view(-1, self.num_bodies, 3), self.num_bodies * 3)
        key_rot = BK(ms['key_rot'].view(-1, self.num_bodies, 4), self.num_bodies * 4)
        key_vel = BK(ms['key_vel'].view(-1, self.num_bodies, 3), self.num_bodies * 3)
        key_ang_vel = BK(ms['key_ang_vel'].view(-1, self.num_bodies, 3), self.num_bodies * 3)

        # 5) 参考你已有的 JIT 特征构造：做 heading 归一化（保证可泛化）
        #    你在 compute_*_observations 里用过 calc_heading_quat_inv / quat_apply 等工具。:contentReference[oaicite:3]{index=3}
        heading_inv = calc_heading_quat_inv(root_rot.reshape(B * K, 4)).view(B, K, 4)
        # 以 root 为局部参考系，key_pos 等做局部化：
        base_pos_expand = root_pos

        def local_vec(v3_flat):  # v3_flat: [B,K,3*nb]
            v3 = v3_flat.view(B, K, -1, 3)
            hinv = heading_inv.unsqueeze(2).expand_as(torch.zeros(B, K, v3.shape[2], 4, device=device))
            v_local = quat_apply(hinv, v3).reshape(B, K, -1)
            return v_local

        key_pos_local = local_vec(key_pos)
        key_vel_local = local_vec(key_vel)
        key_ang_vel_local = key_ang_vel  # 角速度可选做局部化，同上
        # root 部分：
        root_h = root_pos[..., 2:3]  # [B,K,1]
        root_vel_l = quat_apply(heading_inv.view(B, K, 4), root_vel)  # [B,K,3]
        root_ang_l = quat_apply(heading_inv.view(B, K, 4), root_ang_vel)  # [B,K,3]
        # 关节角：你代码里将 exp-map→quat→tan_norm 做过归一化（可沿用/也可直接用原始 dof_pos）。:contentReference[oaicite:4]{index=4}
        # 这里简化：直接使用 dof_pos, dof_vel；若需要可复用你 AMP 的处理。

        # 6) 逐帧特征拼接 → [B,K,C_per_frame]
        feat_per_frame = torch.cat([
            root_h,  # 1
            root_vel_l,  # 3
            root_ang_l,  # 3
            dof_pos,  # n_dof
            dof_vel,  # n_dof
            key_pos_local,  # 3*(num_bodies)
            key_vel_local  # 3*(num_bodies)
            # (可选) key_ang_vel_local, key_rot 的 tan-norm 等
        ], dim=-1)  # [B,K,C_per_frame]

        feat_per_frame = feat_per_frame * valid.unsqueeze(-1).float()
        mask = valid.float()

        # 7) 转为 [B, C_ctx, K] 作为 encoder 时序输入
        ctx = feat_per_frame.transpose(1, 2).contiguous()  # [B, C_ctx, K]

        # 9) prior 统计（没有就全 0）
        prior_stats = torch.zeros(B, 2 * self.d_z if hasattr(self, 'd_z') else 64, device=device)

        # 10) mids：建议用“motion_id + 窗口索引”的组合，确保窗口级 key 唯一
        #     窗口索引按 stride 离散化（和你未来的 Z 更新/插值对齐）
        frame_idx_now = torch.floor(t_now / dt).long()
        center_frame = frame_idx_now

        mids = [(int(m), int(cf)) for m, cf in zip(motion_ids.tolist(), center_frame.tolist())]

        return ctx, prior_stats, mids, mask

    def post_physics_step(self):
        super().post_physics_step()
        alive = torch.where(~self.reset_buf)[0]
        if alive.numel() > 0 and self._z_provider is not None:
            dt = self.dt
            stride = self.z_stride_frames

            # 当前离散帧
            frame_idx_now = torch.floor(((self.episode_length_buf[alive].float() + 1) * dt +
                                         self._motion_start_times[alive]) / dt).long()
            win_idx_now = (frame_idx_now // stride).clamp(min=0)

            # 跨入新窗口的 env
            changed = alive[win_idx_now != self.last_win_idx[alive]]
            if changed.numel() > 0:
                # 1) 刷新 z
                z_vals, aux = self._z_provider(self, changed)
                self.z_global_buf[changed] = z_vals.detach()
                if 'ctx_keys' in aux:
                    self.current_ctx_key[changed] = aux['ctx_keys'].to(self.device, dtype=torch.long, non_blocking=True)

                # 2) 计算并记录本窗口中心
                frame_idx_now_all = torch.floor(((self.episode_length_buf.float() + 1) * dt +
                                                 self._motion_start_times) / dt).long()
                center_all, win_idx_all = self.compute_center_frame(frame_idx_now_all, stride)

                self.ctx_center_frame[changed] = center_all[changed]
                self.ctx_center_time[changed] = center_all[changed].float() * dt + self._motion_start_times[changed]
                self.last_win_idx[changed] = win_idx_all[changed]


        # 每帧更新 local phase（仅基于 ctx_center_time）
        self._update_local_phase_buf(alive)

    def reset_with_motion_ids(self, motion_ids, random: bool = False):
        # 1) 先让父类完成“按 motion_ids 重置状态 + obs 计算”
        obs = super().reset_with_motion_ids(motion_ids, random)

        # 2) 立刻刷新 z / ctx_keys / 窗口中心 / 局部相位
        if self._z_provider is not None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

            # 2.1 刷新 z
            z_vals, aux = self._z_provider(self, env_ids)
            self.z_global_buf[env_ids] = z_vals.detach()
            if 'ctx_keys' in aux:
                self.current_ctx_key[env_ids] = aux['ctx_keys'].to(
                    self.device, dtype=torch.long, non_blocking=True
                )

            # 2.2 记录当前窗口中心（与 post_physics_step / reset_idx 保持一致）
            dt = self.dt
            stride = self.z_stride_frames
            frame_idx_now = torch.floor(((self.episode_length_buf[env_ids].float() + 1) * dt
                                         + self._motion_start_times[env_ids]) / dt).long()
            center_frame, win_idx = self.compute_center_frame(frame_idx_now, stride)
            self.ctx_center_frame[env_ids] = center_frame
            self.ctx_center_time[env_ids] = center_frame.float() * dt + self._motion_start_times[env_ids]
            self.last_win_idx[env_ids] = win_idx

            # 2.3 用新的中心时间更新 local phase（-1..1 → sin/cos）
            self._update_local_phase_buf(env_ids)

        return obs

    def _update_local_phase_buf(self, env_ids=None):
        """
        local phase = 当前帧相对于“本窗口中心”的相对位置（-1..1），
        再用 sin/cos 编码；不依赖全局相位。
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        dt = self.dt
        K = float(self.z_window_K)

        # 当前参考时间
        t_now = (self.episode_length_buf[env_ids].float() + 1) * dt + self._motion_start_times[env_ids]  # [B]
        # 已记录的窗口中心时间
        t_center = self.ctx_center_time[env_ids]  # [B]

        # 归一化到 [-1, 1]：相对中心的半窗长度 = (K*dt)/2
        half_T = (K * dt) * 0.5
        u = torch.clamp((t_now - t_center) / (half_T + 1e-8), min=-1.0, max=1.0)  # [B]

        # 编码为 sin/cos（取 πu，周期正好覆盖一窗）
        angle = torch.pi * u
        self.local_phase_buf[env_ids, 0] = torch.sin(angle)
        self.local_phase_buf[env_ids, 1] = torch.cos(angle)

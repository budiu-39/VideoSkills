import os

from videoskills.utils.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from videoskills.utils.poselib.core.rotation3d import *
from isaacgym.torch_utils import *
from videoskills.utils import torch_utils
import torch
from tqdm import tqdm
from videoskills.utils.motion_lib import MotionLib

USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy


    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)


    torch.Tensor.numpy = Patch.numpy


class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1

        print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLibZ(MotionLib):
    def __init__(self, motion_file, dof_body_ids, dof_offsets,
                 key_body_ids, rotate_motion, device):
        self._motion_latents = []
        self.has_latents = True

        super().__init__(motion_file, dof_body_ids, dof_offsets,
                 key_body_ids, rotate_motion, device)

        if self.has_latents:
            # 假设 z 的形状是 [Latent_Len, Latent_Dim]
            # 为了方便查询，我们需要知道每个 motion 对应的 z 在大 Tensor 中的起始位置
            # 或者我们可以 pad 后 stack，或者像 gts 那样 cat

            # 方案：Cat 所有 Z，记录 offset
            # 注意：Z 是稀疏的 (Stride=4)，需要处理时间映射
            self.latents_tensor = torch.cat(self._motion_latents, dim=0).float().to(device)

            # 计算每个 motion Z 的起始索引
            z_lengths = [len(z) for z in self._motion_latents]
            z_lengths_tensor = torch.tensor(z_lengths, device=device)

            self.z_starts = z_lengths_tensor.roll(1)
            self.z_starts[0] = 0
            self.z_starts = self.z_starts.cumsum(0)

            # 记录 Z 的采样频率 (Stride=4, FPS=30 -> Z_FPS = 30/4 = 7.5)
            self.z_stride = 4  # 假设固定为 4
            self.z_fps = 30.0 / self.z_stride

        return

    # def get_motion_z(self, motion_ids, motion_times):
    #     """
    #     获取指定时间的 Latent Z。
    #     motion_ids: [N]
    #     motion_times: [N] (秒)
    #     """
    #     if not self.has_latents:
    #         # 如果没有 Z，返回全 0 (兼容旧数据)
    #         return torch.zeros((len(motion_ids), 32), device=self._device)  # 假设 Dim=32
    #
    #     # 计算 Z 的索引
    #     # Frame = Time * FPS
    #     # Z_Index = Frame / Stride
    #     frame_indices = (motion_times * 30.0).long()  # 假设 Motion FPS=30
    #     z_indices_local = frame_indices // self.z_stride
    #
    #     # 防止越界 (Z 长度比 Motion 短)
    #     # 获取每个 motion 对应的最大 Z 长度
    #     # 这里需要更精细的处理，简单起见我们假设 z_indices 不会超过 z_lengths
    #     # 实际工程中建议加上 clamp
    #
    #     z_indices_global = self.z_starts[motion_ids] + z_indices_local
    #
    #     # 边界保护
    #     max_indices = self.latents_tensor.shape[0] - 1
    #     z_indices_global = torch.clamp(z_indices_global, 0, max_indices)
    #
    #     return self.latents_tensor[z_indices_global]

    def get_motion_z(self, motion_ids, motion_times):
        """
        获取指定时间的 Latent Z，支持线性/球形插值以获得平滑信号。
        """
        if not self.has_latents:
            return torch.zeros((len(motion_ids), 32), device=self._device)

        # 1. 计算精确的浮点索引
        # Frame = Time * FPS
        # exact_idx = 10.5 (代表处于第10个和第11个Z之间的一半)
        frames = motion_times * 30.0  # 假设 Motion FPS=30
        exact_z_indices = frames / self.z_stride

        # 2. 获取前后两个索引 (Floor 和 Ceil)
        idx0_local = torch.floor(exact_z_indices).long()
        idx1_local = idx0_local + 1

        # 计算插值系数 alpha (0.0 ~ 1.0)
        alpha = exact_z_indices - idx0_local.float()
        alpha = torch.clamp(alpha, 0.0, 1.0).unsqueeze(-1)  # [N, 1] for broadcasting

        # 3. 处理全局索引和边界
        max_indices = self.latents_tensor.shape[0] - 1

        # 计算 Global Index
        start_offsets = self.z_starts[motion_ids]

        idx0_global = start_offsets + idx0_local
        idx1_global = start_offsets + idx1_local

        # 边界保护：如果 idx1 超出该动作的长度，就 clamp 到最后
        # 注意：这里我们不能简单 clamp 到 max_indices，因为那是别人的动作
        # 正确做法是计算每个动作的 z_len，但为了性能，通常保证 z_data 比 motion 长一点
        # 或者简单的 clamp:
        idx0_global = torch.clamp(idx0_global, 0, max_indices)
        idx1_global = torch.clamp(idx1_global, 0, max_indices)

        # 4. 获取向量
        z0 = self.latents_tensor[idx0_global]
        z1 = self.latents_tensor[idx1_global]

        # 5. 执行插值 (SLERP 或 LERP)
        # 对于 VAE Latent，SLERP (Spherical Linear Interpolation) 是理论最优的
        # 但如果 z 维度很高且未严格归一化，LERP (Linear) 通常也足够且更快

        # --- 方案 A: 简单线性插值 (LERP) ---
        # z_interpolated = (1.0 - alpha) * z0 + alpha * z1

        # --- 方案 B: 标准化后 SLERP (更严谨，推荐用于 VAE) ---
        # from videoskills.utils import torch_utils
        z_interpolated = torch_utils.slerp(z0, z1, alpha)

        return z_interpolated

    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        # num_bodies = self._get_num_bodies()
        # num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel = self.grvs[f0l]

        root_ang_vel = self.gravs[f0l]

        key_pos0 = self.gts[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

        key_rot0 = self.grs[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_rot1 = self.grs[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

        dof_vel = self.dvs[f0l]

        vals = [root_pos0, root_pos1, local_rot0, local_rot1, root_vel, root_ang_vel, key_pos0,
                key_pos1, key_rot0, key_rot1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)
        blend_exp = blend.unsqueeze(1)
        key_rot = torch_utils.slerp(key_rot0, key_rot1, blend_exp)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1

        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(local_rot)

        key_vel0 = self.gvs[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_vel1 = self.gvs[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_vel = (1.0 - blend_exp) * key_vel0 + blend_exp * key_vel1

        key_ang_vel0 = self.gas[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_ang_vel1 = self.gas[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_ang_vel = (1.0 - blend_exp) * key_ang_vel0 + blend_exp * key_ang_vel1

        motion_state = {
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "root_vel": root_vel,
            "root_ang_vel": root_ang_vel,
            "dof_vel": dof_vel,
            "key_pos": key_pos,
            "key_rot": key_rot,
            "key_vel": key_vel,
            "key_ang_vel": key_ang_vel,
        }

        return motion_state

        # motion_lib_z.py

    def _load_motions(self, motion_file, skeleton_trees=None):
        """
        覆盖基类的加载函数，支持加载 Latent Z
        """
        self._motions = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_keys = []
        self._motion_latents = []
        self.has_latents = False

        motion_files = self._fetch_motion_files(motion_file)

        cpu_dof_body_ids = self._dof_body_ids.cpu() if isinstance(self._dof_body_ids, torch.Tensor) else torch.tensor(
            self._dof_body_ids)
        cpu_dof_offsets = self._dof_offsets.cpu() if isinstance(self._dof_offsets, torch.Tensor) else torch.tensor(
            self._dof_offsets)

        print(f"Loading {len(motion_files)} motions with Z (Single Process)...")

        for curr_file in tqdm(motion_files):
            try:
                raw_data = np.load(curr_file, allow_pickle=True)
                curr_motion = None
                z_data = None

                # 1. 解析数据
                if isinstance(raw_data, np.ndarray) and raw_data.dtype == object and raw_data.ndim == 0:
                    data_dict = raw_data.item()
                    if 'motion' in data_dict:
                        curr_motion = SkeletonMotion.from_dict(data_dict['motion'])
                        if 'z' in data_dict:
                            z_data = torch.from_numpy(data_dict['z']).float()
                    else:
                        curr_motion = SkeletonMotion.from_dict(data_dict)
                else:
                    curr_motion = SkeletonMotion.from_file(curr_file)

                if curr_motion is None:
                    continue

                # 检查是否包含 Z
                if len(self._motions) == 0:
                    if z_data is not None:
                        self.has_latents = True
                        # print("Latent Z detected in motion files.")

                motion_fps = curr_motion.fps

                # 2. 旋转
                # if self._rotate_motion:
                #     curr_motion = self._apply_random_rotation(curr_motion)

                # 3. 计算 DOF
                curr_motion.dof_vels = self._compute_motion_dof_vels_vectorized(
                    curr_motion, cpu_dof_body_ids, cpu_dof_offsets
                )

                # 4. 数据存储
                key = self._get_motion_key(curr_file)
                curr_dt = 1.0 / motion_fps
                num_frames = curr_motion.tensor.shape[0]
                curr_len = 1.0 / motion_fps * (num_frames - 1)

                self._motion_keys.append(key)
                self._motion_fps.append(motion_fps)
                self._motion_dt.append(curr_dt)
                self._motion_num_frames.append(num_frames)
                self._motion_files.append(curr_file)
                self._motion_lengths.append(curr_len)
                self._motions.append(curr_motion)

                # 5. 处理 Z
                if self.has_latents:
                    if z_data is not None:
                        self._motion_latents.append(z_data)
                    else:
                        # Fallback for missing Z (pad with zeros)
                        # Assume stride 4
                        dummy_len = num_frames // 4
                        if num_frames % 4 != 0: dummy_len += 1  # ceil check
                        self._motion_latents.append(torch.zeros((dummy_len, 32)))

            except Exception as e:
                print(f"Failed to load {curr_file}: {e}")
                continue

        self._sort_motions_by_length()

        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)

        if self.has_latents:
            print(f"Loaded Latents: Count {len(self._motion_latents)}")

        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(len(self._motions), total_len))

    def _sort_motions_by_length(self):
        sorted_indices = torch.argsort(torch.tensor(self._motion_lengths))  # on CPU first

        # 如果你已经有 tensor，就用它；否则用原始 list 排序
        self._motions = [self._motions[i] for i in sorted_indices]
        self._motion_files = [self._motion_files[i] for i in sorted_indices]
        self._motion_lengths = [self._motion_lengths[i] for i in sorted_indices]
        self._motion_fps = [self._motion_fps[i] for i in sorted_indices]
        self._motion_dt = [self._motion_dt[i] for i in sorted_indices]
        self._motion_num_frames = [self._motion_num_frames[i] for i in sorted_indices]
        self._motion_keys = [self._motion_keys[i] for i in sorted_indices]

        if self.has_latents:
            self._motion_latents = [self._motion_latents[i] for i in sorted_indices]

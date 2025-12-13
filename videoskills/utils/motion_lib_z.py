import os

from videoskills.utils.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from videoskills.utils.poselib.core.rotation3d import *
from isaacgym.torch_utils import *
import xml.etree.ElementTree as ET
from videoskills.utils import torch_utils
import torch
import joblib
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
        self.has_latents = False

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

    def get_motion_z(self, motion_ids, motion_times):
        """
        获取指定时间的 Latent Z。
        motion_ids: [N]
        motion_times: [N] (秒)
        """
        if not self.has_latents:
            # 如果没有 Z，返回全 0 (兼容旧数据)
            return torch.zeros((len(motion_ids), 32), device=self._device)  # 假设 Dim=32

        # 计算 Z 的索引
        # Frame = Time * FPS
        # Z_Index = Frame / Stride
        frame_indices = (motion_times * 30.0).long()  # 假设 Motion FPS=30
        z_indices_local = frame_indices // self.z_stride

        # 防止越界 (Z 长度比 Motion 短)
        # 获取每个 motion 对应的最大 Z 长度
        # 这里需要更精细的处理，简单起见我们假设 z_indices 不会超过 z_lengths
        # 实际工程中建议加上 clamp

        z_indices_global = self.z_starts[motion_ids] + z_indices_local

        # 边界保护
        max_indices = self.latents_tensor.shape[0] - 1
        z_indices_global = torch.clamp(z_indices_global, 0, max_indices)

        return self.latents_tensor[z_indices_global]

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



    def _load_motions(self, motion_file, skeleton_trees = None):
        self._motions = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_keys = []

        total_len = 0.0

        motion_files = self._fetch_motion_files(motion_file)
        # num_motion_files = len(motion_files)
        print(f"Loading {len(motion_files)} motions...")

        for f in tqdm(motion_files):
            data = np.load(f, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object and 'z' in data.item():
                # VAE Z+Motion 格式
                data_dict = data.item()
                z_np = data_dict['z']
                motion_dict = data_dict['motion']

                curr_motion = SkeletonMotion.from_dict(motion_dict)

                # 存储 Z
                self._motion_latents.append(torch.from_numpy(z_np))
                self.has_latents = True

            if self._rotate_motion:
                curr_motion = self.apply_rotation(curr_motion, curr_motion.fps)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_keys.append(self._get_motion_key(f))
            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)

            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            # Moving motion tensors to the GPU
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)
            # else:
            #     curr_motion.tensor = curr_motion.tensor.to(self._device)
            #     curr_motion._skeleton_tree._parent_indices = curr_motion._skeleton_tree._parent_indices.to(self._device)
            #     curr_motion._skeleton_tree._local_translation = curr_motion._skeleton_tree._local_translation.to(
            #         self._device)
            #     curr_motion._rotation = curr_motion._rotation.to(self._device)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)

            self._motion_files.append(f)

        self._sort_motions_by_length()

        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)


        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)

        num_motions = len(self._motions)
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return

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

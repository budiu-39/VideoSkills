import os
import torch
import numpy as np
import joblib
from tqdm import tqdm
from videoskills.utils.poselib.skeleton.skeleton3d import SkeletonMotion
from videoskills.utils import torch_utils
from videoskills.utils.motion_lib import MotionLib
from isaacgym.torch_utils import *


class MotionLibHoi(MotionLib):
    def __init__(self, motion_file, dof_body_ids, dof_offsets,
                 key_body_ids, rotate_motion, device):
        # 1. 临时存储 HOI 特有的 list 容器
        self._obj_pos_list = []
        self._obj_rot_list = []
        self._obj_pos_vel_list = []
        self._obj_rot_vel_list = []
        self._has_object = False
        self._motion_obj_names = []
        self.ig_list = []
        self.contact_robot_list = []
        self.collision_tag_list = []
        self.ig_index_list = []

        # 2. 调用基类构造函数
        # 注意：这会触发 self._load_motions -> self._sort_motions_by_length
        super().__init__(motion_file, dof_body_ids, dof_offsets, key_body_ids, rotate_motion, device)

        # 3. 将加载好的 HOI 数据拼接到 GPU 张量
        self._build_hoi_tensors()
        self._precompute_non_collision_frames()

    def _load_motions(self, motion_file):
        """
        覆盖基类的加载逻辑，支持 HOI Bundle 格式。
        基类 __init__ 会调用此函数。
        """
        self._motions = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_keys = []

        motion_files = self._fetch_motion_files(motion_file)

        cpu_dof_body_ids = self._dof_body_ids.cpu() if isinstance(self._dof_body_ids, torch.Tensor) else torch.tensor(
            self._dof_body_ids)
        cpu_dof_offsets = self._dof_offsets.cpu() if isinstance(self._dof_offsets, torch.Tensor) else torch.tensor(
            self._dof_offsets)

        print(f"[MotionLibHoi] Loading {len(motion_files)} HOI motions...")

        for curr_file in tqdm(motion_files):
            payload = np.load(curr_file, allow_pickle=True).item()
            if "motion" in payload:
                curr_motion = SkeletonMotion.from_dict(payload["motion"])
            else:
                # 非 HOI 数据，调用基类加载逻辑
                curr_motion = SkeletonMotion.from_file(curr_file)

            if self._rotate_motion:
                curr_motion = self.apply_rotation(curr_motion, curr_motion.fps)

            # 计算 DOF 速度
            curr_motion.dof_vels = self._compute_motion_dof_vels_vectorized(
                curr_motion, cpu_dof_body_ids, cpu_dof_offsets
            )
            num_f = curr_motion.tensor.shape[0]
            if "object" in payload:
                objd = payload["object"]
                hoi = payload["interaction"]
                self._has_object = True
            else:
                # 伪造全零物体数据 (Stage 1 使用)
                objd = {
                    "obj_pos": np.zeros((num_f, 3)),
                    "obj_rot": np.tile(np.array([0, 0, 0, 1]), (num_f, 1)),  # 单位四元数
                    "obj_pos_vel": np.zeros((num_f, 3)),
                    "obj_rot_vel": np.zeros((num_f, 3)),
                    "name": "none"
                }
                hoi = {
                    "ig": np.zeros((num_f, 52, 3)),  # 假设 52 个 body
                    "contact_robot": np.zeros((num_f, 52)),
                    "collision_tag": np.zeros((num_f, 52)),
                    "ig_index": np.zeros((num_f, 52), dtype=np.int64)  # [NEW] 伪造索引
                }

            to_f32 = lambda a: torch.as_tensor(a, dtype=torch.float32)
            to_long = lambda a: torch.as_tensor(a, dtype=torch.long)

            self._obj_pos_list.append(to_f32(objd["obj_pos"]))
            self._obj_rot_list.append(to_f32(objd["obj_rot"]))
            self._obj_pos_vel_list.append(to_f32(objd["obj_pos_vel"]))
            self._obj_rot_vel_list.append(to_f32(objd["obj_rot_vel"]))
            self.ig_list.append(to_f32(hoi["ig"]))
            self.contact_robot_list.append(to_f32(hoi["contact_robot"]))
            self.collision_tag_list.append(torch.as_tensor(hoi["collision_tag"], dtype=torch.bool))
            if "ig_index" in hoi:
                self.ig_index_list.append(to_long(hoi["ig_index"]))
                self.has_ig_index = True
            else:
                # 如果旧数据没有 ig_index，默认全0
                self.ig_index_list.append(torch.zeros((num_f, 52), dtype=torch.long))
                self.has_ig_index = False
            self._motion_obj_names.append(objd.get("name", "unknown"))

            # 基础信息存储
            motion_fps = curr_motion.fps
            num_frames = curr_motion.tensor.shape[0]
            self._motions.append(curr_motion)
            self._motion_fps.append(motion_fps)
            self._motion_dt.append(1.0 / motion_fps)
            self._motion_num_frames.append(num_frames)
            self._motion_lengths.append((num_frames - 1) / motion_fps)
            self._motion_files.append(curr_file)
            self._motion_keys.append(self._get_motion_key(curr_file))
            self._has_object = True


        # 重要：在此处进行排序并转换为 Tensor，否则基类 __init__ 后半部分会报错
        self._sort_motions_by_length()

    def _sort_motions_by_length(self):
        """
        覆盖基类的排序函数，同时处理 HOI 特有的列表。
        """
        if len(self._motion_lengths) == 0:
            return

        # 计算排序索引
        arg_indices = np.argsort(self._motion_lengths)

        # 1. 排序基础列表
        self._motions = [self._motions[i] for i in arg_indices]
        self._motion_files = [self._motion_files[i] for i in arg_indices]
        self._motion_keys = [self._motion_keys[i] for i in arg_indices]
        self._motion_lengths = [self._motion_lengths[i] for i in arg_indices]
        self._motion_fps = [self._motion_fps[i] for i in arg_indices]
        self._motion_dt = [self._motion_dt[i] for i in arg_indices]
        self._motion_num_frames = [self._motion_num_frames[i] for i in arg_indices]

        # 2. 排序 HOI 列表
        if self._has_object:
            self._obj_pos_list = [self._obj_pos_list[i] for i in arg_indices]
            self._obj_rot_list = [self._obj_rot_list[i] for i in arg_indices]
            self._obj_pos_vel_list = [self._obj_pos_vel_list[i] for i in arg_indices]
            self._obj_rot_vel_list = [self._obj_rot_vel_list[i] for i in arg_indices]
            self.ig_list = [self.ig_list[i] for i in arg_indices]
            self.contact_robot_list = [self.contact_robot_list[i] for i in arg_indices]
            self.collision_tag_list = [self.collision_tag_list[i] for i in arg_indices]
            self._motion_obj_names = [self._motion_obj_names[i] for i in arg_indices]
            self.ig_index_list = [self.ig_index_list[i] for i in arg_indices]

        # 3. 核心修复：将 list 转换为 torch.Tensor，确保基类 roll() 不报错
        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device, dtype=torch.long)

    def _build_hoi_tensors(self):
        """拼接 HOI 特有的 GPU 全局张量"""
        if not self._has_object: return
        dev = self._device

        self.obj_pos = torch.cat(self._obj_pos_list, dim=0).to(dev)
        self.obj_rot = torch.cat(self._obj_rot_list, dim=0).to(dev)
        self.obj_pos_vel = torch.cat(self._obj_pos_vel_list, dim=0).to(dev)
        self.obj_rot_vel = torch.cat(self._obj_rot_vel_list, dim=0).to(dev)
        self.ig = torch.cat(self.ig_list, dim=0).to(dev)
        self.contact_robot = torch.cat(self.contact_robot_list, dim=0).to(dev)
        self.collision_tag = torch.cat(self.collision_tag_list, dim=0).to(dev)
        self.ig_index = torch.cat(self.ig_index_list, dim=0).to(dev)

        # 构建物体词表
        uniq_names = sorted(set(self._motion_obj_names))
        self.object_vocab = {name: i for i, name in enumerate(uniq_names)}
        self.object_vocab_inv = {i: name for name, i in self.object_vocab.items()}

        self._motions_by_object = {}
        for name, idx in self.object_vocab.items():
            m_idxs = [i for i, n in enumerate(self._motion_obj_names) if n == name]
            self._motions_by_object[idx] = torch.tensor(m_idxs, dtype=torch.long, device=dev)

    def _precompute_non_collision_frames(self):
        self._valid_frames_no_cg = []
        for i in range(self.num_motions()):
            T_i = int(self._motion_num_frames[i].item())
            start = int(self.length_starts[i].item())
            end = start + T_i
            cr_t = self.collision_tag[start:end]
            non_contact = torch.logical_not(cr_t)
            idxs = torch.nonzero(non_contact, as_tuple=False).squeeze(1).long()
            # 保护：如果全是碰撞帧，则允许采样所有帧
            if idxs.numel() == 0:
                idxs = torch.arange(T_i, device=self._device, dtype=torch.long)
            self._valid_frames_no_cg.append(idxs)

    def has_object(self):
        return self._has_object

    def get_hoi_state(self, motion_ids, motion_times):
        if not self.has_object(): return None

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        # 1. 计算原始 blend [N]
        f0, f1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = f0 + self.length_starts[motion_ids]
        f1l = f1 + self.length_starts[motion_ids]

        # 2. 核心修复：增加维度使其变为 [N, 1] 以支持 slerp 和 position 插值
        blend_exp = blend.unsqueeze(-1)

        # 3. 使用 blend_exp 进行插值
        o_pos = (1.0 - blend_exp) * self.obj_pos[f0l] + blend_exp * self.obj_pos[f1l]

        # 传递 blend_exp 给 slerp
        o_rot = torch_utils.slerp(self.obj_rot[f0l], self.obj_rot[f1l], blend_exp)

        return {
            "obj_pos": o_pos,
            "obj_rot": o_rot,
            "obj_pos_vel": self.obj_pos_vel[f0l],
            "obj_rot_vel": self.obj_rot_vel[f0l],
            "ig": self.ig[f0l],
            "contact_robot": self.contact_robot[f0l],
            "ig_index": self.ig_index[f0l],  # [NEW] 返回索引
            "collision_tag": self.collision_tag[f0l],
        }

    def sample_motions_by_object(self, object_ids: torch.Tensor) -> torch.Tensor:
        B = object_ids.shape[0]
        out = torch.empty(B, dtype=torch.long, device=self._device)
        global_prob = getattr(self, "_sampling_prob", None)

        for b in range(B):
            obj_id = int(object_ids[b].item())
            subset = self._motions_by_object.get(obj_id, None)
            if subset is None or subset.numel() == 0:
                out[b] = torch.multinomial(global_prob, 1)[0] if global_prob is not None else torch.randint(0,
                                                                                                            len(self._motions),
                                                                                                            (1,))
            else:
                prob_sub = global_prob[subset] if global_prob is not None else torch.ones_like(subset)
                prob_sub = prob_sub / prob_sub.sum().clamp_min(1e-8)
                idx_in_subset = torch.multinomial(prob_sub, 1)[0]
                out[b] = subset[idx_in_subset]
        return out

    def sample_time(self, motion_ids, truncate_time=None):
        B = motion_ids.shape[0]
        out_t = torch.empty(B, dtype=torch.float32, device=self._device)
        for b in range(B):
            i = int(motion_ids[b].item())
            dt_i = float(self._motion_dt[i].item())
            valid = self._valid_frames_no_cg[i]
            if truncate_time is not None:
                T_i = int(self._motion_num_frames[i].item())
                max_idx = max(T_i - 1 - int(truncate_time / dt_i), 0)
                valid = valid[valid <= max_idx]
                if valid.numel() == 0: valid = torch.tensor([0], device=self._device)
            pick = valid[torch.randint(0, valid.numel(), (1,), device=self._device)]
            out_t[b] = pick.float() * dt_i
        return out_t
import os

from videoskills.utils.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from videoskills.utils.poselib.core.rotation3d import *
from isaacgym.torch_utils import *
import xml.etree.ElementTree as ET
from videoskills.utils import torch_utils
import torch
import joblib
from tqdm import tqdm
import glob
from videoskills.utils.motion_lib import MotionLib


class MotionLibHoi(MotionLib):
    def __init__(self, motion_file, dof_body_ids, dof_offsets,
                 key_body_ids, rotate_motion, device):
        # === 不调用 super().__init__，自己初始化 ===
        self._rotate_motion = rotate_motion
        self._dof_body_ids = dof_body_ids
        self._dof_offsets = dof_offsets
        self._num_dof = dof_offsets[-1]
        self._key_body_ids = key_body_ids.to(device)
        self._device = device

        # 容器（与父类保持一致的命名）
        self._motions = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_keys = []

        # HOI 容器
        self._obj_pos_list = []
        self._obj_rot_list = []
        self._obj_pos_vel_list = []
        self._obj_rot_vel_list = []
        self._has_object = False
        self._motion_obj_names = []
        self.ig = []  # interaction guide
        self.ig_sk = []
        self.contact_robot = []  # contact map on human (52 body SMPLH/X robot joints)

        # 实际加载
        self._load_motions_hoi(motion_file)

        # === 下面与父类 __init__ 同步：拼接库级张量 ===
        motions = self._motions
        self.gvs  = torch.cat([m.global_velocity            for m in motions], dim=0).float().to(device)
        self.gas  = torch.cat([m.global_angular_velocity    for m in motions], dim=0).float().to(device)
        self.gts  = torch.cat([m.global_translation         for m in motions], dim=0).float().to(device)
        self.grs  = torch.cat([m.global_rotation            for m in motions], dim=0).float().to(device)
        self.lrs  = torch.cat([m.local_rotation             for m in motions], dim=0).float().to(device)
        self.grvs = torch.cat([m.global_root_velocity       for m in motions], dim=0).float().to(device)
        self.gravs= torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(device)
        self.dvs  = torch.cat([m.dof_vels                   for m in motions], dim=0).float().to(device)

        # 采样辅助
        self._termination_history = torch.ones(len(self._motions), dtype=torch.float32, device=self._device)
        self._sampling_prob = torch.ones(len(self._motions), dtype=torch.float32, device=self._device) / max(1, len(self._motions))

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1); lengths_shifted[0] = 0
        self._num_motions = len(self._motions)
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

        self._precompute_non_contact_frames()

    # === HOI 读入：兼容 bundle(dict) 与旧的 motion.npy ===
    def _load_motions_hoi(self, motion_file):
        motion_files = self._fetch_motion_files(motion_file)
        for curr_file in tqdm(motion_files, desc="[HOI] Loading", unit="file"):
            payload = None; is_bundle = False
            try:
                payload = np.load(curr_file, allow_pickle=True)
                if hasattr(payload, "item"):
                    payload = payload.item()
                if isinstance(payload, dict) and "motion" in payload:
                    is_bundle = True
            except Exception:
                payload = None

            # 恢复 SkeletonMotion
            if is_bundle:
                curr_motion = SkeletonMotion.from_dict(payload["motion"])
            else:
                curr_motion = SkeletonMotion.from_file(curr_file)

            if self._rotate_motion:
                curr_motion = self.apply_rotation(curr_motion, curr_motion.fps)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.tensor.shape[0]
            curr_len = (num_frames - 1) / motion_fps

            self._motion_keys.append(self._get_motion_key(curr_file))
            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
            self._motion_files.append(curr_file)


            # dof_vels
            curr_motion.dof_vels = self._compute_motion_dof_vels(curr_motion)

            # 可选缓存
            if 'USE_CACHE' in globals() and USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)

            # 采集 object 通道（若 bundle）
            if is_bundle and ("object" in payload):
                objd = payload["object"] or {}
                hoi = payload['interaction']

                def _to_torch_clip(arr, expected_shape=None, expected_last_dim=None):
                    if arr is None:
                        return None
                    t = torch.as_tensor(arr, dtype=torch.float32)
                    # 截到 motion 帧长
                    if t.shape[0] != num_frames:
                        t = t[:num_frames]
                    if expected_shape is not None:
                        # 严格匹配完整 shape（除 T 维）
                        assert t.shape[
                               1:] == expected_shape, f"expected shape (*,{expected_shape}), got {tuple(t.shape)}"
                    elif expected_last_dim is not None:
                        assert t.shape[
                                   -1] == expected_last_dim, f"expected last_dim={expected_last_dim}, got {t.shape[-1]}"
                    return t
                o_pos = _to_torch_clip(objd.get("obj_pos"), expected_last_dim=3)
                o_rot = _to_torch_clip(objd.get("obj_rot"), expected_last_dim=4)  # xyzw
                o_pos_vel = _to_torch_clip(objd.get("obj_pos_vel"), expected_last_dim=3)
                o_rot_vel = _to_torch_clip(objd.get("obj_rot_vel"), expected_last_dim=3)
                ig = _to_torch_clip(hoi.get("ig"), expected_shape=(52, 3))
                ig_sk = _to_torch_clip(hoi.get("ig_sk"), expected_shape=(52, 3))
                contact_robot = _to_torch_clip(hoi.get("contact_robot"), expected_last_dim=52)
                obj_name = (payload["object"] or {}).get("name", None)

                if (o_pos is not None) and (o_rot is not None):
                    self._obj_pos_list.append(o_pos)
                    self._obj_rot_list.append(o_rot)
                    self._obj_pos_vel_list.append(o_pos_vel if o_pos_vel is not None else torch.zeros((num_frames,3)))
                    self._obj_rot_vel_list.append(o_rot_vel if o_rot_vel is not None else torch.zeros((num_frames,3)))
                    self.ig.append(ig if ig is not None else torch.zeros((num_frames,3)))
                    self.ig_sk.append(ig_sk if ig_sk is not None else torch.zeros((num_frames,3)))
                    self.contact_robot.append(contact_robot if contact_robot is not None else torch.zeros((num_frames,52)))
                    self._motion_obj_names.append(obj_name)
                    self._has_object = True
                else:
                    # 没有 object 内容也保持占位，便于排序与拼接
                    self._obj_pos_list.append(torch.zeros((num_frames,3)))
                    self._obj_rot_list.append(torch.tensor([[0,0,0,1]]).repeat(num_frames,1))
                    self._obj_pos_vel_list.append(torch.zeros((num_frames,3)))
                    self._obj_rot_vel_list.append(torch.zeros((num_frames,3)))


        # 排序（含 HOI 列表）
        self._sort_motions_by_length_hoi()

        # 张量化元信息
        self._motion_lengths    = torch.tensor(self._motion_lengths,    device=self._device, dtype=torch.float32)
        self._motion_fps        = torch.tensor(self._motion_fps,        device=self._device, dtype=torch.float32)
        self._motion_dt         = torch.tensor(self._motion_dt,         device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)

        # 拼接 object 到库级（若存在）
        if self._has_object:
            self.obj_pos = torch.cat(self._obj_pos_list, dim=0).to(self._device)  # (ΣT,3)
            self.obj_rot = torch.cat(self._obj_rot_list, dim=0).to(self._device)  # (ΣT,4)
            self.obj_pos_vel = torch.cat(self._obj_pos_vel_list, dim=0).to(self._device)
            self.obj_rot_vel = torch.cat(self._obj_rot_vel_list, dim=0).to(self._device)
            # ✅ ig / contact：注意维度
            self.ig = torch.cat(self.ig, dim=0).to(self._device)  # (ΣT, 52, 3)
            self.ig_sk = torch.cat(self.ig_sk, dim=0).to(self._device)  # (ΣT, 52, 3)
            self.contact_robot = torch.cat(self.contact_robot, dim=0).to(self._device)  # (ΣT, 52)

        # 唯一物体名 & 词表
        uniq_names = sorted(set(self._motion_obj_names))
        self.object_vocab = {name: i for i, name in enumerate(uniq_names)}  # name -> index
        self.object_vocab_inv = {i: name for name, i in self.object_vocab.items()}

        # 每个物体对应的 motion 下标列表（tensor on device）
        self._motions_by_object = {}
        for name, idx in self.object_vocab.items():
            motion_idx = [i for i, n in enumerate(self._motion_obj_names) if n == name]
            if len(motion_idx) == 0:
                continue
            self._motions_by_object[idx] = torch.tensor(motion_idx, dtype=torch.long, device=self._device)

        total_len = self.get_total_length()
        print(f"[MotionLibHoi] Loaded {len(self._motions)} motions (total {total_len:.3f}s). has_object={self._has_object}")

    def _sort_motions_by_length_hoi(self):
        idx = torch.argsort(torch.tensor(self._motion_lengths))
        self._motions = [self._motions[i] for i in idx]
        self._motion_files = [self._motion_files[i] for i in idx]
        self._motion_lengths = [self._motion_lengths[i] for i in idx]
        self._motion_fps = [self._motion_fps[i] for i in idx]
        self._motion_dt = [self._motion_dt[i] for i in idx]
        self._motion_num_frames = [self._motion_num_frames[i] for i in idx]
        self._motion_keys = [self._motion_keys[i] for i in idx]
        # 同步对象/HOI通道
        if len(self._obj_pos_list) > 0:
            self._obj_pos_list = [self._obj_pos_list[i] for i in idx]
            self._obj_rot_list = [self._obj_rot_list[i] for i in idx]
            self._obj_pos_vel_list = [self._obj_pos_vel_list[i] for i in idx]
            self._obj_rot_vel_list = [self._obj_rot_vel_list[i] for i in idx]
            self._motion_obj_names = [self._motion_obj_names[i] for i in idx]
        # ✅ 新增：同步 ig / contact_robot
        if len(self.ig) > 0:
            self.ig = [self.ig[i] for i in idx]  # list of (T,52,3)
        if len(self.ig_sk) > 0:
            self.ig_sk = [self.ig_sk[i] for i in idx]
        if len(self.contact_robot) > 0:
            self.contact_robot = [self.contact_robot[i] for i in idx]  # list of (T,52)

    def has_object(self):
        return getattr(self, "_has_object", False)

    # 覆盖：返回时带上 obj_*（若存在）
    def get_hoi_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor):
        """
        返回与 get_motion_state 同步时刻下的物体状态（插帧后）：
          - obj_pos      [B, 3]
          - obj_rot      [B, 4]  (xyzw)
          - obj_pos_vel  [B, 3]
          - obj_rot_vel  [B, 3]
        若库里没有对象数据，返回 None。
        """
        if not self.has_object():
            return None

        # 和父类 get_motion_state 完全一致的时间→帧索引计算
        motion_len = self._motion_lengths[motion_ids]  # [B]
        num_frames = self._motion_num_frames[motion_ids]  # [B]
        dt = self._motion_dt[motion_ids]  # [B]
        f0, f1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)  # [B],[B],[B]
        f0l = f0 + self.length_starts[motion_ids]  # 库级全局索引 [B]
        f1l = f1 + self.length_starts[motion_ids]

        # 线性插值位置
        o_pos0 = self.obj_pos[f0l]  # [B,3]
        o_pos1 = self.obj_pos[f1l]
        o_pos = (1.0 - blend.unsqueeze(-1)) * o_pos0 + blend.unsqueeze(-1) * o_pos1

        # 四元数 slerp（xyzw）
        o_rot0 = self.obj_rot[f0l]  # [B,4]
        o_rot1 = self.obj_rot[f1l]
        o_rot = torch_utils.slerp(o_rot0, o_rot1, blend.unsqueeze(-1))

        # 速度通常取 f0 帧（也可改成线性插值）
        o_pos_vel = self.obj_pos_vel[f0l]  # [B,3]
        o_rot_vel = self.obj_rot_vel[f0l]  # [B,3]

        ig = self.ig[f0l]  # [B,3]
        ig_sk = self.ig_sk[f0l]
        contact_robot = self.contact_robot[f0l]

        return {
            "obj_pos": o_pos,
            "obj_rot": o_rot,  # xyzw
            "obj_pos_vel": o_pos_vel,
            "obj_rot_vel": o_rot_vel,
            "ig": ig,
            "ig_sk": ig_sk,
            "contact_robot": contact_robot,
        }

    def sample_motions_by_object(self, object_ids: torch.Tensor) -> torch.Tensor:
        """
        输入：
          object_ids: [B]，每个 env 的 object 索引（与 vocab 对齐）
        返回：
          motion_ids: [B]，每个 env 对应的 motion 索引（库级）
        规则：
          在 _motions_by_object[obj] 这个子集中，按 self._sampling_prob 受限归一化后抽一个。
        """
        B = object_ids.shape[0]
        out = torch.empty(B, dtype=torch.long, device=self._device)

        # 若你维护了 self._sampling_prob（len=#motions）
        global_prob = getattr(self, "_sampling_prob", None)
        for b in range(B):
            obj_id = int(object_ids[b].item())
            subset = self._motions_by_object.get(obj_id, None)
            if subset is None or subset.numel() == 0:
                # 回退到全局采样
                if global_prob is None:
                    out[b] = torch.randint(low=0, high=len(self._motions), size=(1,), device=self._device)
                else:
                    out[b] = torch.multinomial(global_prob, num_samples=1, replacement=True)[0]
                continue

            if global_prob is None:
                pick = torch.randint(low=0, high=subset.numel(), size=(1,), device=self._device)
                out[b] = subset[pick]
            else:
                # 子集上的受限归一化概率
                prob_sub = global_prob[subset]
                prob_sub = prob_sub / prob_sub.sum().clamp_min(1e-8)
                idx_in_subset = torch.multinomial(prob_sub, num_samples=1, replacement=True)[0]
                out[b] = subset[idx_in_subset]
        return out

    def sample_time(self, motion_ids, truncate_time=None):
        """
        在“非接触帧集合”中等概率采样时间：
          - 不采样 any(contact_robot[t]) == True 的帧
          - 若开启 truncate_time，则不采样尾部时长对应的若干帧
          - 退化保护：若处理后集合为空，回退为全帧均匀随机
        """
        B = motion_ids.shape[0]
        out_t = torch.empty(B, dtype=torch.float32, device=self._device)

        for b in range(B):
            i = int(motion_ids[b].item())
            dt_i = float(self._motion_dt[i].item())
            T_i = int(self._motion_num_frames[i].item())
            # 该 motion 的“无接触帧”索引（局部帧号）
            valid = self._valid_frames_no_cg[i]

            # 处理 truncate_time：裁掉末尾若干帧
            if truncate_time is not None and truncate_time > 0.0:
                trunc_frames = int(max(round(truncate_time / dt_i), 0))
                max_idx = max(T_i - 1 - trunc_frames, 0)
                valid = valid[valid <= max_idx]
                if valid.numel() == 0:
                    # 若被裁空，回退到 [0 .. max_idx]
                    valid = torch.arange(max_idx + 1, device=self._device, dtype=torch.long)

            # 等概率在 valid 中抽一个帧，再转时间
            pick = valid[torch.randint(low=0, high=valid.numel(), size=(1,), device=self._device)]
            out_t[b] = pick.float() * dt_i

        return out_t

    def _precompute_non_contact_frames(self):
        """
        为每个 motion 预计算“非接触帧”（any(contact_robot[t]) == False）的索引列表：
          - self._valid_frames_no_cg: 长度 = #motions，元素是 1D LongTensor（该 motion 内部的帧索引）
          - 若库内没对象数据（_has_object=False）或某段全是接触帧，则回退到全帧可采样
        """
        self._valid_frames_no_cg = []
        for i in range(self._num_motions):
            T_i = int(self._motion_num_frames[i].item())
            if not getattr(self, "_has_object", False) or not hasattr(self, "contact_robot"):
                # 没有对象/接触轨迹：全帧可采样
                idxs = torch.arange(T_i, device=self._device, dtype=torch.long)
                self._valid_frames_no_cg.append(idxs)
                continue

            start = int(self.length_starts[i].item())
            end   = start + T_i
            cr = self.contact_robot[start:end]  # (T_i, 52)
            # “非接触帧”：所有 52 部位都为 0
            non_contact = (cr.abs().sum(dim=1) == 0)  # (T_i,)
            idxs = torch.nonzero(non_contact, as_tuple=False).squeeze(1)
            if idxs.numel() == 0:  #TODO： 这是潜在的 bug,这种片段应该直接删除并警告
                # 该段全是接触帧 → 回退：全帧可采样
                idxs = torch.arange(T_i, device=self._device, dtype=torch.long)
            self._valid_frames_no_cg.append(idxs)
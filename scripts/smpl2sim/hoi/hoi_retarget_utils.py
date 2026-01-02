import os
import sys
import os.path as osp

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
from tqdm import tqdm

import trimesh
import joblib
import torch
import mujoco
import json
import numpy as np
from scipy.spatial import cKDTree

Q_UPRIGHT_XYZW = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
Rupright = sRot.from_quat(Q_UPRIGHT_XYZW)

GRAVITY = torch.tensor([0.0, 0.0, -9.81])  # z 轴向上坐标系
AIRBORNE_H = 0.05        # 空中判定高度阈值（m）
ACC_DEV_THR = 2.0        # “偏离重力”判定阈值（m/s^2）
GROUND_H_TOL = 0.02      # 认为在地面的 z 高度容差（m）
SPEED_THR = 0.05         # 地面时“仍在运动”的速度阈值（m/s）
CLOSE_MIN_THR = 0.1     # 最近距离判定交互阈值（m）
ADAPTIVE_PAD = 0.005     # sigma_t = min_dist + 0.005（m）
FIXED_SIGMA_NO_INTERACT = 0.02  # 若不满足三条交互条件时的保底阈值（可选）

def _set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits); y_middle = np.mean(y_limits); z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

@torch.no_grad()
def compute_cg_ig_via_smplh_contacts_yup(
        smpl_model,  # smplx.create() 创建的模型
        human,  # human_yup 字典
        obj,  # obj_yup 字典
        obj_mesh_path,
        smpl_vert_part: np.ndarray,  # (V,) 顶点到部位/关节的映射索引
        contact_threshold: float = 0.01,
        unconcat_threshold: float = 0.1,
        samples_per_object: int = 1024,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    输入:
      human: 包含 'poses' (T,156), 'trans' (T,3), 'betas' (1,10/16) 的字典 (Y-up)
      obj:   包含 'angles' (T,3), 'trans' (T,3) 的字典 (Y-up)
    返回:
      cg: (T, body_cnt) float32   - 接触标签: 1.0接触, 0.0不接触, -1.0远离
      ig: (T, body_cnt, 3) float32 - 关节到物体最近点的向量 (世界系)
    """
    # 1. 解包与准备
    poses = np.asarray(human['poses'])
    trans = np.asarray(human['trans'])
    betas = human['betas']

    obj_angles = np.asarray(obj['angles'])
    obj_trans = np.asarray(obj['trans'])

    T = poses.shape[0]
    smpl_vert_part = np.asarray(smpl_vert_part, dtype=np.int64)
    body_cnt = int(smpl_vert_part.max()) + 1

    smpl_model = smpl_model.to(device)

    # 2. 物体点云采样
    mesh_obj = trimesh.load(obj_mesh_path, force='mesh')
    obj_pts_local = trimesh.sample.sample_surface_even(
        mesh_obj, count=samples_per_object, seed=2025
    )[0].astype(np.float32)

    cg = np.zeros((T, body_cnt), dtype=np.float32)
    ig = np.zeros((T, body_cnt, 3), dtype=np.float32)

    for t in tqdm(range(T), desc="Computing CG/IG", mininterval=5.0):
        # A. SMPL 前向计算 (适配 smplx 库接口)
        p_t = torch.from_numpy(poses[t:t + 1]).float().to(device)
        tr_t = torch.from_numpy(trans[t:t + 1]).float().to(device)
        betas_t =  torch.from_numpy(betas[t:t + 1]).float().to(device)

        # 适配 SMPL-H / SMPL-X 常见的 156 维输入分段
        # 0:3 global_orient, 3:66 body_pose, 66:111 l_hand, 111:156 r_hand
        out = smpl_model(
            global_orient=p_t[:, :3],
            body_pose=p_t[:, 3:66],
            left_hand_pose=p_t[:, 66:111],
            right_hand_pose=p_t[:, 111:156],
            betas=betas_t,
            transl=tr_t,
        )
        verts = out.vertices[0].cpu().numpy()  # (V, 3)
        joints = out.joints[0].cpu().numpy()[:body_cnt]  # (J, 3) 截取有效的身体关节

        # B. 物体点云变换
        rot_mat = sRot.from_rotvec(obj_angles[t]).as_matrix()
        obj_pts_w = (obj_pts_local @ rot_mat.T) + obj_trans[t]

        # C. 距离查询 (SDF/Interaction)
        tree = cKDTree(obj_pts_w)

        # 顶点到物体距离 (用于接触标签 CG)
        dist_v, _ = tree.query(verts, k=1, workers=-1)

        # 计算接触 mask
        contact_mask = dist_v < contact_threshold
        if np.any(contact_mask):
            hits = np.bincount(smpl_vert_part[contact_mask], minlength=body_cnt)
            cg[t] = (hits > 0).astype(np.float32)

        # 计算远离 mask (用于惩罚)
        near_mask = dist_v <= unconcat_threshold
        hits_near = np.bincount(smpl_vert_part[near_mask], minlength=body_cnt)
        cg[t][hits_near == 0] = -1.0  # 标记为完全不接触且不靠近

        # D. 关节到物体向量 (用于 IG)
        _, qidx = tree.query(joints, k=1, workers=-1)
        nearest_pts = obj_pts_w[qidx]
        ig[t] = (nearest_pts - joints).astype(np.float32)

    return cg, ig




def _quat_rotate_xyzw(q, v):
    """快速批量旋转: q:[T,4] (xyzw)，v:[P,3] 或 [T,P,3]"""
    if v.dim() == 2:
        v = v.unsqueeze(0).expand(q.shape[0], -1, -1)   # -> [T,P,3]
    qv = q[:, :3]                                       # [T,3]
    qw = q[:, 3:4]                                      # [T,1]
    # 公式: v' = v + 2*qv×(qw*v + qv×v)
    cross_qv_v = torch.cross(qv.unsqueeze(1).expand_as(v), v, dim=-1)
    term = qw.unsqueeze(1)*v + cross_qv_v
    return v + 2.0*torch.cross(qv.unsqueeze(1).expand_as(v), term, dim=-1)

def _to_float32_safe(arr, allow_strings_prefix_t=False):
    """Cast to float32, optionally parse string like 't0003.000'."""
    if arr is None:
        return None
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float32)
    # bytes/str/object → str
    s = arr.astype(str)
    s = np.char.strip(s)
    if allow_strings_prefix_t:
        # 去掉开头的 't' 或 'T'，以及多余空格
        s = np.char.lstrip(s, chars='tT')
    # 把逗号小数换成点（以防万一）
    s = np.char.replace(s, ',', '.')
    return s.astype(np.float32)

def load_corrected_behave_sequence(sequence_path, parse_frame_times=True, strict=True):
    out = {'smpl': {}, 'object': {}, 'info': {}}

    # --- SMPL fits ---
    smpl_npz = osp.join(sequence_path, "human.npz")
    with np.load(smpl_npz, allow_pickle=True) as f:
        out['smpl']['trans'] = f['transl']
        out['smpl']['poses'] = np.concatenate([f['glo_rot'], f['body_pose'], f['hand_pose']], axis=1)
        out['smpl']['betas'] = f['betas']
        out['smpl']['gender'] = f['gender']
        # for k in ['poses', 'trans', 'betas']:
        #     if k not in f:
        #         if strict:
        #             raise KeyError(f"`{k}` not found in {smpl_npz}")
        #         else:
        #             out['smpl'][k] = None
        #             continue
        #     out['smpl'][k] = _to_float32_safe(f[k])

    # --- Object fits ---
    obj_npz = osp.join(sequence_path, "object.npz")
    if osp.exists(obj_npz):
        with np.load(obj_npz, allow_pickle=True) as f:
            out['object']['angles'] = _to_float32_safe(f['obj_angle']) if 'obj_angle' in f else None
            out['object']['trans']  = _to_float32_safe(f['obj_trans'])  if 'obj_trans'  in f else None
            if 'frame_times' in f and parse_frame_times:
                try:
                    # 关键：允许像 't0003.000' 这样的字符串
                    out['object']['frame_times'] = _to_float32_safe(f['frame_times'], allow_strings_prefix_t=True)
                except Exception:
                    # 解析失败就置 None（后续按固定 fps=30 处理）
                    out['object']['frame_times'] = None
            else:
                out['object']['frame_times'] = None
    else:
        out['object'] = {'angles': None, 'trans': None, 'frame_times': None}

    # --- Info JSON ---
    info_json = osp.join(sequence_path, "info.json")
    if osp.exists(info_json):
        with open(info_json, 'r', encoding='utf-8') as f:
            info = json.load(f)
        out['info']['gender'] = info.get('gender', 'neutral')
        out['info']['cat']    = info.get('cat', '')
    else:
        out['info'] = {'gender': 'neutral', 'cat': ''}

    # --- 基本形状校验（保守一些） ---
    poses, trans, betas = out['smpl']['poses'], out['smpl']['trans'], out['smpl']['betas']
    if poses is None or trans is None:
        raise ValueError(f"Missing poses/trans in {smpl_npz}")
    if poses.ndim != 2 or poses.shape[1] != 156:
        raise ValueError(f"`poses` expected (T,156), got {poses.shape}")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"`trans` expected (T,3), got {trans.shape}")
    T = poses.shape[0]
    if trans.shape[0] != T:
        raise ValueError(f"Length mismatch: poses(T={T}) vs trans(T={trans.shape[0]})")
    if betas is not None and not (
        betas.ndim == 1 or (betas.ndim == 2 and betas.shape[0] in (1, T))
    ):
        raise ValueError(f"`betas` shape unexpected: {betas.shape}")

    return out


def compute_sdf(points1, points2):
    # type: (Tensor, Tensor) -> Tensor
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat


def angular_velocity_world_from_quat_xyzw(q_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """
    输入:  q_xyzw 形状 (T,4)，SciPy/内存统一为 [x,y,z,w]
    输出:  omega_w 形状 (T,3)，世界系角速度；omega_w[0]=0
    做法:  dq = r[t-1]^{-1} * r[t] -> log(dq)/dt 在 t-1 局部系，
          再用 R_{t-1} 把它映到世界系。
    """
    T = len(q_xyzw)
    omega_w = np.zeros((T, 3), dtype=np.float32)
    if T <= 1 or dt <= 0:
        return omega_w

    r = sRot.from_quat(q_xyzw)  # [x,y,z,w]
    for t in range(1, T):
        dq = r[t - 1].inv() * r[t]  # 相对旋转（定义在 t-1 局部系）
        w_local = dq.as_rotvec() / dt  # 局部角速度
        R_prev = r[t - 1].as_matrix()
        omega_w[t] = (R_prev @ w_local).astype(np.float32)
    return omega_w


def get_smpl_vert_part(smpl_model) -> np.ndarray:
    """
    适配 smplx 官方库创建的模型。
    返回: smpl_vert_part 形状 (V,), 值域 [0..J-1]
    通过对 lbs_weights 取 argmax 得到每个顶点所属权重最大的关节索引。
    """
    weights = smpl_model.lbs_weights
    part = torch.argmax(weights, dim=1).cpu().numpy().astype(np.int64)

    return part

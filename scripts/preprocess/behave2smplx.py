import os
import sys
import os.path as osp

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
from tqdm import tqdm
import argparse
import glob

from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPLH_MUJOCO_NAMES, SMPLH_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_parser import SMPLX_Parser
from scripts.preprocess.mujoco_contact_inference import build_local_templates_by_body, contacts_from_xml_pointcloud
from scripts.preprocess.mujoco_contact_inference import penetration_depth_sequence_ig
from scripts.preprocess.mujoco_contact_inference import build_qpos_seq_from_state, quick_viz_frame, build_sk2mj_index
from scripts.render.mujoco_render import vis_mujoco_hoi, create_temp_xml_with_object
import smplx
import trimesh
import joblib
import torch
import mujoco
import json
import numpy as np
from scipy.spatial import cKDTree
from scripts.libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer

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
def compute_cg_ig_via_smplh_contacts(
    smplh_layer,                 # torch nn.Module，前向返回 (verts, joints)
    pose_aa: np.ndarray,         # (T, D)
    betas: torch.Tensor,         # (10,) 或 (T,10)
    trans: np.ndarray,           # (T, 3)
    obj_mesh_path: str,
    obj_pos_world: np.ndarray,   # (T, 3)
    obj_quat_xyzw: np.ndarray,   # (T, 4) xyzw
    smplh_vert_part: np.ndarray, # (V,) in [0, body_cnt-1]
    contact_threshold: float = 0.01,
    samples_per_object: int = 1024,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    # -------- 新增：可视化参数 --------
    viz_t: int = 1,           # 指定要可视化的帧索引；None 则不画
    viz_max_arrows: int = 120,   # 控制箭头数量，避免太密
    viz_show: bool = False,       # 是否 plt.show()
):
    """
    返回:
      cg: (T, body_cnt) float32   # 每帧部位接触(0/1)
      ig: (T, body_cnt, 3) float32 # 部位关节 -> 物体最近点 的向量(世界系)
    """
    T = pose_aa.shape[0]
    smplh_vert_part = np.asarray(smplh_vert_part, dtype=np.int64)
    body_cnt = int(smplh_vert_part.max()) + 1

    smplh_layer = smplh_layer.to(device)

    mesh_obj = trimesh.load(obj_mesh_path, force='mesh')
    obj_pts_local = trimesh.sample.sample_surface_even(mesh_obj, count=samples_per_object, seed=2025)[0].astype(np.float32)

    betas = betas[None, :] if betas.ndim == 1 else betas   # (1,10) 或 (T,10)
    betas_th = betas.to(device)

    cg = np.zeros((T, body_cnt), dtype=np.float32)
    ig = np.zeros((T, body_cnt, 3), dtype=np.float32)

    # 为可视化的那一帧缓存物体点与最近点（只在命中 viz_t 时赋值）
    _viz_cache = {}

    for t in range(T):
        pose_th  = torch.from_numpy(pose_aa[t:t+1].astype(np.float32)).to(device)
        trans_th = torch.from_numpy(trans[t:t+1].astype(np.float32)).to(device)

        out = smplh_layer(pose_th, th_betas=betas_th, th_trans=trans_th)
        verts  = out[0][0].detach().cpu().numpy().astype(np.float32)   # (V,3)
        joints = out[1][0].detach().cpu().numpy().astype(np.float32)   # (J,3)

        rot = sRot.from_quat(obj_quat_xyzw[t])                         # xyzw
        obj_pts_w = rot.apply(obj_pts_local) + obj_pos_world[t]        # (P,3)

        tree = cKDTree(obj_pts_w)
        dist, _ = tree.query(verts, k=1, workers=-1)
        contact_mask = dist < contact_threshold

        if np.any(contact_mask):
            hits = np.bincount(smplh_vert_part[contact_mask], minlength=body_cnt)
            cg[t] = (hits > 0).astype(np.float32)

        # 关节到最近点向量
        _, qidx = tree.query(joints, k=1, workers=-1)
        nearest = obj_pts_w[qidx]                                      # (J,3)
        ig_vecs = (nearest - joints).astype(np.float32)                # (J,3)
        ig[t] = ig_vecs

        # 命中可视化帧：把需要的东西先存起来
        if viz_t is not None and t == viz_t:
            _viz_cache["joints"] = joints
            _viz_cache["obj_pts_w"] = obj_pts_w
            _viz_cache["nearest"] = nearest
            _viz_cache["ig_vecs"] = ig_vecs
            _viz_cache["contact_mask"] = contact_mask
            _viz_cache["smplh_vert_part"] = smplh_vert_part
            _viz_cache["cg_row"] = cg[t].copy()

    # ---- 可视化：只画一帧（viz_t）----
    if viz_t is not None and "joints" in _viz_cache:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        joints  = _viz_cache["joints"]          # (J,3)
        obj_w   = _viz_cache["obj_pts_w"]       # (P,3)
        nearest = _viz_cache["nearest"]         # (J,3)
        ig_vecs = _viz_cache["ig_vecs"]         # (J,3)
        cg_row  = _viz_cache["cg_row"]          # (body_cnt,)

        # 选一部分箭头避免太密
        J = joints.shape[0]
        step = max(1, int(np.ceil(J / max(1, viz_max_arrows))))
        sel = np.arange(0, J, step, dtype=int)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 物体点云（灰）/ 关节（蓝）
        ax.scatter(obj_w[:, 0], obj_w[:, 1], obj_w[:, 2], s=1, alpha=0.35)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=18)

        # 箭头（从关节指向最近点）
        U = nearest[sel, 0] - joints[sel, 0]
        V = nearest[sel, 1] - joints[sel, 1]
        W = nearest[sel, 2] - joints[sel, 2]
        ax.quiver(joints[sel, 0], joints[sel, 1], joints[sel, 2], U, V, W, length=1.0, normalize=False)

        # 画一部分最近点（红）
        ax.scatter(nearest[sel, 0], nearest[sel, 1], nearest[sel, 2], s=10)

        # 轴设置
        ax.set_title(f"cg/ig preview @ t={viz_t}")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        _set_axes_equal(ax)

        if viz_show:
            plt.tight_layout()
            plt.show()

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

def apply_cam2world_rotvec_trans(rotvec, trans, R3x3):
    r_new = sRot.from_matrix(R3x3) * sRot.from_rotvec(rotvec)
    t_new = (R3x3 @ trans.T).T
    return r_new.as_rotvec().astype(np.float32), t_new.astype(np.float32)

def apply_upright_quat_xyzw(q_xyzw):
    # 人体做的是: global = global * Rupright.inv()  （SciPy右乘）
    return (sRot.from_quat(q_xyzw) * Rupright.inv()).as_quat().astype(np.float32)

def fix_trans_height(pose_aa, trans, betas, mesh_parser):
    with torch.no_grad():
        frame_check = pose_aa.shape[0]
        betas = betas
        mesh_parser = mesh_parser
        height_tolorance = 0.0
        vertices_curr, joints_curr = mesh_parser.get_joints_verts(pose_aa[:frame_check], betas[None,],
                                                                  trans[:frame_check])

        offset = joints_curr[:, 0] - trans[
                                     :frame_check]  # account for SMPL root offset. since the root trans we pass in has been processed, we have to "add it back".

        diff_fix = ((vertices_curr - offset[:, None])[:frame_check, ..., -1].min(
            dim=-1).values - height_tolorance).min()  # Only acount the first 30 frames, which usually is a calibration phase.

        trans[..., -1] -= diff_fix
        return trans, diff_fix

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

def load_behave_sequence(sequence_path, parse_frame_times=True, strict=True):
    out = {'smpl': {}, 'object': {}, 'info': {}}

    # --- SMPL fits ---
    smpl_npz = osp.join(sequence_path, "smpl_fit_all.npz")
    with np.load(smpl_npz, allow_pickle=True) as f:
        for k in ['poses', 'trans', 'betas']:
            if k not in f:
                if strict:
                    raise KeyError(f"`{k}` not found in {smpl_npz}")
                else:
                    out['smpl'][k] = None
                    continue
            out['smpl'][k] = _to_float32_safe(f[k])

    # --- Object fits ---
    obj_npz = osp.join(sequence_path, "object_fit_all.npz")
    if osp.exists(obj_npz):
        with np.load(obj_npz, allow_pickle=True) as f:
            out['object']['angles'] = _to_float32_safe(f['angles']) if 'angles' in f else None
            out['object']['trans']  = _to_float32_safe(f['trans'])  if 'trans'  in f else None
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

def smplh_vert_part_from_custom_layer(smpl_layer) -> np.ndarray:
    """
    返回: smplh_vert_part 形状 (V,), 值域 [0..J-1]
    读取 smpl_layer.smpl_data['weights'] (可能是 chumpy/sparse/numpy/torch)，
    转成 torch.FloatTensor 后做 argmax。
    """
    W = smpl_layer.smpl_data.get("weights", None)
    if W is None:
        raise ValueError("smpl_layer.smpl_data['weights'] not found.")

    # ---- 统一转 numpy ----
    try:
        import chumpy as ch
        if isinstance(W, ch.Ch):
            W = W.r  # chumpy -> numpy
    except Exception:
        pass

    # scipy.sparse 支持
    try:
        from scipy.sparse import issparse
        if issparse(W):
            W = W.toarray()
    except Exception:
        pass

    if isinstance(W, torch.Tensor):
        W_t = W
    else:
        W_t = torch.from_numpy(np.asarray(W))  # numpy/list/other -> torch

    if W_t.dtype != torch.float32:
        W_t = W_t.to(torch.float32)

    part = torch.argmax(W_t, dim=1).cpu().numpy().astype(np.int64)  # (V,)
    return part

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--src", type=str, default="", help="Path to BEHAVE dataset")
    parser.add_argument("--process_split", type=str, default="train", choices=["train", "test", "valid"])
    parser.add_argument("--render", action="store_true", default=False, help="Whether to render the \
                                                                        retargeted motion using scenepic animation.")
    parser.add_argument("--dst", type=str, default="dataset/smplx_motion/behave_small", help="Output path")
    args = parser.parse_args()
    output_dir = args.dst

    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": True,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smplx",
    }
    skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/smplx_humanoid.xml")
    smpl_local_robot = LocalRobot(robot_cfg, data_dir="data/SMPL/smplx")


    # BEHAVE dataset structure: each sequence contains SMPL fits and object interactions
    # We look for sequences with SMPL fits
    all_sequences = glob.glob(f"{args.src}/**/", recursive=True)
    behave_full_motion_dict = {}

    smplx_parser_n = SMPLX_Parser(
        model_path='data/SMPL/smplx',
        gender='neutral',
        use_pca=False,  # 关键：不用 PCA，接受 45D/手
        create_transl=False,
        flat_hand_mean=True,
        num_betas=20  # SMPL-X 20 维 beta
    )

    skipped_due_to_first_frame_collision = []

    for sequence_dir in tqdm(all_sequences):
        if not osp.exists(osp.join(sequence_dir, "smpl_fit_all.npz")):
            continue

        print("Processing", sequence_dir)

        # Load BEHAVE sequence data
        sequence_data = load_behave_sequence(sequence_dir)
        smpl_data = sequence_data['smpl']
        pose_aa_smpl = smpl_data['poses']
        trans = smpl_data['trans']
        betas = smpl_data.get('betas', np.zeros(10))
        gender = sequence_data['info']['gender']

        N = pose_aa_smpl.shape[0]
        if N < 10:
            print(f"Sequence too short ({N} frames), skipping")
            continue

        # 模型
        D = pose_aa_smpl.shape[1]

        # 基于 SMPL 的修正,
        model = smplx.create(
            model_path='data/SMPL',  # 包含 SMPL/SMPLX 模型文件的目录
            model_type='smpl',  # 或 'smplx' / 'smplh'，取决于你的模型
            gender='neutral',  # male/female/neutral
            use_pca=False
        )

        # 2. 创建全 0 参数
        betas = torch.zeros([1, 10])  # 形状参数（shape）
        body_pose = torch.zeros([1, 69])  # 姿态参数（23*3）
        global_orient = torch.zeros([1, 3])  # 全局旋转
        transl = torch.zeros([1, 3])  # 平移

        output = model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl
        )

        # 4. 导出关节或顶点位置
        joints = output.joints.detach().cpu().numpy()  # (1, N_joints, 3)
        vertices = output.vertices.detach().cpu().numpy()  # (1, 6890, 3)

        skeleton_tree_smpl = SkeletonTree.from_mjcf(f"data/robots/smpl/smpl_humanoid.xml")
        # 世界坐标系旋转
        R_cam2world = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
        # R_cam2world = [[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]
        # R_cam2world = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        world_shift = np.zeros(3, dtype=np.float32)
        root_trans = trans + skeleton_tree_smpl.local_translation[0].numpy()
        pose_aa = pose_aa_smpl.copy()
        pose_aa[:, :3], root_trans_offset = apply_cam2world_rotvec_trans(pose_aa_smpl[:, :3], root_trans, R_cam2world)
        root_trans_offset = torch.from_numpy(root_trans_offset).float()
        # 关节重排
        pose_aa_mj = pose_aa.reshape(N, 52, 3)
        smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa_mj[:, smpl_2_mujoco]

        # 轴角 -> 四元数（注意 scipy 返回 [x,y,z,w]）
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 52, 4)
        # root_trans_offset = torch.zeros(N, 3) + root_trans
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True)

        frame_check = 100
        height_tolorance = 0
        fix_height = True  # 开启更稳妥
        if fix_height:
            with torch.no_grad():
                frame_check = min(frame_check, N)
                pose_t = torch.from_numpy(pose_aa[:frame_check]).float()  # (F,72) 关键！
                beta_np = np.zeros(20, dtype=np.float32)  # 10 维 betas
                beta_t = torch.from_numpy(beta_np[None, ...]).float()  # (1,10)
                trans_t = root_trans_offset[:frame_check].float()  # (F,3)

                verts, joints = smplx_parser_n.get_joints_verts(pose_t, beta_t, trans_t)
                offset = joints[:, 0] - trans_t
                feet_z = (verts - offset[:, None])[..., -1]
                diff_fix = feet_z.min().item()
                root_trans_offset[..., -1] -= diff_fix
                world_shift -= np.array([0, 0, diff_fix], dtype=np.float32)

        # 局部坐标系旋转
        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) *
                                sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                                            torch.from_numpy(pose_quat_global),
                                                                            root_trans_offset, is_local=False)
        fps = 30
        motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30)

        # 物体修正(Behave 世界坐标系 -> Isaac Gym 世界坐标系)
        obj_angles = sequence_data.get('object', {}).get('angles', None)
        obj_trans = sequence_data.get('object', {}).get('trans', None)
        obj_times = sequence_data.get('object', {}).get('frame_times', None)
        key_str = os.path.basename(os.path.normpath(sequence_dir))
        object_name_str = key_str.split('_')[2]

        model_root = 'data/smplh'
        smplh_layer = SMPL_Layer(center_idx=0, gender=gender, num_betas=10,
                               model_root=str(model_root), hands=True).to('cuda')

        obj_root = "dataset/behave/objects_centered"  # 根据你的资源路径调整
        obj_name = object_name_str
        obj_mesh_path = osp.join(obj_root, "objects", obj_name, f"{obj_name}.obj")
        mesh_obj = trimesh.load(obj_mesh_path, force='mesh')
        obj_points, _ = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2025)

        obj_angles_w, obj_trans_w = apply_cam2world_rotvec_trans(obj_angles, obj_trans, R_cam2world)
        obj_quat_xyzw = sRot.from_rotvec(obj_angles_w).as_quat().astype(np.float32)


        # 位置
        obj_pos = obj_trans_w.astype(np.float32)
        obj_pos = (obj_pos + world_shift[None, :]).astype(np.float32)
        # 速度
        dt = 1.0 / fps
        obj_pos_vel = np.zeros_like(obj_pos)
        obj_pos_vel[1:] = (obj_pos[1:] - obj_pos[:-1]) / dt

        # 角速度
        obj_rot_vel = angular_velocity_world_from_quat_xyzw(obj_quat_xyzw, dt)

        key_str = os.path.basename(os.path.normpath(sequence_dir))
        object_name_str = key_str.split('_')[2]
        motion_dict = motion.to_dict()  # 与 motion.to_file() 里保存的结构一致

        # ==== 推理接触 ====
        # 1) 准备 body_pos (世界坐标): [T,52,3]
        body_pos_t = new_sk_state.global_translation  # Tensor, 与上文一致
        if not isinstance(body_pos_t, torch.Tensor):
            body_pos_t = torch.from_numpy(body_pos_t)
        body_pos_t = body_pos_t.to(torch.float32)  # [T,52,3]

        q_xyzw = torch.from_numpy(obj_quat_xyzw.astype(np.float32))  # [T,4]
        p_w = torch.from_numpy(obj_pos.astype(np.float32))  # [T,3]
        dt = 1.0 / fps

        obj_points_local = torch.from_numpy(
            trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2025)[0].astype(np.float32))
        obj_pts_world = _quat_rotate_xyzw(q_xyzw, obj_points_local) + p_w.unsqueeze(1)  # [T,P,3]

        with torch.no_grad():
            # obj_pts_world 已在上文计算
            ig_torch = compute_sdf(body_pos_t, obj_pts_world)  # [T,52,3]
            ref_ig = ig_torch.cpu().numpy().astype(np.float32)

        body_clouds, body_geoms, mj_model = build_local_templates_by_body("data/robots/smpl/smplx_humanoid.xml",
                                                                samples_per_geom=500)

        qpos_seq_np = build_qpos_seq_from_state(mj_model, new_sk_state)

        obj_pts_world_np = obj_pts_world.cpu().numpy().astype(np.float32)  # [T,P,3]
        obj_pos_np = p_w.cpu().numpy().astype(np.float32)  # [T,3]
        obj_vel_np = obj_pos_vel.astype(np.float32)  # [T,3]
        obj_acc_np = np.zeros_like(obj_vel_np, dtype=np.float32)
        obj_acc_np[1:] = (obj_vel_np[1:] - obj_vel_np[:-1]) / dt

        sk2mj, mj2sk = build_sk2mj_index(mj_model, skeleton_tree, drop_world=True)
        names = np.array([str(n) for n in skeleton_tree.node_names])  # or skeleton_tree.body_names
        # 你想标记为“地面/脚/胸”的集合（名字要与 XML/Skeleton 一致）
        mask_names = {"L_Ankle", "R_Ankle", "L_Toe", "R_Toe"}
        # Skeleton 顺序的布尔掩码：True 表示该 body 属于“接地”集合
        ground_mask_sk = np.isin(names, list(mask_names))
        # 调 contacts + ig
        # contact_robot, ig_np = contacts_from_xml_pointcloud(
        #     mj_model, body_clouds,
        #     qpos_seq=qpos_seq_np.astype(np.float32),
        #     obj_pts_world=obj_pts_world_np,
        #     obj_pos=obj_pos_np, vel=obj_vel_np, acc=obj_acc_np,
        #     sigma_pad=ADAPTIVE_PAD, sigma_no_interact=FIXED_SIGMA_NO_INTERACT,
        #     ground_height=0.0,
        #     sk2mj=sk2mj, mj2sk=mj2sk,
        #     ig_body_pos_world = body_pos_t.numpy().astype(np.float32),
        #     ig_body_rot_world = new_sk_state.global_rotation.numpy().astype(np.float32),
        #     ground_mask_sk=ground_mask_sk,
        # )

        pen_seq = penetration_depth_sequence_ig(
            mj_model,
            body_geoms,
            mj2sk,
            obj_pts_world_np,  # e.g. List[np.ndarray], len T, each (P_t,3)
            body_pos_t.numpy().astype(np.float32),  # (T, B, 3)
            new_sk_state.global_rotation.numpy().astype(np.float32),  # (T, B, 4) xyzw
        )

        first_frame_collided = (pen_seq[0] > 0).any()
        if first_frame_collided:
            key_str = os.path.basename(os.path.normpath(sequence_dir))  # 样本名
            print(f"[SKIP-FIRST-FRAME-COLLISION] {key_str}")
            skipped_due_to_first_frame_collision.append(key_str)
            continue  # 直接跳到下一个 sequence


        cg_np, ig_np = compute_cg_ig_via_smplh_contacts(
            smplh_layer=smplh_layer,
            pose_aa=pose_aa_smpl,  # (T,D)
            betas=betas,  # (10,) 或 (1,10)
            trans=trans,  # (T,3)
            obj_mesh_path=obj_mesh_path,  # 物体mesh（局部坐标）
            obj_pos_world=obj_trans,  # (T,3)
            obj_quat_xyzw=sRot.from_rotvec(obj_angles).as_quat(),  # (T,4) xyzw
            smplh_vert_part=smplh_vert_part_from_custom_layer(smplh_layer),
            contact_threshold=0.01,
            samples_per_object=1024,
        )

        ig_mj = ig_np[:, smpl_2_mujoco, :]  # (T,52,3)
        ig_mj = ig_mj @ np.array(R_cam2world).T
        cg_mj = cg_np[:, smpl_2_mujoco]  # (T,52)


        # body_ids_wo_foot_ankel = np.where(~ground_mask_sk)[0]

        bundle = {
            "motion": motion_dict,  # SkeletonMotion 的 dict（含关节、根姿态、fps 等）
            "object": {
                "name": object_name_str,
                "obj_pos": obj_pos,
                "obj_rot": obj_quat_xyzw,  # xyzw —— Isaac Gym 对齐
                "obj_pos_vel": obj_pos_vel,
                "obj_rot_vel": obj_rot_vel,
            },
            "interaction": {
                "ig": ig_mj,  # (T,52,3) float32（世界系）
                "contact_robot": cg_mj,  # (T,52)   0/1 float32
                "collision_tag": (pen_seq > 0).any(axis=1)
            }
        }

        # === 可视化
        t = min(100, obj_pts_world_np.shape[0] - 1)
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = qpos_seq_np[t]
        mujoco.mj_forward(mj_model, mj_data)

        body_pos_frame = body_pos_t[t].cpu().numpy().astype(np.float64)
        body_rot_frame = new_sk_state.global_rotation[t].cpu().numpy().astype(np.float64)
        # body_pos_frame = mj_data.xpos[sk2mj].copy()  # 严格用 MJCF body 原点
        quick_viz = False

        if quick_viz:
            quick_viz_frame(
                mj_model, mj_data,
                body_local_clouds=body_clouds,
                obj_pts=obj_pts_world_np[t],
                contact_row=cg_mj[t],
                body_rot_frame=body_rot_frame,
                mj2sk=mj2sk,
                title=f"seq:{key_str} t={t}",
                body_pos_frame=body_pos_frame,
                ig_frame=ig_mj[t],
            )

        if args.render:
            temp_xml = create_temp_xml_with_object(
                "data/robots/smpl/smplx_humanoid.xml",
                obj_mesh_path
            )
            motion_traj = {}
            motion_traj['root_trans_offset'] = new_sk_state.root_translation.numpy()
            motion_traj['root_rotation'] = new_sk_state.global_root_rotation.numpy()
            motion_traj['dof'] = sRot.from_quat(new_sk_state.local_rotation[:, 1:].reshape(-1, 4)).as_rotvec().reshape(N, -1, 3)
            vis_mujoco_hoi(
                motion_traj,
                obj_pos=obj_pos,
                obj_quat_xyzw=obj_quat_xyzw,
                xml_path=temp_xml
            )

        # Construct save path
        rel_path = osp.relpath(sequence_dir, args.src)
        save_path = osp.join(output_dir, rel_path, f"{key_str}.npy")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        np.save(save_path, bundle, allow_pickle=True)


    print("Done")


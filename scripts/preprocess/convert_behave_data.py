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
from scripts.preprocess.mujoco_contact_inference import build_qpos_seq_from_state, quick_viz_frame, build_sk2mj_index
from scripts.render.mujoco_render import vis_mujoco_hoi, create_temp_xml_with_object
import trimesh
import joblib
import torch
import mujoco
import json
import numpy as np


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--path", type=str, default="", help="Path to BEHAVE dataset")
    parser.add_argument("--process_split", type=str, default="train", choices=["train", "test", "valid"])
    parser.add_argument("--render", action="store_true", default=False, help="Whether to render the \
                                                                        retargeted motion using scenepic animation.")
    args = parser.parse_args()
    output_dir = "dataset/smplx_motion/behave_small"

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
    skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/smplx_humanoid_v2.xml")
    smpl_local_robot = LocalRobot(robot_cfg, data_dir="data/SMPL/smplx")


    # BEHAVE dataset structure: each sequence contains SMPL fits and object interactions
    # We look for sequences with SMPL fits
    all_sequences = glob.glob(f"{args.path}/**/", recursive=True)
    behave_full_motion_dict = {}

    smplx_parser_n = SMPLX_Parser(
        model_path='data/SMPL/smplx',
        gender='neutral',
        use_pca=False,  # 关键：不用 PCA，接受 45D/手
        create_transl=False,
        flat_hand_mean=True,
        num_betas=20  # SMPL-X 20 维 beta
    )

    for sequence_dir in tqdm(all_sequences):
        if not osp.exists(osp.join(sequence_dir, "smpl_fit_all.npz")):
            continue

        print("Processing", sequence_dir)

        # Load BEHAVE sequence data
        try:
            sequence_data = load_behave_sequence(sequence_dir)
            if 'smpl' not in sequence_data:
                print(f"No SMPL data found in {sequence_dir}")
                continue

            smpl_data = sequence_data['smpl']

            # Extract SMPL parameters - BEHAVE provides fitted parameters
            if 'poses' in smpl_data and 'trans' in smpl_data:
                pose_aa = smpl_data['poses']  # Should be in axis-angle format
                root_trans = smpl_data['trans']
                betas = smpl_data.get('betas', np.zeros(10))

                # Ensure pose_aa has correct shape (N, 72) for SMPL
                if pose_aa.shape[1] == 69:  # Body poses only (23 joints * 3)
                    # Add global rotation (root) as zeros - will be handled by root_trans
                    pose_aa = np.concatenate([np.zeros((pose_aa.shape[0], 3)), pose_aa], axis=-1)

                if pose_aa.shape[1] < 72:
                    # Pad with zeros for hand poses if missing
                    padding = np.zeros((pose_aa.shape[0], 72 - pose_aa.shape[1]))
                    pose_aa = np.concatenate([pose_aa, padding], axis=-1)

            else:
                print(f"Missing pose or trans data in {sequence_dir}")
                continue

        except Exception as e:
            print(f"Error loading sequence {sequence_dir}: {e}")
            continue

        N = pose_aa.shape[0]
        if N < 10:
            print(f"Sequence too short ({N} frames), skipping")
            continue

        # 模型
        D = pose_aa.shape[1]
        # TODO: 这里对于 非 156 的缺少 padding
        if D == 156:
            # SMPL-X 全身
            pose_aa = pose_aa
        elif D == 72:
            # 已是 SMPL body
            pose_aa = pose_aa
        elif D == 69:
            # 缺少 global；补 3 维零作为 global
            pose_aa_body72 = np.concatenate([np.zeros((pose_aa.shape[0], 3), dtype=pose_aa.dtype),
                                             pose_aa], axis=-1)
        else:
            raise ValueError(f"Unexpected SMPL pose dim {D}. Expect 69/72/156.")

        skeleton_tree_smpl = SkeletonTree.from_mjcf(f"data/robots/smpl/smpl_humanoid_v1.xml")
        # 世界坐标系旋转
        R_cam2world = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
        # R_cam2world = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        world_shift = np.zeros(3, dtype=np.float32)
        root_trans_offset = root_trans + skeleton_tree_smpl.local_translation[0].numpy()
        # root_trans_offset = root_tran
        pose_aa[:, :3], root_trans_offset = apply_cam2world_rotvec_trans(pose_aa[:, :3], root_trans_offset, R_cam2world)
        root_trans_offset = torch.from_numpy(root_trans_offset).float()
        # 关节重排
        pose_aa_mj = pose_aa.reshape(N, 52, 3)
        smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa_mj[:, smpl_2_mujoco]

        # 轴角 -> 四元数（注意 scipy 返回 [x,y,z,w]）
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 52, 4)

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

        # 2) 准备物体表面点（以物体中心为原点）
        obj_root = "dataset/behave/objects_centered"  # 根据你的资源路径调整
        obj_name = object_name_str
        obj_mesh_path = osp.join(obj_root, "objects", obj_name, f"{obj_name}.obj")
        mesh_obj = trimesh.load(obj_mesh_path, force='mesh')
        obj_points, _ = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2025)

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

        body_clouds, body_geoms, mj_model = build_local_templates_by_body("data/robots/smpl/smplx_humanoid_v2.xml",
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
        contact_robot, ig_np = contacts_from_xml_pointcloud(
            mj_model, body_clouds,
            qpos_seq=qpos_seq_np.astype(np.float32),
            obj_pts_world=obj_pts_world_np,
            obj_pos=obj_pos_np, vel=obj_vel_np, acc=obj_acc_np,
            sigma_pad=ADAPTIVE_PAD, sigma_no_interact=FIXED_SIGMA_NO_INTERACT,
            ground_height=0.0,
            sk2mj=sk2mj, mj2sk=mj2sk,
            ig_body_pos_world = body_pos_t.numpy().astype(np.float32),
            ig_body_rot_world = new_sk_state.global_rotation.numpy().astype(np.float32),
            ground_mask_sk=ground_mask_sk,
        )

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
                "ig": ref_ig,  # (T,52,3) float32（世界系）
                "contact_robot": contact_robot,  # (T,52)   0/1 float32
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
                contact_row=contact_robot[t],
                body_rot_frame=body_rot_frame,
                mj2sk=mj2sk,
                title=f"seq:{key_str} t={t}",
                body_pos_frame=body_pos_frame,
                ig_frame=ig_np[t],
            )

        if args.render:
            temp_xml = create_temp_xml_with_object(
                "data/robots/smpl/smplx_humanoid_v2.xml",
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
        rel_path = osp.relpath(sequence_dir, args.path)
        save_path = osp.join(output_dir, rel_path, f"{key_str}.npy")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        np.save(save_path, bundle, allow_pickle=True)


    print("Done")


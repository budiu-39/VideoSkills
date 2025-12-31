# Write the converter script to disk without executing it as a CLI.
script_path = "/mnt/data/convert_from_hoi_data_to_bundle.py"
# -*- coding: utf-8 -*-
"""
Convert HOI tensors (as in the user's `_load_motion` output) into the
'bundle' format used by behave2smplx_deprecated.py.

Example:
    python convert_from_hoi_data_to_bundle.py \
        --inputs /path/to/one.pt /path/to/folder \
        --out_root /path/to/save

Assumptions:
- Each input contains a Tensor of shape (T, 1211) matching the
  concatenation order in the user's code.
- We store:
    bundle["motion"]: minimal-but-useful kinematics
    bundle["object"]: obj_pos/rot and velocities
    bundle["interaction"]: ig (T,52,3) and contact_robot (T,52)
- Quaternion conventions are passed through as-is.
"""
import os
import argparse
import pathlib
from scipy.spatial.transform import Rotation as sRot
from typing import List, Dict
import os.path as osp
import numpy as np
import torch
from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
import trimesh
from smpl_sim.smpllib.smpl_parser import SMPLX_Parser
from scripts.utils.smpl_humanoid_tool import  humanoid2smpl
from scripts.preprocess.mujoco_contact_inference import penetration_depth_sequence_ig
from scripts.preprocess.mujoco_contact_inference import build_local_templates_by_body
from scripts.preprocess.mujoco_contact_inference import build_sk2mj_index
# ---------------------- slicing spec inferred from user's code ----------------------
SLICE_SPEC = {
    "root_pos":       (0, 3),      # (T,3)
    "root_rot":       (3, 7),      # (T,4) quat
    "unused0":        (7, 9),      # (T,2) 未使用
    "dof_pos":        (9, 162),    # (T,153)

    "body_pos":       (162, 318),  # (T,156) -> (T,52,3)
    "obj_pos":        (318, 321),  # (T,3)
    "obj_rot":        (321, 325),  # (T,4) quat
    "unused1":        (325, 330),  # (T,5) 未使用

    "contact_obj":    (330, 331),  # (T,1)
    "contact_human":  (331, 383),  # (T,52)

    "body_rot":       (383, 591),  # (T,208) -> (T,52,4) quat
}
FRAME_DIM = 591
FPS_DEFAULT = 30

def _discover_inputs(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for p in inputs:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            for ext in ("*.pt", "*.pth", "*.ptl", "*.bin", "*.pth.tar"):
                paths.extend([str(x) for x in pathlib.Path(p).rglob(ext)])
        else:
            paths.append(p)
    # de-dup preserving order
    seen = set(); uniq = []
    for x in paths:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

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

def _split_slices(arr_2d: torch.Tensor) -> Dict[str, torch.Tensor]:
    if arr_2d.dim() != 2 or arr_2d.size(1) != FRAME_DIM:
        raise ValueError(f"Expected tensor of shape (T,{FRAME_DIM}), got {tuple(arr_2d.shape)}")
    out = {}
    for name, (a, b) in SLICE_SPEC.items():
        out[name] = arr_2d[:, a:b].contiguous()
    return out

def _reshape_fields(s: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    T = s["root_pos"].size(0)
    shaped = {}


    shaped["root_pos"] = s["root_pos"]                            # (T,3)
    shaped["root_rot"] = s["root_rot"]                            # (T,4)

    shaped["dof_pos"] = s["dof_pos"]                              # (T,153)

    shaped["body_pos"] = s["body_pos"].view(T, 52, 3)             # (T,52,3)
    shaped["body_rot"] = s["body_rot"].view(T, 52, 4)             # (T,52,4)

    shaped["obj_pos"] = s["obj_pos"]                              # (T,3)
    shaped["obj_rot"] = s["obj_rot"]                              # (T,4)

    shaped["contact_human"] = s["contact_human"]                  # (T,52)
    shaped["contact_obj"] = s["contact_obj"].squeeze(-1)          # (T,)

    return shaped

def _to_numpy32(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)

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

def compute_sdf(points1, points2):
    # type: (Tensor, Tensor) -> Tensor
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat


smplx_parser_n = SMPLX_Parser(
    model_path='data/SMPL/smplx',
    gender='neutral',
    use_pca=False,  # 关键：不用 PCA，接受 45D/手
    create_transl=False,
    flat_hand_mean=True,
    num_betas=20  # SMPL-X 20 维 beta
)

def convert_single_tensor(hoi_tensor: torch.Tensor, out_path_npy: str, fps: int = FPS_DEFAULT) -> None:
    hoi_tensor = hoi_tensor.detach().to(torch.float32)
    slices = _split_slices(hoi_tensor)
    shaped = _reshape_fields(slices)

    base = os.path.splitext(os.path.basename(out_path_npy))[0]
    obj_name = base.split("_")[-2]

    fix_height = True  # 开启更稳妥
    N  = shaped["root_pos"].size(0)

    skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/smplx_humanoid_hand.xml")

    root_trans_offset = shaped["root_pos"].clone()  # (N,3)

    pose_aa, transl = humanoid2smpl(shaped["body_rot"], root_trans_offset, skeleton_tree, True)  # 这个里的  transl 是错的

    if fix_height:
        with torch.no_grad():
            frame_check = min(100, N)
            pose_t = torch.from_numpy(pose_aa[:frame_check].numpy()).float()  # (F,72) 关键！
            beta_np = np.zeros(20, dtype=np.float32)  # 10 维 betas
            beta_t = torch.from_numpy(beta_np[None, ...]).float()  # (1,10)
            trans_t = root_trans_offset[:frame_check].float()  # (F,3)

            verts, joints = smplx_parser_n.get_joints_verts(pose_t, beta_t, trans_t)
            offset = joints[:, 0] - trans_t
            feet_z = (verts - offset[:, None])[..., -1]
            diff_fix = feet_z.min().item()
            root_trans_offset[..., -1] -= diff_fix


    # Prepare rotations and root translation
    body_rot = shaped["body_rot"]
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        body_rot,  # (T, 52, 4)
        root_trans_offset,  # (T, 3) root translation
        is_local=False
    )

    motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
    motion_dict = motion.to_dict()  # portable structure

    body_pos_t = new_sk_state.global_translation  # Tensor, 与上文一致
    if not isinstance(body_pos_t, torch.Tensor):
        body_pos_t = torch.from_numpy(body_pos_t)
    body_pos_t = body_pos_t.to(torch.float32)  # [T,52,3]


    obj_root = "/home/miku/Documents/VideoSkills/dataset/OMOMO_new/objects"  # 根据你的资源路径调整
    obj_mesh_path = osp.join(obj_root, "objects", obj_name, f"{obj_name}.obj")
    mesh_obj = trimesh.load(obj_mesh_path, force='mesh')

    obj_points_local = torch.from_numpy(
        trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2025)[0].astype(np.float32))

    obj_pts_world = _quat_rotate_xyzw(shaped["obj_rot"], obj_points_local) + shaped["obj_pos"].unsqueeze(1)  # [T,P,3]

    ig_torch = - compute_sdf(shaped["body_pos"], obj_pts_world)  # [T,52,3]
    ref_ig = ig_torch.cpu().numpy().astype(np.float32)

    dt = 1.0 / fps
    obj_pos = shaped["obj_pos"].numpy()  # [T,3]
    obj_pos_vel = np.zeros_like(obj_pos)
    obj_pos_vel[1:] = (obj_pos[1:] - obj_pos[:-1]) / dt
    obj_vel_np = obj_pos_vel.astype(np.float32)  # [T,3]
    obj_acc_np = np.zeros_like(obj_vel_np, dtype=np.float32)
    obj_acc_np[1:] = (obj_vel_np[1:] - obj_vel_np[:-1]) / dt
    obj_rot = shaped["obj_rot"].numpy()  # [T,4]
    obj_rot_vel = angular_velocity_world_from_quat_xyzw(obj_rot, dt)

    body_clouds, body_geoms, mj_model = build_local_templates_by_body("data/robots/smpl/smplx_humanoid_hand.xml",
                                                                      samples_per_geom=500)
    sk2mj, mj2sk = build_sk2mj_index(mj_model, skeleton_tree, drop_world=True)
    pen_seq = penetration_depth_sequence_ig(
        mj_model,
        body_geoms,
        mj2sk,
        obj_pts_world.numpy(),  # e.g. List[np.ndarray], len T, each (P_t,3)
        body_pos_t.numpy().astype(np.float32),  # (T, B, 3)
        new_sk_state.global_rotation.numpy().astype(np.float32),  # (T, B, 4) xyzw
    )

    first_frame_collided = (pen_seq[0] > 0).any()
    if first_frame_collided:
        key_str = os.path.basename(os.path.normpath(out_path_npy))  # 样本名
        print(f"[SKIP-FIRST-FRAME-COLLISION] {key_str}")
        return

    bundle = {
        "motion": motion_dict,
        "object": {
            "name": obj_name,
            "obj_pos": obj_pos,                    # (T,3)
            "obj_rot": obj_rot,                    # (T,4)
            "obj_pos_vel": obj_vel_np,            # (T,3)
            "obj_rot_vel": obj_rot_vel,            # (T,3)
        },
        "interaction": {
            "ig": ref_ig,                              # (T,52,3)
            "contact_robot": _to_numpy32(shaped["contact_human"]),        # (T,52)
            "collision_tag": (pen_seq > 0).any(axis=1)
        }
    }

    os.makedirs(os.path.dirname(out_path_npy), exist_ok=True)
    np.save(out_path_npy, bundle, allow_pickle=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Files or directories containing HOI tensors (.pt/.pth/...).")
    parser.add_argument("--out_root", type=str, required=True,
                        help="Root folder to place converted .npy bundles.")
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT)
    args = parser.parse_args()

    # discover inputs
    in_paths = _discover_inputs(args.inputs)
    if not in_paths:
        print("No inputs found."); return

    for p in in_paths:
        print(f"[LOAD] {p}")
        data = torch.load(p, map_location="cpu")
        # Support dict payloads
        if isinstance(data, dict) and "hoi_data" in data:
            hoi_tensor = data["hoi_data"]
        else:
            hoi_tensor = data

        if not isinstance(hoi_tensor, torch.Tensor):
            raise TypeError(f"Unsupported payload type in {p}: {type(hoi_tensor)}")

        # Optional: squeeze leading batch dim if present
        if hoi_tensor.dim() == 3 and hoi_tensor.size(0) == 1:
            hoi_tensor = hoi_tensor.squeeze(0)

        # Destination
        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(args.out_root, base + ".npy")

        convert_single_tensor(hoi_tensor, out_path, fps=args.fps)
        print(f"[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()

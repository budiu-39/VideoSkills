import os
import sys
import os.path as osp

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
from tqdm import tqdm
import argparse
import glob
from videoskills.utils.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
import joblib
import torch
import mujoco
import time
import pickle

def rotate_root_and_trans(root_aa, trans, R):
    # root_aa: (N,3) 仅 root 的轴角
    r = sRot.from_rotvec(root_aa)
    r_new = sRot.from_matrix(R) * r
    root_aa_new = r_new.as_rotvec()
    trans_new = (np.asarray(R) @ trans.T).T
    return root_aa_new, trans_new

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

def vis_mujoco(motion_traj, xml_path, humanoid_type='g1'):

    print(mujoco.__version__)  # 应该输出 3.2.3
    print(hasattr(mujoco, "viewer"))

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    num_frames = len(motion_traj['root_trans_offset'])

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for t in range(num_frames):
            mj_data.qpos[:3] = motion_traj['root_trans_offset'][t]
            mj_data.qpos[3:7] = motion_traj['root_rotation'][t][[3, 0, 1, 2]]  # Convert from wxyz to xyzw
            mj_data.qpos[7:] = motion_traj['dof'][t].flatten()
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            time.sleep(1 / 30)

import os
import os.path as osp
import json
import numpy as np

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--path", type=str, default="", help="Path to BEHAVE dataset")
    parser.add_argument("--process_split", type=str, default="train", choices=["train", "test", "valid"])
    parser.add_argument("--render", action="store_true", default=False, help="Whether to render the \
                                                                        retargeted motion using scenepic animation.")
    args = parser.parse_args()
    output_dir = "dataset/smpl_motion/behave_small_obj"

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
        "model": "smpl",
    }
    skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/{robot_cfg['model']}_humanoid.xml")
    smpl_local_robot = LocalRobot(robot_cfg, data_dir="data/smpl")

    # BEHAVE dataset structure: each sequence contains SMPL fits and object interactions
    # We look for sequences with SMPL fits
    all_sequences = glob.glob(f"{args.path}/**/", recursive=True)
    behave_full_motion_dict = {}

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
        if D == 156:
            # SMPL-H：global(3) + body(23*3=69) + hands(30*3=90?) → 常见为 156（有的实现手是30*3=90）
            # 只保留 SMPL body（含 global），忽略手部，得到 72 维
            pose_aa_body72 = pose_aa[:, :72].copy()
        elif D == 72:
            # 已是 SMPL body
            pose_aa_body72 = pose_aa
        elif D == 69:
            # 缺少 global；补 3 维零作为 global
            pose_aa_body72 = np.concatenate([np.zeros((pose_aa.shape[0], 3), dtype=pose_aa.dtype),
                                             pose_aa], axis=-1)
        else:
            raise ValueError(f"Unexpected SMPL pose dim {D}. Expect 69/72/156.")


        # 世界坐标系旋转
        R1 = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
        pose_aa_body72[:, :3], root_trans = rotate_root_and_trans(pose_aa_body72[:, :3], root_trans, R1)
        root_trans_offset = torch.from_numpy(root_trans).float() + skeleton_tree.local_translation[0]

        # 关节重排
        pose_aa_mj = pose_aa_body72.reshape(N, 24, 3)
        smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa_mj[:, smpl_2_mujoco]

        # 轴角 -> 四元数（注意 scipy 返回 [x,y,z,w]）
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)
        smpl_parser_n = SMPL_Parser(model_path='data/smpl', gender="neutral")
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True)

        # 高度修正
        frame_check = 100
        height_tolorance = 0
        fix_height = True  # 开启更稳妥
        if fix_height:
            with torch.no_grad():
                frame_check = min(frame_check, N)
                pose_t = torch.from_numpy(pose_aa_body72[:frame_check]).float()  # (F,72) 关键！
                beta_np = np.zeros(10, dtype=np.float32)  # 10 维 betas
                beta_t = torch.from_numpy(beta_np[None, ...]).float()  # (1,10)
                trans_t = root_trans_offset[:frame_check].float()  # (F,3)

                verts, joints = smpl_parser_n.get_joints_verts(pose_t, beta_t, trans_t)
                offset = joints[:, 0] - trans_t
                feet_z = (verts - offset[:, None])[..., -1]
                diff_fix = feet_z.min().item()
                root_trans_offset[..., -1] -= diff_fix




        # 局部坐标系旋转
        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) *
                                sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                                            torch.from_numpy(pose_quat_global),
                                                                            root_trans_offset, is_local=False)
        fps = 30
        motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30)

        # 物体修正
        obj_angles = sequence_data.get('object', {}).get('angles', None)
        obj_trans = sequence_data.get('object', {}).get('trans', None)
        obj_times = sequence_data.get('object', {}).get('frame_times', None)
        obj_angles_new, obj_trans_new = rotate_root_and_trans(obj_angles, obj_trans, R1)

        obj_rot_xyzw = sRot.from_rotvec(obj_angles_new).as_quat().astype(np.float32)
        # 位置
        obj_pos = obj_trans_new.astype(np.float32)
        # 速度
        dt = 1.0 / fps
        obj_pos_vel = np.zeros_like(obj_pos)
        obj_pos_vel[1:] = (obj_pos[1:] - obj_pos[:-1]) / dt

        # 角速度
        def angular_velocity_from_quats_world(q_xyzw, dt):
            T = len(q_xyzw)
            omega_w = np.zeros((T, 3), dtype=np.float32)
            if T <= 1: return omega_w
            r = sRot.from_quat(q_xyzw)
            for t in range(1, T):
                dq = r[t - 1].inv() * r[t]
                rotvec = dq.as_rotvec()
                omega_local = rotvec / dt
                R_world = r[t].as_matrix()
                omega_w[t] = (R_world @ omega_local).astype(np.float32)
            return omega_w

        obj_rot_vel = angular_velocity_from_quats_world(obj_rot_xyzw, dt)





        # Save motion object
        motion.to_dict( )  # Ensure compatibility

        motion_dict = motion.to_dict()  # 与 motion.to_file() 里保存的结构一致
        bundle = {
            "motion": motion_dict,  # SkeletonMotion 的 dict（含关节、根姿态、fps 等）
            "object": {
                "obj_pos": obj_pos,
                "obj_rot": obj_rot_xyzw,  # xyzw —— Isaac Gym 对齐
                "obj_pos_vel": obj_pos_vel,
                "obj_rot_vel": obj_rot_vel,
            }
        }

        # Construct save path
        key_str = os.path.basename(os.path.normpath(sequence_dir))
        rel_path = osp.relpath(sequence_dir, args.path)
        save_path = osp.join(output_dir, rel_path, f"{key_str}.npy")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        np.save(save_path, bundle, allow_pickle=True)

        # Optionally render
        if args.render:
            motion_traj = {}
            motion_traj['root_trans_offset'] = new_sk_state.root_translation.numpy()
            motion_traj['root_rotation'] = new_sk_state.global_root_rotation.numpy()
            motion_traj['dof'] = sRot.from_quat(new_sk_state.local_rotation[:,1:].reshape(-1, 4)).as_rotvec().reshape(N, -1, 3)
            vis_mujoco(motion_traj, f"data/robots/smpl/smpl_humanoid.xml", humanoid_type=robot_cfg['model'])

    print("Done")
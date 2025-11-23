'''
This file is used to transform all smpl to z+ direction.
'''

from scripts.preprocess.face_z_align_util import *
import os
import torch
import copy
from tqdm import tqdm
import argparse
import glob
from smplx import SMPL
from scipy import ndimage
from scripts.preprocess.body_model_smplx import BodyModelSMPLX
import json
import os.path as osp
from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import joblib
from scipy.spatial.transform import Rotation as R

def rot_yaw_z(yaw):
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    return np.array([[cs, -sn, 0],
                     [sn,  cs, 0],
                     [ 0,   0, 1]], dtype=np.float32)


def my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


def calc_heading(q):
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 2] = 1
    rot_dir = my_quat_rotate(q, ref_dir)
    heading = torch.atan2(rot_dir[..., 0], rot_dir[..., 2])
    return heading


def apply_cam2world_rotvec_trans(rotvec, trans, R3x3):
    r_new = sRot.from_matrix(R3x3) * sRot.from_rotvec(rotvec)
    t_new = (R3x3 @ trans.T).T
    return r_new.as_rotvec().astype(np.float32), t_new.astype(np.float32)

if __name__ == '__main__':
    folder_path = "/home/miku/Documents/VideoSkills/logs/smpl_ppo/amass_test_rollout/refine_results/succeed"
    # folder_path = "dataset/humanml3d"
    output_dir = "/home/miku/Documents/VideoSkills/logs/smpl_ppo/amass_test_rollout/292_w_action"
    # npy_files = sorted(glob.glob(os.path.join(folder_path,'*.npy')))
    pkl_files = sorted(glob.glob(os.path.join(folder_path, '*.pkl'), recursive=True))
    os.makedirs(output_dir, exist_ok=True)
    bad_cnt = 0
    for fpath in tqdm(pkl_files):
        motion_data = joblib.load(fpath)
        position_data = motion_data['pred_pos']
        pred_rot = motion_data['pred_rot']

        nfrm, njoint, _ = position_data.shape
        root_idx = 0

        if nfrm > 5000:
            bad_cnt += 1
            continue

        # ================ 检测最后一步是否有 RESET ==================

        # root_pos = position_data[:, root_idx]  # (T, 3)
        # root_pos_diff = np.linalg.norm(root_pos[1:] - root_pos[:-1], axis=-1)  # (T-1,)
        #
        # # 计算 root 在相邻帧间的旋转差（用四元数）
        # root_quat = pred_rot[:, root_idx]  # (T, 4)  这里假设格式是 [x, y, z, w]
        # rot_all = R.from_quat(root_quat)   # (T,)
        # rel_rot = rot_all[1:] * rot_all[:-1].inv()   # 相对旋转 (T-1,)
        # rel_angle = rel_rot.magnitude()             # 每一步的旋转角度（弧度） (T-1,)
        #
        # # 最后一步的跳变
        # last_pos_jump = root_pos_diff[-1]
        # last_rot_jump_deg = np.degrees(rel_angle[-1])
        #
        # print(f"[{os.path.basename(fpath)}] last step: "
        #       f"Δpos={last_pos_jump:.3f} m, Δrot={last_rot_jump_deg:.1f}°")
        #
        # # 简单阈值判断（可以根据你的数据再调）
        # pos_thr = 2.0       # 2 米以上认为可疑
        # rot_thr_deg = 90.0  # 90 度以上认为可疑
        #
        # last_is_reset = (last_pos_jump > pos_thr) or (last_rot_jump_deg > rot_thr_deg)
        # if last_is_reset:
        #     print(f"  -> detected possible RESET at last frame (index {nfrm-1})")

        # ================= 转换到 292 维表示 =======================

        # 取第一帧 root 的平面位置 (x0, y0)
        root_xy0 = position_data[0, root_idx, :2].copy()  # (2,)

        # 所有点的 x、y 减去这两个值，z 不动
        position_data[:, :, 0] -= root_xy0[0]  # 减 x0
        position_data[:, :, 1] -= root_xy0[1]  # 减 y0
        # position_data[:, :, 2] 保持不变 = 不调整高度


        # 计算 root 的世界速度
        velocities_root = position_data[1:, root_idx] - position_data[:-1, root_idx]

        # 从 root 的旋转里提取 yaw（heading）
        root_quat = pred_rot[:, root_idx]  # (T, 4)
        rot_obj = R.from_quat(root_quat)
        rot_mats = rot_obj.as_matrix()
        fx = rot_mats[:, 0, 0]  # forward.x
        fy = rot_mats[:, 1, 0]  # forward.y
        # inverse yaw (Z-up → yaw around Z)
        global_heading = -np.arctan2(fy, fx)

        global_heading_rot = np.stack([rot_yaw_z(a) for a in global_heading], axis=0)  # (T, 3, 3)

        # heading 差
        global_heading_diff = global_heading[1:] - global_heading[:-1]  # (T-1,)
        global_heading_diff_rot = np.stack([rot_yaw_z(a) for a in global_heading_diff], axis=0)  # (T-1, 3, 3)

        # 所有关节位置去 heading（在“身体朝前”坐标系里）
        # 对每一帧，positions_no_heading[t, j] = R_yaw(t) @ position_data[t, j]
        positions_no_heading = np.einsum(
            'tij,tbj->tbi', global_heading_rot, position_data
        )  # (T, J, 3)

        # joint velocities (no heading)
        velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]  # (T-1, J, 3)

        # 把 root_vel 也旋转到 heading frame
        vel_root_no_heading = np.einsum(
            'tij,tj->ti', global_heading_rot[:-1], velocities_root
        )  # (T-1, 3)

        # 只取水平面的 x,z（前后+左右） root_vel_xz（no heading）
        velocities_root_xz_no_heading = vel_root_no_heading[:, [0, 2]]  # (T-1, 2)

        # 旋转矩阵 no-heading + 6D rotation
        # 所有关节的旋转矩阵 (T, J, 3,3)
        body_rot_obj = R.from_quat(pred_rot.reshape(-1, 4))
        rotations_matrix = body_rot_obj.as_matrix().reshape(nfrm, njoint, 3, 3)

        # 把 root 的旋转里也去掉 heading（和原来的代码一样）
        rotations_matrix[:, root_idx] = np.einsum(
            'tij,tjk->tik', global_heading_rot, rotations_matrix[:, root_idx]
        )

        rot6d = rotations_matrix[..., :, :2, :].reshape(nfrm, -1)  # (T, 6*J)

        size_frame = 8 + njoint * 3 + njoint * 3 + njoint * 6 + motion_data['action'].shape[1]
        final_x = np.zeros((nfrm, size_frame), dtype=np.float32)

        # root vel xz (no heading)，第一帧没有速度，设为 0
        final_x[1:, :2] = velocities_root_xz_no_heading  # (T-1, 2)

        # heading delta 6D：第一帧没有 delta，用单位旋转（和你原来一致）
        final_x[0, 2] = 1.0
        final_x[0, 6] = 1.0
        # 假设你有 matrix_to_rotation_6d，也可以先把 global_heading_diff_rot 从 numpy 转 torch
        # 这里给个 numpy 版示意：取前两列 flatten
        heading6d = global_heading_diff_rot[:, :, :2].reshape(nfrm - 1, 6)
        final_x[1:, 2:8] = heading6d  # (T-1, 6)

        # joint positions_no_heading
        final_x[:, 8:8 + 3 * njoint] = positions_no_heading.reshape(nfrm, -1)

        # joint velocities_no_heading（第一帧为 0）
        final_x[1:, 8 + 3 * njoint: 8 + 6 * njoint] = velocities_no_heading.reshape(nfrm - 1, -1)

        # joint rotations 6D
        final_x[:, 8 + 6 * njoint: 8 + 12 * njoint] = rot6d
        final_x[:, 8 + 12 * njoint:] = motion_data['action']

        file_name = os.path.basename(fpath)
        save_path = file_name.replace(".pkl", ".npy")
        output_path = os.path.join(output_dir, save_path)
        np.save(output_path, final_x)
    print(f"bad_cnt: {bad_cnt}")
    print(f"Processed files are saved in 292 dim representation.")




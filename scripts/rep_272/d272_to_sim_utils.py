from scripts.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

import os
import sys
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
import glob
from joblib import Parallel, delayed
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion

# 环境变量
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制 CPU

sys.path.append(os.getcwd())

from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState


def accumulate_rotations_zup(rot_mats_diff):
    """ 累乘旋转矩阵 (Z-up) """
    T = rot_mats_diff.shape[0]
    device = rot_mats_diff.device
    global_rots = torch.zeros((T, 3, 3), device=device)
    current_rot = torch.eye(3, device=device)
    for t in range(T):
        current_rot = torch.matmul(current_rot, rot_mats_diff[t])
        global_rots[t] = current_rot
    return global_rots


def convert_272_and_pad_to_24(data_272, full_tree, fps=30):
    """
    1. 从 272D 还原 22 关节的 Global 数据
    2. 填充到 24 关节 (Pad Hands)
    3. 封装为 SkeletonMotion
    """
    if isinstance(data_272, np.ndarray):
        data_272 = torch.from_numpy(data_272)

    device = torch.device('cpu')
    data_272 = data_272.to(device).float()

    T, D = data_272.shape
    num_joints_in = 22
    num_joints_out = 24

    # === 1. 解析数据 (22关节) ===
    root_vel_xy_local = data_272[:, 0:2]
    heading_diff_6d = data_272[:, 2:8]
    pos_local_no_heading = data_272[:, 8:8 + num_joints_in * 3].reshape(T, num_joints_in, 3)
    rot_global_no_heading_6d = data_272[:, 140:].reshape(T, num_joints_in, 6)

    # === 2. 恢复 Global Heading (积分) ===
    heading_diff_mat = rotation_6d_to_matrix(heading_diff_6d)
    global_heading_mat = accumulate_rotations_zup(heading_diff_mat)  # [T, 3, 3]

    # === 3. 恢复 Global Root Position (积分) ===
    root_vel_local = torch.zeros((T, 3), device=device)
    root_vel_local[:, :2] = root_vel_xy_local

    # 使用上一帧的 Heading 旋转当前帧的速度增量
    heading_for_vel = torch.roll(global_heading_mat, shifts=1, dims=0)
    heading_for_vel[0] = torch.eye(3, device=device)

    root_vel_global = torch.matmul(heading_for_vel, root_vel_local.unsqueeze(-1)).squeeze(-1)
    root_pos_global = torch.cumsum(root_vel_global, dim=0)  # [T, 3]

    # === 4. 恢复 22 关节的 Global Pose ===
    # 恢复位置: P_global = Root + (Heading @ P_local)
    joint_pos_rotated = torch.matmul(
        global_heading_mat.unsqueeze(1),
        pos_local_no_heading.unsqueeze(-1)
    ).squeeze(-1)

    # 22关节 Global Translation
    pos_22 = joint_pos_rotated + root_pos_global.unsqueeze(1)

    # 恢复旋转: R_global = Heading @ R_no_heading
    rot_no_heading_mat = rotation_6d_to_matrix(rot_global_no_heading_6d)
    global_rotation_mat = torch.matmul(
        global_heading_mat.unsqueeze(1),
        rot_no_heading_mat
    )

    # 转四元数 (wxyz -> xyzw for SkeletonState)
    global_quat_wxyz = matrix_to_quaternion(global_rotation_mat)
    rot_22 = global_quat_wxyz[..., [1, 2, 3, 0]]  # [T, 22, 4]

    # === 5. 填充到 24 关节 (Padding) ===
    pos_24 = torch.zeros((T, num_joints_out, 3), dtype=torch.float32, device=device)
    rot_24 = torch.zeros((T, num_joints_out, 4), dtype=torch.float32, device=device)
    rot_24[..., 3] = 1.0  # Identity Quaternion (0,0,0,1)

    # 映射索引 (Drop 掉了 18 和 23)
    # 0-17 -> 0-17
    pos_24[:, :18] = pos_22[:, :18]
    rot_24[:, :18] = rot_22[:, :18]

    # 18-21 (原数据) -> 19-22 (目标数据)
    pos_24[:, 19:23] = pos_22[:, 18:22]
    rot_24[:, 19:23] = rot_22[:, 18:22]

    # 处理缺失的手部 (18, 23)
    # 位置设为父节点位置 (L_Hand->L_Wrist=17, R_Hand->R_Wrist=22)
    pos_24[:, 18] = pos_24[:, 17]
    pos_24[:, 23] = pos_24[:, 22]
    # 旋转保持 Identity

    # === 6. 构建 SkeletonMotion (24 关节) ===
    # is_local=False 表示传入的是 Global Rotation
    sk_state = SkeletonState.from_rotation_and_root_translation(
        full_tree,
        rot_24,
        pos_24[:, 0],  # Root translation
        is_local=False
    )

    return SkeletonMotion.from_skeleton_state(sk_state, fps=fps)
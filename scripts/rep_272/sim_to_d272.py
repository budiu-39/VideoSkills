import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from pytorch3d.transforms import matrix_to_rotation_6d, quaternion_to_matrix, matrix_to_quaternion
from scripts.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from scripts.rep_272.recover_visualize import recover_from_272_zup
from scripts.dataset_process.amass_split import get_amass_splits_hierarchical
from scripts.rep_272.plot_3d_global import draw_to_batch

def get_z_rotation_matrix(theta):
    """ 生成绕 Z 轴旋转的矩阵 (Batch, 3, 3) """
    c = torch.cos(theta)
    s = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    row1 = torch.stack([c, -s, zeros], dim=-1)
    row2 = torch.stack([s, c, zeros], dim=-1)
    row3 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row1, row2, row3], dim=-2)


def normalize_skeleton_motion(skel_motion: SkeletonMotion, root_idx=0, face_joint_idx=[1, 5],
                              target_heading_vec=[1, 0, 0]):
    device = skel_motion.tensor.device

    # 1. 获取基础数据
    global_positions = skel_motion.global_translation.clone().float()

    # [关键修改]: SkeletonState (xyzw) -> PyTorch3D (wxyz)
    # 取出 xyzw
    raw_quat_xyzw = skel_motion.global_rotation.clone().float()
    # 转换为 wxyz: w(3), x(0), y(1), z(2)
    global_rotations_quat = raw_quat_xyzw[..., [3, 0, 1, 2]]

    global_rotations_mat = quaternion_to_matrix(global_rotations_quat)

    # 2. 位置归一化 (保持不变)
    origin = global_positions[0, root_idx].clone()
    z_min = torch.min(global_positions[:, :, 2])
    origin[2] = z_min
    global_positions = global_positions - origin.view(1, 1, 3)

    # 3. 旋转归一化 (保持不变)
    l_hip, r_hip = face_joint_idx
    across_vec0 = global_positions[0, r_hip] - global_positions[0, l_hip]
    across_vec0 = across_vec0 / torch.norm(across_vec0, dim=-1, keepdim=True)
    up_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    forward_vec0 = torch.cross(up_axis, across_vec0, dim=-1)
    forward_vec0 = forward_vec0 / torch.norm(forward_vec0, dim=-1, keepdim=True)

    init_heading = torch.atan2(forward_vec0[1], forward_vec0[0])
    target_angle = torch.atan2(torch.tensor(target_heading_vec[1]), torch.tensor(target_heading_vec[0]))
    angle_diff = target_angle - init_heading

    R_align = get_z_rotation_matrix(angle_diff)

    global_positions = torch.matmul(global_positions, R_align.T)
    global_rotations_mat = torch.matmul(
        R_align.unsqueeze(0).unsqueeze(0),
        global_rotations_mat
    )

    # 4. 重建 SkeletonMotion
    # PyTorch3D 转换回来的是 wxyz
    normalized_quat_wxyz = matrix_to_quaternion(global_rotations_mat)

    # [关键修改]: PyTorch3D (wxyz) -> SkeletonState (xyzw)
    # w(0), x(1), y(2), z(3) -> x(1), y(2), z(3), w(0)
    normalized_quat_xyzw = normalized_quat_wxyz[..., [1, 2, 3, 0]]

    normalized_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=skel_motion.skeleton_tree,
        r=normalized_quat_xyzw,  # 传入 xyzw
        t=global_positions[:, root_idx, :],
        is_local=False
    )
    normalized_motion = SkeletonMotion.from_skeleton_state(normalized_state, fps=skel_motion.fps)

    return normalized_motion


# =========================================================================
# 函数 2: 生成 272D 特征 (修复四元数顺序)
# =========================================================================
def skeleton_motion_to_272d(normalized_motion_22: SkeletonMotion, root_idx=0):
    device = normalized_motion_22.tensor.device

    # 获取数据
    global_positions = normalized_motion_22.global_translation.clone().float()

    # [关键修改]: SkeletonState (xyzw) -> PyTorch3D (wxyz)
    raw_quat_xyzw = normalized_motion_22.global_rotation.clone().float()
    global_rotations_quat = raw_quat_xyzw[..., [3, 0, 1, 2]]

    global_rotations_mat = quaternion_to_matrix(global_rotations_quat)
    n_frames, n_joints, _ = global_positions.shape

    if n_joints != 22:
        raise ValueError(f"Input motion must have 22 joints, got {n_joints}")

    # ... (后续所有计算逻辑保持不变，因为此时 global_rotations_mat 已经是正确的了) ...

    # 1. Root Velocity
    root_traj = global_positions[:, root_idx, :]
    root_vel_global = root_traj[1:] - root_traj[:-1]

    # 2. Local Centering
    curr_root_xy = global_positions[:, root_idx:root_idx + 1, :].clone()
    curr_root_xy[:, :, 2] = 0
    positions_centered = global_positions - curr_root_xy

    # 3. Heading
    l_hip, r_hip = 1, 5
    across = positions_centered[:, r_hip] - positions_centered[:, l_hip]
    across = across / torch.norm(across, dim=-1, keepdim=True)
    up = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).expand_as(across)
    fwd = torch.cross(up, across, dim=-1)
    fwd = fwd / torch.norm(fwd, dim=-1, keepdim=True)

    heading_angles = torch.atan2(fwd[:, 1], fwd[:, 0])

    # 注意: 这里是否需要减去第一帧取决于 normalized_motion 是否完美对齐。
    # 理论上 normalize 后 heading[0]=0，减不减都一样。
    # heading_angles = heading_angles - heading_angles[0]

    heading_rot_inv = get_z_rotation_matrix(-heading_angles)

    # 4. De-heading
    root_vel_vec = root_velocity_global = root_vel_global.unsqueeze(-1)
    root_vel_local = torch.matmul(heading_rot_inv[:-1], root_vel_vec).squeeze(-1)
    root_vel_xy_local = root_vel_local[:, :2]

    heading_diff = heading_angles[1:] - heading_angles[:-1]
    heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
    heading_diff_6d = matrix_to_rotation_6d(get_z_rotation_matrix(heading_diff))

    positions_no_heading = torch.matmul(
        heading_rot_inv.unsqueeze(1),
        positions_centered.unsqueeze(-1)
    ).squeeze(-1)

    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]

    # 这里的 rotations_no_heading 计算是正确的，因为 global_rotations_mat 已经修正过了
    rotations_no_heading = torch.matmul(
        heading_rot_inv.unsqueeze(1),
        global_rotations_mat
    )
    # PyTorch3D 的 6D 表示是从矩阵来的，这里没问题
    rotations_6d = matrix_to_rotation_6d(rotations_no_heading)

    # 5. Concat
    dim_p, dim_v, dim_r = n_joints * 3, n_joints * 3, n_joints * 6
    final_x = torch.zeros((n_frames, 8 + dim_p + dim_v + dim_r), device=device, dtype=torch.float32)

    final_x[0, 2] = 1;
    final_x[0, 6] = 1
    final_x[1:, 0:2] = root_vel_xy_local
    final_x[1:, 2:8] = heading_diff_6d
    final_x[:, 8: 8 + dim_p] = positions_no_heading.reshape(n_frames, -1)
    final_x[1:, 8 + dim_p: 8 + dim_p + dim_v] = velocities_no_heading.reshape(n_frames - 1, -1)
    final_x[:, 8 + dim_p + dim_v:] = rotations_6d.reshape(n_frames, -1)

    return final_x.cpu().numpy()


# =========================================================================
# Metric Calculation
# =========================================================================
def calc_mpjpe(gt, pred):
    """
    计算 MPJPE。因为 pred 是从 272D (Normalized) 还原的，
    所以这里的 gt 也应该是 Normalized 后的数据。
    在此前提下，两者第一帧位置和朝向应该是一致的，不需要额外对齐。
    """
    if isinstance(gt, torch.Tensor): gt = gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor): pred = pred.detach().cpu().numpy()

    # Local MPJPE
    gt_local = gt - gt[:, 0:1, :]
    pred_local = pred - pred[:, 0:1, :]
    diff_local = np.linalg.norm(gt_local - pred_local, axis=-1)
    mpjpe_local = np.mean(diff_local) * 1000

    # Global MPJPE (Trajectory)
    # 因为输入都是 Normalized 的，理论上第 0 帧都在 (0,0,0) 且面朝 X
    # 如果有微小误差，可以对齐第一帧，但理论上可以直接比
    diff_global = np.linalg.norm(gt - pred, axis=-1)
    mpjpe_global = np.mean(diff_global) * 1000

    return mpjpe_local, mpjpe_global


# =========================================================================
# Processing Pipeline
# =========================================================================
def process_file(src_path, dst_root, input_root, dst_root_motion):
    rel_path = os.path.relpath(src_path, input_root)
    save_name = rel_path.replace(os.sep, "-")
    dst_path = os.path.join(dst_root, save_name)
    dst_path_motion = os.path.join(dst_root_motion, save_name)

    if os.path.exists(dst_path):
        return "Skipped"

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    os.makedirs(os.path.dirname(dst_path_motion), exist_ok=True)

    # 1. Load Data
    data = np.load(src_path, allow_pickle=True)
    if src_path.endswith('.npz'):
        pass
    elif isinstance(data, np.ndarray) and data.dtype == object:
        motion_dict = data.item()
    else:
        return "Error"

    skel_motion = SkeletonMotion.from_dict(motion_dict)

    # 2. Normalize (Step A: 全局对齐，保留 24 关节)
    # 返回的是已经摆正的、贴地的 Motion
    motion_norm_24 = normalize_skeleton_motion(skel_motion, root_idx=0)

    # 3. Drop Hands (Step B: 裁剪到 22 关节)
    # 这一步得到的数据用于保存为 .npy (Motion) 以及生成特征
    joints_to_drop = ['L_Hand', 'R_Hand']
    if 'L_Hand' in motion_norm_24.skeleton_tree.node_names:
        motion_norm_22 = motion_norm_24.drop_nodes_by_names(joints_to_drop)

    # 4. Convert to 272D (Step C: 生成特征)
    feature_272 = skeleton_motion_to_272d(motion_norm_22, root_idx=0)

    # 5. Verification (Optional - Local Check)
    # 这一步是在内存里验证，确保数学无误
    pred_xyz = recover_from_272_zup(feature_272, 22)
    gt_xyz = motion_norm_22.global_translation.clone().float().numpy()
    err_l, err_g = calc_mpjpe(gt_xyz, pred_xyz)
    if err_g > 1.0: print(f"Warning: Large error in {src_path}")

    visualize = False
    if visualize:
        file_name = '000'
        output_dir = '/home/miku/Documents/VideoSkills'
        out_path_rec = os.path.join(output_dir, f"{file_name}_recovered.mp4")
        draw_to_batch(
            pred_xyz.reshape(1, -1, 22, 3),
            outname=[out_path_rec],
            fps=30,
            kinetic_chain='sim_22'
        )
        print(f"Saved: {out_path_rec}")

        # Video 2: Normalized Ground Truth
        out_path_gt = os.path.join(output_dir, f"{file_name}_gt_norm.mp4")
        draw_to_batch(
            gt_xyz.reshape(1, -1, 22, 3),
            outname=[out_path_gt],
            fps=30,
            kinetic_chain='sim_22'
        )
        print(f"Saved: {out_path_gt}")

    # 6. Save Files
    np.save(dst_path, feature_272)
    motion_norm_24.to_file(dst_path_motion)

    return "Success"



def main(args):
    print(f"Scanning files in {args.input_root}...")
    train_files, val_files, test_files = get_amass_splits_hierarchical(args.input_root)
    all_files = train_files + val_files + test_files
    print(f"Total files: {len(all_files)}")

    counts = {"Success": 0, "Fail": 0, "Skipped": 0}
    for fpath in tqdm(all_files):
        status = process_file(fpath, args.output_root_272, args.input_root, args.output_root_motion)
        counts[status] += 1

    print(f"Done. {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root_272", type=str, required=True)
    parser.add_argument("--output_root_motion", type=str, required=True)
    args = parser.parse_args()
    main(args)
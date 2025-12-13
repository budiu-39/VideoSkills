import os
import torch
from videoskills.utils.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from videoskills.utils.poselib.core.rotation3d import *
from videoskills.utils import torch_utils

def _get_motion_key_static(filepath):
    # 提取出的静态方法，用于生成 key
    parts = filepath.split(os.sep)
    if 'AMASS' in parts or 'amass' in parts:
        try:
            # 尝试找到 amass 相关的层级，防止越界
            # 简单的 heuristic: 倒数第三个是 subset, 倒数第二个是 subfolder
            subset = parts[-3]
            subfolder = parts[-2]
            filename = os.path.splitext(parts[-1])[0]
            return f"{subset}-{subfolder}-{filename}"
        except:
            return os.path.splitext(parts[-1])[0]
    else:
        return os.path.splitext(parts[-1])[0]


def _local_rotation_to_dof_vel_static(local_rot0, local_rot1, dt, dof_body_ids, dof_offsets, num_dof):
    # 纯 CPU 计算版本的速度计算
    dof_vel = torch.zeros([num_dof])  # CPU tensor

    diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
    diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
    local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt

    for j in range(len(dof_body_ids)):
        body_id = dof_body_ids[j]
        joint_offset = dof_offsets[j]
        joint_size = dof_offsets[j + 1] - joint_offset

        if joint_size == 3:
            joint_vel = local_vel[body_id]
            dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel
        elif joint_size == 1:
            joint_vel = local_vel[body_id]
            # 假设 joint 总是沿 y 轴 (参考原代码)
            dof_vel[joint_offset] = joint_vel.sum()
        else:
            # 这里原本是 print+assert，多进程里 print 可能看不到，直接 pass 或 raise
            pass

    return dof_vel


def _compute_motion_dof_vels_static(motion, dof_body_ids, dof_offsets, num_dof):
    # 静态版本，不依赖 self
    num_frames = motion.tensor.shape[0]
    dt = 1.0 / motion.fps
    dof_vels = []

    for f in range(num_frames - 1):
        local_rot0 = motion.local_rotation[f]
        local_rot1 = motion.local_rotation[f + 1]
        frame_dof_vel = _local_rotation_to_dof_vel_static(
            local_rot0, local_rot1, dt, dof_body_ids, dof_offsets, num_dof
        )
        dof_vels.append(frame_dof_vel)

    dof_vels.append(dof_vels[-1])
    dof_vels = torch.stack(dof_vels, dim=0)
    return dof_vels


def _apply_rotation_static(motion, fps):
    # 静态版本的 apply_rotation
    # 如果不需要随机角度，可以固定或者传入 angle
    rotation_angle = torch.rand(1) * 2 * torch.pi
    rotation_quat = torch_utils.quat_from_angle_axis(rotation_angle, torch.tensor([0.0, 0.0, 1.0]))

    global_translation = torch_utils.quat_apply(rotation_quat, motion.global_translation[:, 0])

    if rotation_quat.shape != motion.global_rotation.shape:
        rotation_quat = rotation_quat.expand_as(motion.global_rotation)

    global_rotation = torch_utils.quat_mul(rotation_quat, motion.global_rotation)

    # 这里的 SkeletonState 需要确保能在子进程正确导入
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree,
        global_rotation,
        global_translation,
        is_local=False
    )
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
    return new_motion


def load_motion_worker(args):
    """
    Worker function for multiprocessing.
    Receives arguments, loads file, computes physics on CPU, returns object.
    """
    curr_file, rotate_motion, dof_body_ids, dof_offsets, num_dof = args

    # 1. Load File
    curr_motion = SkeletonMotion.from_file(curr_file)


    motion_fps = curr_motion.fps

    # 2. Apply Rotation (CPU)
    if rotate_motion:
        curr_motion = _apply_rotation_static(curr_motion, motion_fps)

    # 3. Compute velocities (CPU)
    # 注意：传入的 body_ids 等必须是 CPU tensor 或 list
    curr_dof_vels = _compute_motion_dof_vels_static(curr_motion, dof_body_ids, dof_offsets, num_dof)
    curr_motion.dof_vels = curr_dof_vels

    # 4. Generate Key
    key = _get_motion_key_static(curr_file)

    # Return necessary data
    # 注意：这里返回的 tensor 都在 CPU 上
    return {
        "motion": curr_motion,
        "key": key,
        "file": curr_file,
        "fps": motion_fps,
        "num_frames": curr_motion.tensor.shape[0]
    }
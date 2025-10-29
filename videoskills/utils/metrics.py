# following code is copied from PHC

import os
import sys

sys.path.append(os.getcwd())

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from scipy.spatial.transform import Rotation as sRot
from scipy.ndimage import uniform_filter1d


def compute_metrics(pred_pos_all, gt_pos_all, pred_rot_all=None, gt_rot_all=None, root_idx=0,
                                 use_tqdm=True):
    # Returns motion-level metrics (1 value per motion)
    metrics = defaultdict(list)
    valid_mask = []

    if use_tqdm:
        pbar = tqdm(range(len(pred_pos_all)))
    else:
        pbar = range(len(pred_pos_all))

    for idx in pbar:
        jpos_pred = pred_pos_all[idx]  # shape: [T, J, 3]
        jpos_gt = gt_pos_all[idx]  # shape: [T, J, 3]
        rot_pred = pred_rot_all[idx] if pred_rot_all is not None else None
        rot_gt = gt_rot_all[idx] if gt_rot_all is not None else None
        T = jpos_pred.shape[0]

        if T < 3:
            valid_mask.append(False)
            continue  # skip this motion, or you can record default values

        valid_mask.append(True)

        # Global MPJPE (before removing root)
        mpjpe_g = np.linalg.norm(jpos_gt - jpos_pred, axis=2).mean() * 1000

        # Velocity and acceleration error
        vel_dist = compute_error_vel(jpos_pred, jpos_gt).mean() * 1000
        accel_dist = compute_error_accel(jpos_pred, jpos_gt).mean() * 1000

        # Root-relative (local) coordinates
        jpos_pred_local = jpos_pred - jpos_pred[:, [root_idx]]
        jpos_gt_local = jpos_gt - jpos_gt[:, [root_idx]]

        # Procrustes-aligned MPJPE
        # pa_mpjpe = p_mpjpe(jpos_pred_local, jpos_gt_local).mean() * 1000
        mpjpe_l = np.linalg.norm(jpos_pred_local - jpos_gt_local, axis=2).mean() * 1000

        metrics["mpjpe_g"].append(mpjpe_g)
        metrics["mpjpe_l"].append(mpjpe_l)
        # metrics["mpjpe_pa"].append(pa_mpjpe)
        metrics["vel_dist"].append(vel_dist)
        metrics["accel_dist"].append(accel_dist)

        if rot_pred is not None and rot_gt is not None:
            # Flatten and compute rotation error (in rad)
            rot_pred_flat = rot_pred.reshape(-1, 4)
            rot_gt_flat = rot_gt.reshape(-1, 4)
            rot_error = np.linalg.norm(
                (sRot.from_quat(rot_gt_flat) * sRot.from_quat(rot_pred_flat).inv()).as_rotvec(),
                axis=-1
            ).mean()
            metrics["rot_error"].append(rot_error)

    return metrics, valid_mask


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)

# def compute_error_accel(joints_gt, joints_pred, vis=None):
#     """
#     Computes acceleration error:
#         1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
#     Note that for each frame that is not visible, three entries in the
#     acceleration error should be zero'd out.
#     Args:
#         joints_gt (Nx14x3).
#         joints_pred (Nx14x3).
#         vis (N).
#     Returns:
#         error_accel (N-2).
#     """
#     # (N-2)x14x3
#     accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
#     accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]
#
#     normed = np.linalg.norm(accel_pred - accel_gt, axis=2)
#
#     if vis is None:
#         new_vis = np.ones(len(normed), dtype=bool)
#     else:
#         invis = np.logical_not(vis)
#         invis1 = np.roll(invis, -1)
#         invis2 = np.roll(invis, -2)
#         new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
#         new_vis = np.logical_not(new_invis)
#
#     return np.mean(normed[new_vis], axis=1)


# def compute_vel(joints):
#     velocities = joints[1:] - joints[:-1]
#     velocity_normed = np.linalg.norm(velocities, axis=2)
#     return np.mean(velocity_normed, axis=1)

def compute_error_vel(pred, gt):
    return np.linalg.norm(np.diff(pred, axis=0) - np.diff(gt, axis=0), axis=2)

def compute_error_accel(pred, gt):
    return np.linalg.norm(np.diff(pred, n=2, axis=0) - np.diff(gt, n=2, axis=0), axis=2)


# from CLoSD

def zup_to_yup(motions):
    idx = torch.tensor([0, 2, 1], device=motions.device)
    return motions.index_select(dim=2, index=idx)


def physical_metrics(pred_motion):
    results = {
        'skate_ratio': [],
        'mean_penetration': [],
        'penetration': [],
        'floating': [],
        'skating': []
    }

    for m in pred_motion:                 # m: numpy [T, J, 3]
        L = m.shape[0]
        lengths = [L]

        # -> torch [1, J, 3, L]，只取前22个关节
        m_t = torch.tensor(m[:, :22], dtype=torch.float32).permute(1, 2, 0).unsqueeze(0)
        m_t = zup_to_yup(m_t)

        # 调用指标函数
        skate_ratio, skate_vel = calculate_skating_ratio(m_t)
        mean_penetration = calculate_mean_penetration(m_t)
        penetration = calculate_penetration(m_t, lengths)
        floating = calculate_floating(m_t, lengths)
        skating = calculate_foot_sliding(m_t, lengths)

        results['skate_ratio'].append(float(np.mean(skate_ratio)))
        results['mean_penetration'].append(float(mean_penetration))
        results['penetration'].append(float(penetration))
        results['floating'].append(float(floating))
        results['skating'].append(float(skating))

    return results


def calculate_skating_ratio(motions):
    '''
    Code adopted from the GMD codebase.
    '''
    thresh_height = 0.05  # 10
    fps = 20.0
    thresh_vel = 0.50  # 20 cm /s
    avg_window = 5  # frames

    batch_size = motions.shape[0]
    # 10 left, 11 right foot. XZ plane, y up
    # motions [bs, 22, 3, max_len]
    verts_feet = motions[:, [10, 11], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],
                                          axis=2) * fps  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height),
                                  (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :])  # [bs, max_len -1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]

    return skating_ratio, skate_vel


def calculate_mean_penetration(motions, mask=None):
    '''
    Mean penetration of the joints into the ground.
    '''
    import torch
    # samples shape:  B x n_joints x 3 x L
    joint_heights = motions[:, :, 1]
    penetration = torch.clamp(joint_heights, max=0)  # B x n_joints x L
    if mask is not None:
        penetration *= mask.view(mask.shape[0], 1, mask.shape[-1], 1)
    per_joint_penalty = penetration.amin(dim=1).sum() * -1.
    per_joint_frame_penalty = penetration.amin(dim=1).amin(dim=-1).sum() * -1.
    non_zero_count = torch.sum(penetration != 0, dtype=torch.float32)

    return per_joint_frame_penalty.numpy() * 100.  # / non_zero_count # convert to centimeters


def calculate_penetration(motions, lengths):
    '''
    Based on PhysDiff's implementation.
    '''
    penetration_list = []
    joint_heights = motions[:, :, 1, :]  # B x n_joints x L
    lowest_heights = joint_heights.min(dim=1)[0]  # B x L
    tolerance = 0.005
    for i, motion_len in enumerate(lengths):
        penetration = (lowest_heights[i, :motion_len] + tolerance).clamp_max(0) * 1000 * -1.
        penetration_list += penetration.tolist()

    return np.mean(penetration_list)


def calculate_floating(motions, lengths):
    '''
    Based on PhysDiff's implementation.
    '''
    floating_list = []
    joint_heights = motions[:, :, 1, :]  # B x n_joints x L
    lowest_heights = joint_heights.min(dim=1)[0]  # B x L
    tolerance = 0.005
    for i, motion_len in enumerate(lengths):
        floating = (lowest_heights[i, :motion_len] - tolerance).clamp_min(0) * 1000
        floating_list += floating.tolist()

    return np.mean(floating_list)


def calculate_foot_sliding(motions, lengths):
    '''
    Based on PhysDiff's implementation.
    '''
    import torch
    skating_list = []
    margin = 0.005
    for joints, motion_len in zip(motions, lengths):
        # joints: n_joints x 3 x L
        for t in range(motion_len - 1):
            contact_idx = joints[:, 1, t].min(dim=-1)[1]
            if joints[contact_idx, 1, t] <= margin and joints[contact_idx, 1, t + 1] <= margin:
                offset = joints[contact_idx, [0, 2], t + 1] - joints[contact_idx, [0, 2], t]  # horizontal distance
                skate_i = torch.norm(offset).mean().item() * 1000
            else:
                skate_i = 0.
            skating_list.append(skate_i)

    return np.mean(skating_list)

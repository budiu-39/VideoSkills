# following code is copied from PHC
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import numpy as np

import torch
import numpy as np
import pickle as pk
from tqdm import tqdm
from collections import defaultdict

from scipy.spatial.transform import Rotation as sRot


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


#
# def compute_error_vel(joints_gt, joints_pred, vis=None):
#     vel_gt = joints_gt[1:] - joints_gt[:-1]
#     vel_pred = joints_pred[1:] - joints_pred[:-1]
#     normed = np.linalg.norm(vel_pred - vel_gt, axis=2)
#
#     if vis is None:
#         new_vis = np.ones(len(normed), dtype=bool)
#     return np.mean(normed[new_vis], axis=1)
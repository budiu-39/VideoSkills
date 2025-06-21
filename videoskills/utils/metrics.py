



def compute_metrics_lite_motions(pred_pos_all, gt_pos_all, pred_rot_all=None, gt_rot_all=None, root_idx=0,
                                 use_tqdm=True):
    # Returns motion-level metrics (1 value per motion)
    metrics = defaultdict(list)

    for idx in pbar:
        jpos_pred = pred_pos_all[idx]  # shape: [T, J, 3]
        jpos_gt = gt_pos_all[idx]  # shape: [T, J, 3]
        rot_pred = pred_rot_all[idx] if pred_rot_all is not None else None
        rot_gt = gt_rot_all[idx] if gt_rot_all is not None else None

        # Global MPJPE (before removing root)
        mpjpe_g = np.linalg.norm(jpos_gt - jpos_pred, axis=2).mean() * 1000

        # Velocity and acceleration error
        vel_dist = compute_error_vel(jpos_pred, jpos_gt).mean() * 1000
        accel_dist = compute_error_accel(jpos_pred, jpos_gt).mean() * 1000

        # Root-relative (local) coordinates
        jpos_pred_local = jpos_pred - jpos_pred[:, [root_idx]]
        jpos_gt_local = jpos_gt - jpos_gt[:, [root_idx]]

        # Procrustes-aligned MPJPE
        pa_mpjpe = p_mpjpe(jpos_pred_local, jpos_gt_local).mean() * 1000
        mpjpe_l = np.linalg.norm(jpos_pred_local - jpos_gt_local, axis=2).mean() * 1000

        metrics["mpjpe_g"].append(mpjpe_g)
        metrics["mpjpe_l"].append(mpjpe_l)
        metrics["mpjpe_pa"].append(pa_mpjpe)
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

    return metrics
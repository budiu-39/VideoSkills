import numpy as np
import torch

def compare_root_joint_smplx_vs_bodymodel(
    smplh_layer,          # smplx.SMPLHLayer 或兼容 forward(pose, betas, trans)
    body_model,           # human_body_prior.body_model.body_model.BodyModel
    pose_zup_np,          # (T, 156) for SMPL-H axis-angle (root+body+hands)  or whatever your pipeline uses
    tran_zup_np,          # (T, 3)
    betas_10,             # torch.Tensor (10,)
    device=None,
    smplx_root_joint_idx=0,
    bm_root_joint_idx=0,
):
    """
    输出：
      - pelvis_smplx: (T,3)
      - pelvis_bm:    (T,3)
      - diff:         (T,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smplh_layer = smplh_layer.to(device)
    body_model = body_model.to(device)

    # ---- prepare tensors ----
    pose = torch.from_numpy(pose_zup_np).float().to(device)      # (T, P)
    trans = torch.from_numpy(tran_zup_np).float().to(device)     # (T, 3)
    betas = betas_10.view(1, -1).float().to(device)              # (1, 10)
    betas_T = betas.expand(pose.shape[0], -1).contiguous()       # (T, 10)

    # ---- SMPLX/SMPLHLayer forward ----
    with torch.no_grad():
        out_smplx = smplh_layer(pose, th_betas=betas_T, th_trans=trans)
        # 常见返回：out_smplx[0]=verts, out_smplx[1]=joints
        verts_smplx = out_smplx[0]               # (T, V, 3)
        joints_smplx = out_smplx[1]              # (T, J, 3)
        pelvis_smplx = joints_smplx[:, smplx_root_joint_idx, :]  # (T,3)

    # ---- BodyModel forward ----
    # BodyModel 的参数名可能不同：常见是 root_orient / pose_body / pose_hand / trans / betas
    # 你现在 pose_zup 是一个拼好的 full pose (axis-angle)：
    # 通用做法：拆成 root(3) + body(63) + lh(45) + rh(45) = 156
    # 如果你的 pose 不是 156，请按你的格式改 slicing。
    T, P = pose.shape
    if P < 3:
        raise ValueError(f"pose dim too small: {P}")

    root_orient = pose[:, 0:3]  # (T,3)

    # 下面按 SMPL-H 的常见拼法：root(3) + body(63) + lh(45) + rh(45)
    # 若你的顺序不同，请调整
    pose_body = pose[:, 3:3+63] if P >= 3+63 else None
    pose_lh = pose[:, 3+63:3+63+45] if P >= 3+63+45 else None
    pose_rh = pose[:, 3+63+45:3+63+45+45] if P >= 3+63+45+45 else None

    bm_kwargs = dict(
        betas=betas_T,
        trans=trans,
        root_orient=root_orient,
    )
    if pose_body is not None:
        bm_kwargs["pose_body"] = pose_body
    # human_body_prior 有些版本用 pose_hand / pose_hand_l / pose_hand_r，下面两种都尝试
    if pose_lh is not None and pose_rh is not None:
        # 优先常见字段
        bm_kwargs["pose_hand"] = torch.cat([pose_lh, pose_rh], dim=1)

    with torch.no_grad():
        out_bm = body_model(**bm_kwargs)

    # BodyModel 输出字段在不同版本里可能叫：Jtr / joints / v / verts
    # 下面做一个兼容读取
    joints_bm = None
    if isinstance(out_bm, dict):
        for k in ["Jtr", "joints", "J", "joint_positions"]:
            if k in out_bm:
                joints_bm = out_bm[k]
                break
    else:
        # 有些版本返回一个对象/tuple
        if hasattr(out_bm, "Jtr"):
            joints_bm = out_bm.Jtr
        elif hasattr(out_bm, "joints"):
            joints_bm = out_bm.joints
        elif isinstance(out_bm, (list, tuple)) and len(out_bm) >= 2:
            # 不保证顺序，只能兜底
            joints_bm = out_bm[1]

    if joints_bm is None:
        raise RuntimeError("Cannot find joints from BodyModel output. Please print(out_bm) to inspect keys/attrs.")

    pelvis_bm = joints_bm[:, bm_root_joint_idx, :]  # (T,3)

    # ---- Compare ----
    pelvis_smplx_np = pelvis_smplx.detach().cpu().numpy()
    pelvis_bm_np = pelvis_bm.detach().cpu().numpy()
    diff = np.linalg.norm(pelvis_smplx_np - pelvis_bm_np, axis=1)

    print("Root/Pelvis joint comparison (SMPLHLayer vs BodyModel):")
    print(f"  mean L2 diff: {diff.mean():.6f}")
    print(f"  max  L2 diff: {diff.max():.6f}")
    print(f"  min  L2 diff: {diff.min():.6f}")

    return pelvis_smplx_np, pelvis_bm_np, diff


import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

def smpl_get_pelvis(smpl_layer, pose_np, trans_np, betas_10, device=None, pelvis_idx=0):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smpl_layer = smpl_layer.to(device)

    pose = torch.from_numpy(pose_np).float().to(device)     # (T, P)
    trans = torch.from_numpy(trans_np).float().to(device)   # (T, 3)

    betas = betas_10.float()
    if betas.ndim == 1:
        betas = betas[None, :]
    betas = betas.to(device).expand(pose.shape[0], -1)      # (T,10)

    with torch.no_grad():
        out = smpl_layer(pose, th_betas=betas, th_trans=trans)
        joints = out[1]   # (T, J, 3)
        pelvis = joints[:, pelvis_idx, :]  # (T,3)

    return pelvis.detach().cpu().numpy()

def check_equivalence_with_known_R(smpl_layer, pose_yup, trans_yup, pose_zup, trans_zup, betas_10, R_yup2zup, pelvis_idx=0):
    pelvis_y = smpl_get_pelvis(smpl_layer, pose_yup, trans_yup, betas_10, pelvis_idx=pelvis_idx)
    pelvis_z = smpl_get_pelvis(smpl_layer, pose_zup, trans_zup, betas_10, pelvis_idx=pelvis_idx)

    # 预测的 zup pelvis：R @ pelvis_yup
    pelvis_z_pred = (R_yup2zup @ pelvis_y.T).T

    err = np.linalg.norm(pelvis_z - pelvis_z_pred, axis=1)
    print("pelvis_zup vs R @ pelvis_yup")
    print(f"  mean L2 err: {err.mean():.8f}")
    print(f"  max  L2 err: {err.max():.8f}")
    print(f"  min  L2 err: {err.min():.8f}")

    return err, pelvis_y, pelvis_z, pelvis_z_pred

import torch
import smplx

def compute_root_pelvis_offset(model_path: str, model_type: str, gender: str = "neutral", device="cpu"):
    """
    返回 pelvis - root 的 offset（单位取决于你的模型：通常是 meters 或 mm，看你模型文件）
    model_type: "smplh" 或 "smplx"
    """
    model = smplx.create(
        model_path=model_path,
        model_type=model_type,
        gender=gender,
        use_pca=False,
        batch_size=1,
    ).to(device)

    # ---- 关键：rest pose ----
    # pose=0, betas=0, transl=0
    betas = torch.zeros([1, model.num_betas], device=device)
    transl = torch.zeros([1, 3], device=device)

    # smplx/smplh 的 forward 接口略有差异；但 body_pose 都是 (1, 63) (21*3)
    body_pose = torch.zeros([1, 63], device=device)

    # smplx 还多了手/脸等；全部置 0 即 rest
    kwargs = dict(betas=betas, transl=transl, body_pose=body_pose)

    if model_type.lower() == "smplx":
        kwargs.update({
            "global_orient": torch.zeros([1, 3], device=device),
            "left_hand_pose": torch.zeros([1, 45], device=device),
            "right_hand_pose": torch.zeros([1, 45], device=device),
            "jaw_pose": torch.zeros([1, 3], device=device),
            "leye_pose": torch.zeros([1, 3], device=device),
            "reye_pose": torch.zeros([1, 3], device=device),
            "expression": torch.zeros([1, model.num_expression_coeffs], device=device),
        })
    else:  # smplh
        kwargs.update({
            "global_orient": torch.zeros([1, 3], device=device),
            "left_hand_pose": torch.zeros([1, 45], device=device),
            "right_hand_pose": torch.zeros([1, 45], device=device),
        })

    out = model(**kwargs)

    # out.joints: (1, J, 3)  —— 这些 joints 是 model 定义的关节（通常 joint 0 就是 pelvis）
    joints = out.joints[0]  # (J, 3)

    # 绝大多数 SMPL 系列：joint 0 = pelvis
    pelvis = joints[0]

    offset = pelvis
    return offset.detach().cpu().numpy()


if __name__ == "__main__":
    model_path = 'data/SMPL'
    offset = compute_root_pelvis_offset(model_path, 'smplx', gender = "neutral"
                                    , device = "cuda")
    print(offset)
import os
import sys
import os.path as osp

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
from tqdm import tqdm
import argparse
import glob
import trimesh
import torch
import mujoco
import numpy as np
import smplx

from scripts.libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPLH_MUJOCO_NAMES, SMPLH_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_parser import SMPLX_Parser
from scripts.smpl2sim.hoi.mujoco_contact_inference import build_local_templates_by_body
from scripts.smpl2sim.hoi.mujoco_contact_inference import quick_viz_frame, build_sk2mj_index
from scripts.render.mujoco_render import export_mujoco_video_hoi, create_temp_xml_with_object
from scripts.render.render_smplh_hoi import render_smplh_hoi_video
from scripts.smpl2sim.hoi.hoi_retarget_utils import (load_behave_sequence, apply_cam2world_rotvec_trans,
                                                     _quat_rotate_xyzw,
                                                     angular_velocity_world_from_quat_xyzw, penetration_depth_sequence_ig,
                                                     compute_cg_ig_via_smplh_contacts_yup, smplh_vert_part_from_custom_layer)


def from_yup_to_simulation(human, obj, smpl_model, smplx_parser_n, skeleton_tree):
    R_yup2zup_mat = sRot.from_euler('x', np.pi / 2).as_matrix()

    device = next(smpl_model.parameters()).device

    # 1. 准备数据
    poses = human['poses']  # (T, D)
    trans = human['trans']
    betas = human['betas']

    obj_trans = obj['trans']  # (T, 3)
    obj_angles = obj['angles']  # (T, 3)
    obj_name = obj['name']


    # 2. 获取 yup 坐标系下的 pelvis 轨迹， 计算物体相对 pelvis 的位置
    with torch.no_grad():
        smplx_output = smpl_model(
            body_pose=torch.from_numpy(poses[:, 3:66]).float().to(device),
            global_orient=torch.from_numpy(poses[:, :3]).float().to(device),
            betas=torch.from_numpy(betas).float().to(device),
            transl=torch.from_numpy(trans).float().to(device)
        )
        pelvis = smplx_output.joints.detach().cpu().numpy()[:, 0, :]

    obj_delta_yup = obj_trans - pelvis  # (T, 3)
    # 将位移向量轨迹整体旋转到世界系
    rel_pelvis_obj_world = sRot.from_matrix(R_yup2zup_mat).apply(obj_delta_yup)

    # 3. 计算人体在 Z up 下的旋转和平移
    pose_aa = poses.copy()
    pose_aa[:, :3], root_trans_origin_world = apply_cam2world_rotvec_trans(
        poses[:, :3], trans, R_yup2zup_mat
    )

    # 4. 高度修正，这里直接使用 SMPL-X Parser（等效于机器人骨骼）
    with torch.no_grad():
        f_check = min(100, pose_aa.shape[0])
        p_t = torch.from_numpy(pose_aa[:f_check]).float()
        t_t = torch.from_numpy(root_trans_origin_world[:f_check]).float()
        verts, _ = smplx_parser_n.get_joints_verts(p_t, torch.zeros((1, 20)), t_t)
        diff_fix = verts[..., -1].min().item()

    final_root_trans = root_trans_origin_world - diff_fix

    # 5. 物体旋转 (位置相对于人体平移，只需独立旋转角度)
    obj_pos = final_root_trans + rel_pelvis_obj_world
    obj_angles_w, _ = apply_cam2world_rotvec_trans(
        obj_angles, obj_trans, R_yup2zup_mat
    )

    # 6. 计算物体额外信息（速度、角速度），默认 30 fps
    dt = 1.0 / 30
    obj_pos_vel = np.zeros_like(obj_pos)
    obj_pos_vel[1:] = (obj_pos[1:] - obj_pos[:-1]) / dt

    obj_quat = sRot.from_rotvec(obj_angles_w).as_quat()  # (T,4) xyzw
    obj_rot_vel = angular_velocity_world_from_quat_xyzw(obj_quat, dt)

    # 7. 把 Zup SMPL (final_root_trans 和 pose_aa) 格式化成合适的 SkeletonMotion 格式
    # 7.1 先做骨骼重排序
    N = pose_aa.shape[0]
    pose_aa_mj = pose_aa.reshape(N, 52, 3)
    smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
    pose_aa_mj = pose_aa_mj[:, smpl_2_mujoco]

    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 52, 4)

    # 局部坐标系旋转（这个是骨架区别，因为 SMPL 的旋转坐标系是Yup, 而机器人是 Zup）
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat),
        torch.from_numpy(final_root_trans).float(),  # 显式转为 Tensor
        is_local=True)

    pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) *
                        sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
    new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                                    torch.from_numpy(pose_quat_global),
                                                                    torch.from_numpy(final_root_trans).float(),
                                                                    is_local=False)

    object = {"name": obj_name, "obj_pos": obj_pos, "obj_rot": obj_quat,
              "obj_pos_vel": obj_pos_vel, "obj_rot_vel": obj_rot_vel}

    return new_sk_state, object
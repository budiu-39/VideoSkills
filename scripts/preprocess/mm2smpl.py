import os
import sys
import os.path as osp

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
from tqdm import tqdm
import argparse
import glob
from utils.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from scripts.ms_utils import recover_from_local_rotation, smpl85_2_smpl322
import joblib
import torch
import mujoco
import time

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

def rot6d_to_rotmat(x):  # x: (..., 6)
    # Zhou et al. CVPR'19 的常见实现
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)
    return R

def rot6d_to_quat(x):  # x: (..., 6) -> (..., 4) in xyzw
    R = rot6d_to_rotmat(x)
    return sRot.from_matrix(R.reshape(-1, 3, 3)).as_quat().reshape(*x.shape[:-1], 4)

def apply_cam2world_rotvec_trans(rotvec, trans, R3x3):
    r_new = sRot.from_matrix(R3x3) * sRot.from_rotvec(rotvec)
    t_new = (R3x3 @ trans.T).T
    return r_new.as_rotvec().astype(np.float32), t_new.astype(np.float32)

def quat_mul_xyzw(q, r):
    """
    Hamilton product q * r，均为(...,4)[x,y,z,w]，返回同形状。
    支持批量和广播，例如 (T,1,4) 与 (T,J,4)。
    """
    x1, y1, z1, w1 = np.split(q, 4, axis=-1)
    x2, y2, z2, w2 = np.split(r, 4, axis=-1)
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.concatenate([x, y, z, w], axis=-1)

def pose_272_to_smpl(data_272):
    smpl_85_data = recover_from_local_rotation(data_272, 22)  # get the 85-dim smpl data
    if len(smpl_85_data.shape) == 3:
        smpl_85_data = np.squeeze(smpl_85_data, axis=0)

    pose = smpl85_2_smpl322(smpl_85_data)

    assert pose.shape[1] == 322
    use_flame = (pose.shape[1] == 322)

    root_and_body = pose[:, :66].reshape(-1, 22, 3)

    if use_flame:
        trans = pose[:, 309:309 + 3]
    else:
        trans = pose[:, 159:159 + 3]

    trans = trans.reshape(-1, 3)
    return trans, root_and_body

if __name__ == "__main__":

    output_dir = "dataset/smpl_motion/Kungfu"
    upright_start = True
    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": upright_start,
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

    smpl_local_robot = LocalRobot(robot_cfg, data_dir="data/SMPL/smpl")
    folder_path = "/home/miku/Documents/Dataset/kungfu"
    npy_files = sorted(glob.glob(os.path.join(folder_path,'*.npy')))
    framerate = 30

    for fpath in tqdm(npy_files):
        rel_path = os.path.relpath(fpath, folder_path)
        motion = np.load(fpath, allow_pickle=True)
        skip = int(framerate / 30)

        N = motion.shape[0]

        root_trans, pose_aa = pose_272_to_smpl(motion)

        # 经典字段划分
        pose_aa_smpl = np.zeros((N, 24, 3), dtype=pose_aa.dtype)
        pose_aa_smpl[:, :22, :] = pose_aa
        # ------- 组装成 24 关节（缺失补单位四元数）-------

        smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
        R_cam2world = [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]
        skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/{robot_cfg['model']}_humanoid.xml")
        root_trans_offset = root_trans + skeleton_tree.local_translation[0].numpy()
        pose_aa_smpl[:, 0], root_trans_offset = apply_cam2world_rotvec_trans(pose_aa_smpl[:, 0], root_trans_offset, R_cam2world)
        root_trans_offset = torch.from_numpy(root_trans_offset)
        pose_aa_mj = pose_aa_smpl[:, smpl_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

        beta = np.zeros(16)
        gender_number, beta[:], gender = [0], 0, "neutral"



        smpl_parser_n = SMPL_Parser(model_path='data/SMPL/smpl', gender="neutral")
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True)

        frame_check = 100
        height_tolorance = 0
        fix_height = True
        if fix_height:
            with torch.no_grad():
                frame_check = min(frame_check, N)
                pose_t = torch.from_numpy(pose_aa_smpl[:frame_check])
                beta_t = torch.from_numpy(beta[None,])
                trans_t = root_trans_offset[:frame_check]

                verts, joints = smpl_parser_n.get_joints_verts(pose_t, beta_t, trans_t)
                offset = joints[:, 0] - trans_t # offset is the difference between the smpl root joint and the mujoco root joint
                feet_z = (verts - offset[:, None])[..., -1]  # Z 轴  # feet_z is the mujoco lower point of each frame
                diff_fix = feet_z.min().item() # diff_fix is the lowest frame of lowest point in each frame

                root_trans_offset[..., -1] -= diff_fix
                # we move the mujoco root joint down by the lowest point of the smpl feet, so that the feet are on the ground.


        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) *
                                sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)

            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                                            torch.from_numpy(pose_quat_global),
                                                                            root_trans_offset, is_local=False)

        fps = 30

        motion_obj = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)

        # 构建保存路径
        save_path = osp.join(output_dir, rel_path).replace(".npz", ".npy")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        # 保存 motion 对象为 numpy 文件
        motion_obj.to_file(save_path)
        motion_traj = {}
        motion_traj['root_trans_offset'] = new_sk_state.root_translation.numpy()
        motion_traj['root_rotation'] = new_sk_state.global_root_rotation.numpy()
        motion_traj['dof'] = sRot.from_quat(new_sk_state.local_rotation[:,1:].reshape(-1, 4)).as_rotvec().reshape(N, -1, 3)
        # vis_mujoco(motion_traj, f"data/robots/smpl/smpl_humanoid.xml", humanoid_type=robot_cfg['model'])


    print("Done")




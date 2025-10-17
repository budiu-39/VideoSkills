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

if __name__ == "__main__":

    output_dir = "AMASS_valid_fixed_height"
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
        entry_data = np.load(fpath, allow_pickle=True)
        skip = int(framerate / 30)
        root_trans = entry_data['trans'][::skip, :]
        pose_aa = np.concatenate([entry_data['poses'][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
        betas = entry_data['betas']
        gender = entry_data['gender']
        N = pose_aa.shape[0]

        if bound == 0:
            bound = N

        root_trans = root_trans[:bound]
        pose_aa = pose_aa[:bound]
        N = pose_aa.shape[0]
        if N < 10:
            continue

        smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

        beta = np.zeros(16)
        gender_number, beta[:], gender = [0], 0, "neutral"

        # smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
        # smpl_local_robot.write_xml(f"data/robots/smpl/{robot_cfg['model']}_0_humanoid.xml")
        skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/{robot_cfg['model']}_humanoid.xml")
        # This is the root translation offset, which is the distance from the SMPL root to the skeleton root.
        # 也就是说，机器人和 smpl 的root 几乎一致。
        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]
        smpl_parser_n = SMPL_Parser(model_path='data/smpl', gender="neutral")

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
                pose_t = torch.from_numpy(pose_aa[:frame_check])
                beta_t = torch.from_numpy(beta[None,])
                trans_t = root_trans_offset[:frame_check]

                verts, joints = smpl_parser_n.get_joints_verts(pose_t, beta_t, trans_t)
                offset = joints[:, 0] - trans_t # offset is the difference between the smpl root joint and the mujoco root joint
                feet_z = (verts - offset[:, None])[..., -1]  # Z 轴  # feet_z is the mujoco lower point of each frame
                diff_fix = feet_z.min().item() # diff_fix is the lowest frame of lowest point in each frame

                root_trans_offset[..., -1] -= diff_fix
                # we move the mujoco root joint down by the lowest point of the smpl feet, so that the feet are on the ground.

                # if abs(diff_fix) > 0.1:
                dataset = path_parts[-3]
                subset = path_parts[-2]
                filename = path_parts[-1].replace(".npy", "")
                key_str = f"{dataset}-{subset}-{filename}"

                fix_height_keys.append(key_str)
                fix_height_values.append(diff_fix)


        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) *
                                sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)

            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                                            torch.from_numpy(pose_quat_global),
                                                                            root_trans_offset, is_local=False)

        fps = 30

        motion_obj = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)

        # 构建保存路径
        rel_path = osp.relpath(data_path, args.path)  # 相对路径，如 CMU/123/xxx.npz
        save_path = osp.join(output_dir, rel_path).replace(".npz", ".npy")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        # 保存 motion 对象为 numpy 文件
        motion_obj.to_file(save_path)
        motion_traj = {}
        motion_traj['root_trans_offset'] = new_sk_state.root_translation.numpy()
        motion_traj['root_rotation'] = new_sk_state.global_root_rotation.numpy()
        motion_traj['dof'] = sRot.from_quat(new_sk_state.local_rotation[:,1:].reshape(-1, 4)).as_rotvec().reshape(N, -1, 3)
        vis_mujoco(motion_traj, f"data/robots/smpl/smpl_humanoid.xml", humanoid_type=robot_cfg['model'])

    fix_height_dict = {k: round(v, 5) for k, v in zip(fix_height_keys, fix_height_values)}
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(fix_height_dict, osp.join(output_dir, "fixed_height_keys.pkl"))
    fix_height_dict_load = joblib.load(osp.join(output_dir, "fixed_height_keys.pkl"))

    print("Done")




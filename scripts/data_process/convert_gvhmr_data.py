import os
import sys
sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
import joblib
from tqdm import tqdm
import argparse

from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
import torch
import mujoco.viewer


def rotate(pose, trans, rotate_matrix = [[1., 0., 0.], [0., 0., 1], [0., -1., 0.]]):
    pose[:, :3] = torch.tensor(
        (sRot.from_matrix(rotate_matrix) * sRot.from_rotvec(pose))
        .as_rotvec().reshape(-1, 3))

    trans = (torch.tensor(rotate_matrix) @ trans.transpose(1, 0)).transpose(1, 0)

    return pose, trans

def process_folder(folder_path):

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

    smpl_local_robot = LocalRobot(robot_cfg,data_dir="phc/data/smpl")

    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    amass_full_motion_dict = {}


    for subfolder in tqdm(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):

            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                if os.path.isfile(file_path) and file_name.endswith('.pt'):  # 假设是 .pkl 文件
                    with open(file_path, 'rb') as f:

                        data = torch.load(f, map_location='cpu')

                        root_trans = data['smpl_params_global']['transl']
                        betas = data['smpl_params_global']['betas']
                        pose_aa = data['smpl_params_global']['body_pose']
                        zeros_tensor = torch.zeros((root_trans.shape[0], 6), device=pose_aa.device,
                                                   dtype=pose_aa.dtype)
                        # Concatenate along the last dimension
                        global_orient = data['smpl_params_global']['global_orient']
                        pose_aa = torch.cat([global_orient, pose_aa, zeros_tensor], dim=-1)

                        skeleton_tree = SkeletonTree.from_mjcf(
                            f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
                        root_trans_offset = root_trans + skeleton_tree.local_translation[0]


                        pose_aa[:, :3], root_trans_offset = rotate(pose_aa[:, :3], root_trans_offset.squeeze())
                        pose_aa[:, :3], root_trans_offset = rotate(pose_aa[:, :3], root_trans_offset.squeeze(),
                                                                   [[1., 0., 0.], [0., -1., 0.], [0., 0., -1]])

                        N = pose_aa.shape[0]
                        pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
                        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

                        beta = np.zeros(16)
                        gender_number, beta[:], gender = [0], 0, "neutral"

                        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]),
                                                            gender=gender_number, objs_info=None)
                        smpl_local_robot.write_xml(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")

                        new_sk_state = SkeletonState.from_rotation_and_root_translation(
                            skeleton_tree,
                            # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
                            torch.from_numpy(pose_quat),
                            root_trans_offset,
                            is_local=True)


                        if robot_cfg['upright_start']:

                            pose_quat_global = (sRot.from_quat(
                                new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat(
                                [0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)  # should fix pose_quat as well here...

                            # The following code applies a coordinate system transformation for joint local rotation
                            # and produces the same effect as the previous code.

                            # pose_quat_local = (sRot.from_quat(
                            #     [0.5, 0.5, 0.5, 0.5]) * sRot.from_quat(
                            #     new_sk_state.local_rotation.reshape(-1, 4).numpy()) * sRot.from_quat(
                            #     [0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
                            #
                            # root_quat =(sRot.from_quat(
                            #     [0.5, 0.5, 0.5, 0.5]).inv() * sRot.from_quat(pose_quat_local[:,0])).as_quat().reshape(N, 4)

                            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                                                            torch.from_numpy(
                                                                                                pose_quat_global),
                                                                                            root_trans_offset,
                                                                                            is_local=False)

                        pose_quat_global = new_sk_state.global_rotation.numpy()
                        pose_quat = new_sk_state.local_rotation.numpy()
                        # pose_aa = sRot.from_quat(pose_quat.reshape(-1, 4)).as_rotvec().reshape(-1, 72)
                        # Extract data
                        new_motion_out = {}
                        key_name_dump = subfolder
                        new_motion_out['pose_quat_global'] = pose_quat_global
                        new_motion_out['pose_quat'] = pose_quat
                        new_motion_out['trans_orig'] = root_trans
                        new_motion_out['root_trans_offset'] = root_trans_offset.double()
                        new_motion_out['beta'] = beta
                        new_motion_out['gender'] = gender
                        new_motion_out['pose_aa'] = pose_aa
                        # pose aa is pose for smpl motion, spl motion lab is build on smpl. I see.
                        # But why pose aa seems to only affect the position rather than the rotation?
                        # Because pose_aa is just used to fix height!!! So the index of it is base on smpl rather than
                        # smpl humanoid
                        new_motion_out['fps'] = 30

                        amass_full_motion_dict[key_name_dump] = new_motion_out

    return amass_full_motion_dict

def vis_mujoco(motion_traj):
    import mujoco
    import time

    xml_path = f"phc/data/assets/mjcf/smpl_humanoid.xml"

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    num_frames = len(motion_traj['root_trans_offset'])

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for t in range(num_frames):
            mj_data.qpos[:3] = motion_traj['root_trans_offset'][t]
            mj_data.qpos[3:7] = motion_traj['pose_quat'][t][0][[3, 0, 1, 2]]
            mj_data.qpos[7:] = sRot.from_quat(motion_traj['pose_quat'][t][1:].reshape(-1,4)).as_euler(
                "XYZ").flatten()

            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            time.sleep(1/30)


def quaternion_distance(q1, q2):
    # Ensure the quaternions are normalized
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute the dot product between the two quaternions
    dot_product = np.dot(q1, q2)

    # Ensure the dot product is within the valid range [-1, 1]
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute the angle between the two quaternions
    angle = 2 * np.arccos(dot_product)

    return angle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str, default='output/GVHMR_output/new_folder')
parser.add_argument('--output_path', type=str, default='output/Humanoid_motion/dance.pkl')
parser.add_argument('--pkl_per_motoin', type=bool, default=False)
args = parser.parse_args()
folder_path = args.folder_path
output_path = args.output_path
pkl_per_motoin = args.pkl_per_motoin
result = process_folder(folder_path)
pkl_per_motoin = False
if pkl_per_motoin is True:
    for key in result.keys():
        motion = {}
        motion[key] = result[key]
        output_path = os.path.join(args.output_path, key + ".pkl")
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            joblib.dump(motion, f)
        # vis_mujoco(motion[key])

joblib.dump(result, output_path, compress=True)
print(f"Processing is complete and the data has been saved to {output_path}")

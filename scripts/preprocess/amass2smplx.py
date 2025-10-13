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
from smpl_sim.smpllib.smpl_joint_names import SMPLH_MUJOCO_NAMES, SMPLH_BONE_ORDER_NAMES
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--process_split", type=str, default="train", choices=["train", "test", "valid"])
    parser.add_argument("--render", action="store_true", default=False, help="Whether to render the \
                                                                        retargeted motion using scenepic animation.")
    args = parser.parse_args()
    output_dir = "SMPLX_AMASS"

    process_split = args.process_split
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

    if not osp.isdir(args.path):
        print("Please specify AMASS data path")
        import ipdb;

        ipdb.set_trace()

    all_pkls = glob.glob(f"{args.path}/**/*.npz", recursive=True)
    amass_occlusion = joblib.load("data/amass_copycat_occlusion_v3.pkl")
    amass_full_motion_dict = {}
    amass_splits = {
        'valid': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'KIT', 'EKUT', 'TCD_handMocap', "BMLhandball", "DanceDB",
                  "ACCAD", "BMLmovi", "BioMotionLab_NTroje", "Eyes_Japan_Dataset", "DFaust_67"]  # Adding ACCAD
    }
    process_set = amass_splits[process_split]
    length_acc = []
    fix_height_keys = []
    fix_height_values = []
    for data_path in tqdm(all_pkls):
        print("Processing", data_path)
        bound = 0

        path_parts = data_path.split("/")
        for _, part in enumerate(path_parts):
            if 'amass' in path_parts or 'AMASS' in part:
                amass_parts = part # ['AMASS_processed']
        if amass_parts in path_parts:
            amass_index = path_parts.index(amass_parts)
            splits = path_parts[amass_index + 1]  # 获取 AMASS 后的下一个部分作为 split
            key_name = path_parts[amass_index + 1:]
        else:
            print("AMASS not found in path:", data_path)
            continue

        key_name_dump = "0-" + "_".join(key_name).replace(".npz", "")

        if (not splits in process_set):
            print("Skipping", splits, key_name_dump)
            continue

        if key_name_dump in amass_occlusion:
            issue = amass_occlusion[key_name_dump]["issue"]
            if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[key_name_dump]:
                bound = amass_occlusion[key_name_dump]["idxes"][0]  # This bounded is calucaled assuming 30 FPS.....
                if bound < 10:
                    print("bound too small", key_name_dump, bound)
                    continue
            else:
                print("issue irrecoverable", key_name_dump, issue)
                continue

        entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

        if not 'mocap_framerate' in entry_data:
            continue
        framerate = entry_data['mocap_framerate']

        if "0-KIT_442_PizzaDelivery02_poses" == key_name_dump:
            bound = -2

        skip = int(framerate / 30)
        root_trans = entry_data['trans'][::skip, :]
        pose_aa = np.concatenate([entry_data['poses'][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
        N = pose_aa.shape[0]

        smpl_name_to_idx = {n: i for i, n in enumerate(SMPL_BONE_ORDER_NAMES)}
        smplx_name_to_idx = {n: i for i, n in enumerate(SMPLH_BONE_ORDER_NAMES)}
        Jx = len(SMPLH_BONE_ORDER_NAMES)  # ~52

        pose_aa_x = np.zeros((N, Jx, 3), dtype=np.float32)
        shared_body = [n for n in SMPL_BONE_ORDER_NAMES if (n in smplx_name_to_idx)]
        for name in shared_body:
            pose_aa_x[:, smplx_name_to_idx[name], :] = pose_aa[:, smpl_name_to_idx[name], :]


        betas = entry_data['betas']
        gender = entry_data['gender']
        N = pose_aa.shape[0]

        if bound == 0:
            bound = N

        root_trans = root_trans[:bound]
        pose_aa = pose_aa[:bound]

        if N < 10:
            continue

        smplx_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa.reshape(N, 52, 3)[:, smplx_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 52, 4)

        beta = np.zeros(16)
        gender_number, beta[:], gender = [0], 0, "neutral"

        skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/smplx_humanoid_v2.xml")
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
        vis_mujoco(motion_traj, f"data/robots/smpl/smplx_humanoid_v2.xml", humanoid_type=robot_cfg['model'])

    print("Done")




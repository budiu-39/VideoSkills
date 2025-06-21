import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import torch
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import joblib
from tqdm import tqdm
import argparse
import glob
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_parser import SMPL_Parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--process_split", type=str, default="train", choices=["train", "test", "valid"])
    parser.add_argument("--render", action="store_true", default=False, help="Whether to render the \
                                                                        retargeted motion using scenepic animation.")
    args = parser.parse_args()

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

    smpl_local_robot = LocalRobot(robot_cfg, data_dir="data/smpl")
    if not osp.isdir(args.path):
        print("Please specify AMASS data path")
        import ipdb;

        ipdb.set_trace()

    all_pkls = glob.glob(f"{args.path}/**/*.npz", recursive=True)
    amass_occlusion = joblib.load("output/SMPL_Robot_motion/amass_copycat_occlusion_v3.pkl")
    amass_full_motion_dict = {}
    amass_splits = {
        'valid': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'KIT', 'EKUT', 'TCD_handMocap', "BMLhandball", "DanceDB",
                  "ACCAD", "BMLmovi", "BioMotionLab_NTroje", "Eyes_Japan_Dataset", "DFaust_67"]  # Adding ACCAD
    }
    process_set = amass_splits[process_split]
    length_acc = []
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

        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
        smpl_local_robot.write_xml(f"data/robots/smpl/{robot_cfg['model']}_0_humanoid.xml")
        skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/{robot_cfg['model']}_0_humanoid.xml")
        # This is the root translation offset, which is the distance from the SMPL root to the skeleton root.
        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]
        # fixed_height_offset
        smpl_parser_n = SMPL_Parser(model_path='data/smpl', gender="neutral")


        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True)

        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) *
                                sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
            # 这个的作用是把 SMPL 的姿态从 Y-up 转换为 Z-up。

            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                                            torch.from_numpy(pose_quat_global),
                                                                            root_trans_offset, is_local=False)

        frame_check = 200
        height_tolorance = 0
        vertices_curr, joints_curr = smpl_parser_n.get_joints_verts(torch.from_numpy(pose_aa[:frame_check]),
                                                                    torch.from_numpy(beta[None,]),
                                                                    torch.from_numpy(root_trans[:frame_check]))
        offset = joints_curr[:, 0] - root_trans_offset[:frame_check]
        diff_fix = ((vertices_curr - offset[:, None])[:frame_check, ..., -1].min(
            dim=-1).values - height_tolorance).min()
        root_trans -= diff_fix.numpy()

        pose_quat_global = new_sk_state.global_rotation.numpy()
        pose_quat = new_sk_state.local_rotation.numpy()
        fps = 30

        # new_motion_out = {}
        # new_motion_out['pose_quat_global'] = pose_quat_global
        # new_motion_out['pose_quat'] = pose_quat
        # new_motion_out['trans_orig'] = root_trans
        # new_motion_out['root_trans_offset'] = root_trans_offset
        # new_motion_out['fix_height_offset'] = -diff_fix
        # new_motion_out['beta'] = beta
        # new_motion_out['gender'] = gender
        # new_motion_out['pose_aa'] = pose_aa
        # new_motion_out['fps'] = fps
        # amass_full_motion_dict[key_name_dump] = new_motion_out

        motion_obj = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)

        motion_light = {}
        motion_light['num_frames'] = N
        motion_light['fps'] = fps
        motion_light['local_rotation'] = pose_quat_global
        motion_light['global_velocity'] = motion_obj.global_velocity.numpy()
        motion_light['root_trans_offset'] = root_trans_offset.numpy()
        motion_light['diff_fix'] = (-diff_fix).numpy()



        # 构建保存路径
        rel_path = osp.relpath(data_path, args.path)  # 相对路径，如 CMU/123/xxx.npz
        save_path = osp.join("AMASS_processed", rel_path).replace(".npz", ".npy")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        # 保存 motion 对象为 numpy 文件
        motion_obj.to_file(save_path)

        # if args.render:
        #     vis_motion_use_scenepic_animation(
        #         asset_filename=f"phc/data/assets/mjcf/{robot_cfg['model']}_0_humanoid.xml",
        #         rigidbody_global_pos=motion_obj.global_translation,
        #         rigidbody_global_rot=motion_obj.global_rotation,
        #         fps=fps,
        #         up_axis="z",
        #         color= np.array([0.94, 0.97, 1.00]) * 255,
        #         output_path=osp.join('output/retarget_render', f"{key_name_dump}_render.html"),
        #     )

    # if upright_start:
    #     os.makedirs("output/Humanoid_motion/amass", exist_ok=True)
    #     joblib.dump(amass_full_motion_dict, f"output/Humanoid_motion/amass/amass_selected_{process_split}.pkl", compress=True)
    # else:
    #     os.makedirs("output/Humanoid_motion/amass", exist_ok=True)
    #     joblib.dump(amass_full_motion_dict, "output/Humanoid_motion/amass/amass_train_take6.pkl", compress=True)
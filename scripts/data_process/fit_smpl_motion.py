import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from smpl_sim.utils import torch_utils
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from hydra.utils import to_absolute_path
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser, 
)

import joblib
import torch
import torch.nn.functional as F
import math
from smpl_sim.utils.pytorch3d_transforms import axis_angle_to_matrix
from torch.autograd import Variable
from tqdm import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from phc.utils.torch_humanoid_batch import Humanoid_Batch
from smpl_sim.utils.smoothing_utils import gaussian_kernel_1d, gaussian_filter_1d_batch
from easydict import EasyDict
import hydra
from omegaconf import DictConfig, OmegaConf


def rotate(pose, trans, rotate_matrix = [[1., 0., 0.], [0., 0., 1], [0., -1., 0.]]):

    pose[:, :3] = torch.tensor(
        (sRot.from_matrix(rotate_matrix) * sRot.from_rotvec(pose))   ## 这里的 * 不是代表矩阵乘法，而是 rotation * rotation！
        .as_rotvec().reshape(-1, 3), dtype=torch.float32)
    # trans = torch.tensor(trans - obj_centroid[obj_name], dtype=torch.float32)
    trans = (torch.tensor(rotate_matrix, dtype=torch.float32) @ trans.transpose(1, 0)).transpose(1, 0)

    return pose, trans

def load_amass_data(data_path):

    try:
        with np.load(data_path, allow_pickle=True) as data:
            entry_data = dict(data)
    except Exception as e:
        print(f"Failed to load npz file: {data_path} | Error: {e}")
        return None

    if not 'mocap_framerate' in  entry_data:
        return 
    framerate = entry_data['mocap_framerate']


    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": framerate
    }


def load_GVHMR_data(data_path):
    entry_data = torch.load(data_path, map_location='cpu')['smpl_params_global']
    # entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    framerate = 30

    root_trans = entry_data['transl']
    pose_aa = entry_data['body_pose']
    zeros_tensor = torch.zeros((root_trans.shape[0], 6), device=pose_aa.device,
                               dtype=pose_aa.dtype)
    # Concatenate along the last dimension
    global_orient =entry_data['global_orient']
    pose_aa = torch.cat([global_orient, pose_aa, zeros_tensor], dim=-1)

    betas = entry_data['betas']
    gender = "neutral"
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate
    }

def process_motion(key_names, key_name_to_pkls, cfg):
    device = torch.device("cpu")

    humanoid_fk = Humanoid_Batch(cfg.robot) # load forward kinematics model
    num_augment_joint = len(cfg.robot.extend_config)

    #### Define corresonpdances between h1 and smpl joints
    robot_joint_names_augment = humanoid_fk.body_names_augment 
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
    robot_joint_pick_idx = [robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    smpl_parser_n = SMPL_Parser(model_path="phc/data/smpl", gender="neutral")
    shape_new, scale = joblib.load(f"output/Unitree_motion/{cfg.robot.humanoid_type}/shape_optimized_v1.pkl") # TODO: run fit_smple_shape to get this

    
    all_data = {}
    pbar = tqdm(key_names, position=0, leave=True)

    for data_key in key_names:
        if cfg.robot.input_motion_type == "AMASS":
            process_data = load_amass_data(key_name_to_pkls[data_key])
            if process_data is None: continue
            skip = int(process_data['fps'] // 30)
            trans = torch.from_numpy(process_data['trans'][::skip])
            pose_aa_walk = torch.from_numpy(process_data['pose_aa'][::skip]).float()
        elif cfg.robot.input_motion_type == "GVHMR":
            process_data = load_GVHMR_data(key_name_to_pkls[data_key])
            if process_data is None: continue
            pose_aa_walk = process_data['pose_aa']
            trans = process_data['trans']
            # This is because of the difference between GVHMR and AMASS
            pose_aa_walk[:, :3], trans = rotate(pose_aa_walk[:, :3], trans.squeeze())
            pose_aa_walk[:, :3], trans = rotate(pose_aa_walk[:, :3], trans.squeeze(),
                                                       [[1., 0., 0.], [0., -1., 0.], [0., 0., -1]])

        N = trans.shape[0]
        if N < 10:
            print("to short")
            continue

        with torch.no_grad():
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            root_pos = joints[:, 0:1]
            joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos
        joints[..., 2] -= verts[0, :, 2].min().item()
        
            
        offset = joints[:, 0] - trans
        root_trans_offset = (trans + offset).clone()

        # It can be thought as the initialization of rot (so???)c why??
        # this is rotating the root of Humnanoid
        gt_root_rot_quat = torch.from_numpy((sRot.from_rotvec(pose_aa_walk[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()).float() # can't directly use this
        gt_root_rot = torch.from_numpy(sRot.from_quat(torch_utils.calc_heading_quat(gt_root_rot_quat)).as_rotvec()).float() # so only use the heading.

        dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))
        # the coordinate system of h1 and smpl robot are different, so we need to rotate the root rotation, since there is not
        # rotation for the joint of smpl robot, we want to optimize it. Therefore we also need to calculate the rotation between it
        # and correct root rotation? is it?    H1 G1 coordniate correct
        dof_pos_new = Variable(dof_pos.clone(), requires_grad=True)
        root_rot_new = Variable(gt_root_rot.clone(), requires_grad=True)
        root_pos_offset = Variable(torch.zeros(1, 3), requires_grad=True)
        # optimizer_pose = torch.optim.Adam([dof_pos_new],lr=0.01)
        # optimizer_root = torch.optim.Adam([root_rot_new, root_pos_offset],lr=0.01)
        optimizer = torch.optim.Adam([dof_pos_new, root_rot_new, root_pos_offset],lr=0.02)
        # print("root_rot_new_origin", root_rot_new[0])
        kernel_size = 5  # Size of the Gaussian kernel
        sigma = 0.75  # Standard deviation of the Gaussian kernel


        for iteration in range(cfg.get("fitting_iterations", 500)):
            pose_aa_h1_new = torch.cat([root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new, torch.zeros((1, N, num_augment_joint, 3)).to(device)], axis = 2)
            fk_return = humanoid_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ] + root_pos_offset )
            
            if num_augment_joint > 0:
                diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            else:
                diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
                
            loss_g = diff.norm(dim = -1).mean() + 0.01 * torch.mean(torch.square(dof_pos_new))
            loss = loss_g
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])

            pbar.set_description_str(f"{data_key}-Iter: {iteration} \t {loss.item() * 1000:.3f}")

            dof_pos_new.data = gaussian_filter_1d_batch(dof_pos_new.squeeze().transpose(1, 0)[None, ], kernel_size, sigma).transpose(2, 1).contiguous()[..., None]
        pbar.update(1)
        dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])
        pose_aa_h1_new = torch.cat([root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new, torch.zeros((1, N, num_augment_joint, 3)).to(device)], axis = 2)

        root_trans_offset_dump = (root_trans_offset + root_pos_offset).clone()

        combined_mesh = humanoid_fk.mesh_fk(pose_aa_h1_new[:, :1].detach(), root_trans_offset_dump[None, :1].detach())
        height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()
        
        root_trans_offset_dump[..., 2] -= height_diff
        joints_dump = joints.numpy().copy()
        joints_dump[..., 2] -= height_diff

        data_dump = {
                    "root_trans_offset": root_trans_offset_dump.squeeze().detach().numpy(),
                    "pose_aa": pose_aa_h1_new.squeeze().detach().numpy(),   
                    "dof": dof_pos_new.squeeze().detach().numpy(),
                    "smpl_joints": joints_dump, 
                    "fps": 30
                    }

        all_data[data_key] = data_dump
    return all_data

def vis_mujoco(motion_traj, humanoid_type = 'h1'):
    import mujoco
    import time
    print(mujoco.__version__)  # 应该输出 3.2.3
    print(hasattr(mujoco, "viewer"))

    xml_path = f"/home/miku/Documents/PHC/phc/data/assets/robot/unitree_{humanoid_type}/{humanoid_type}.xml"

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    num_frames = len(motion_traj['root_trans_offset'])

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for t in range(num_frames):
            mj_data.qpos[:3] = motion_traj['root_trans_offset'][t]
            mj_data.qpos[3:7] = sRot.from_rotvec(motion_traj['pose_aa'][t][0]).as_quat()[[3, 0, 1, 2]]
            mj_data.qpos[7:] = motion_traj['dof'][t].flatten()
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            time.sleep(1/30)

def match_amass_dataset(key_name, all_datasets):
    """
    Match the dataset name from the key name.
    """
    suffix = key_name.split("-")[1]
    for dataset in all_datasets:
        if suffix.startswith(dataset + "_"):
            return dataset
    return None


@hydra.main(version_base=None, config_path="../../phc/data/cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    import ipdb
    if ("gvhmr_path" in cfg and cfg.gvhmr_path) and ("amass_root" in cfg and cfg.amass_root):
        raise ValueError("Both 'gvhmr_path' and 'amass_root' are provided. Please specify only one.")
    elif not ("gvhmr_path" in cfg and cfg.gvhmr_path) and not ("amass_root" in cfg and cfg.amass_root):
        raise ValueError("Neither 'gvhmr_path' nor 'amass_root' is provided. Please specify one.")

    if "gvhmr_path" in cfg and cfg.gvhmr_path:
        cfg.robot.input_motion_type = "GVHMR"
        pt_files = glob.glob(osp.join(cfg.gvhmr_path, "*", "*.pt"))
        key_name_to_pkls = {
            osp.basename(osp.dirname(f)): f for f in pt_files
        }
    else:
        cfg.robot.input_motion_type = "AMASS"
        if "amass_root" in cfg:
            amass_root = cfg.amass_root
        else:
            raise ValueError("amass_root is not specified in the config")

        all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
        split_len = len(amass_root.split("/"))
        key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for
                            data_path in all_pkls}

        amass_occlusion = joblib.load("output/Humanoid_motion/amass_copycat_occlusion_v3.pkl")
        amass_full_motion_dict = {}
        amass_splits = {
            'valid': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
            'test': ['Transitions_mocap', 'SSM_synced'],
            'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'KIT', 'EKUT', 'TCD_handMocap', "BMLhandball", "DanceDB",
                      "ACCAD", "BMLmovi", "BioMotionLab_NTroje", "Eyes_Japan_Dataset", "DFaust_67"]  # Adding ACCAD
        }
        all_datasets = set(sum(amass_splits.values(), []))  # flatten list
        # ==== AMASS check ====
        print(f"Final pkl count: {len(key_name_to_pkls)}")

        # 2. amass_occlusion item count
        print(f"Aamass_occlusion item count: {len(amass_occlusion)}")

        # 3.  split count
        split_count = {'train': 0, 'valid': 0, 'test': 0}
        n = 0
        for key in key_name_to_pkls:
            split_label = match_amass_dataset(key, all_datasets)
            n = n + 1
            a = 0
            for split in amass_splits:
                if split_label in amass_splits[split]:
                    split_count[split] += 1
                    a += 1
                if a > 1:
                    print("multiple splits", key, split_label)

        print("Split counts:")
        for split, count in split_count.items():
            print(f"  {split}: {count}")
        print("Total count:", n)

        # 4. check if all splits are present in amass_root path
        all_expected_splits = set(sum(amass_splits.values(), []))  # flatten list
        splits_found = set([match_amass_dataset(key, all_datasets) for key in key_name_to_pkls])
        missing_splits = all_expected_splits - splits_found

        if missing_splits:
            print("Missing splits in amass_root path:")
            for m in sorted(missing_splits):
                print(f"  - {m}")
        else:
            print("all split found in amass_root path")


        # ==== AMASS check ====

        keys_to_remove = []
        for key in key_name_to_pkls.keys():
            splits = match_amass_dataset(key, all_datasets)
            if splits not in amass_splits[cfg.robot.process_split]:
                keys_to_remove.append(key)
                # print("not in process split", key_name, splits)
            elif key in amass_occlusion:
                issue = amass_occlusion[key]["issue"]
                if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[key]:
                    bound = amass_occlusion[key]["idxes"][0]  # This bounded is calucaled assuming 30 FPS.....
                    if bound < 10:
                        # print("bound too small", key_name, bound)
                        keys_to_remove.append(key)
                else:
                    # print("issue irrecoverable", key_name, issue)
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del key_name_to_pkls[key]


        # ==== AMASS check ====
        from collections import Counter

        removal_split_counter = Counter()
        for key in keys_to_remove:
            split_name = key.split("-")[1].split("_")[0]
            removal_split_counter[split_name] += 1

        print("Removals per split:")
        for split, count in sorted(removal_split_counter.items()):
            print(f"  {split}: {count}")

        split_count = {'train': 0, 'valid': 0, 'test': 0}
        for key in key_name_to_pkls:
            split_label = match_amass_dataset(key, all_datasets)
            for split in amass_splits:
                if split_label in amass_splits[split]:
                    split_count[split] += 1

        print("Split counts after filtering:")
        for split, count in split_count.items():
            print(f"  {split}: {count}")

        # ipdb.set_trace()

        # ==== AMASS check ====


    from multiprocessing import Pool
    key_names = list(key_name_to_pkls.keys())
    jobs = key_names
    num_jobs = cfg.num_jobs
    chunk = np.ceil(len(jobs)/num_jobs).astype(int)
    jobs= [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    job_args = [(jobs[i], key_name_to_pkls, cfg) for i in range(len(jobs))]

    if len(job_args) == 1:
        all_data = process_motion(key_names, key_name_to_pkls, cfg)
    else:
        try:
            pool = Pool(num_jobs)   # multi-processing
            all_data_list = pool.starmap(process_motion, job_args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        all_data = {}
        for data_dict in all_data_list:
            all_data.update(data_dict)
    if cfg.robot.input_motion_type == 'GVHMR':
        # data_key = list(all_data.keys())[0]
        # os.makedirs(f"output/Unitree_motion/{cfg.robot.humanoid_type}/v1/singles", exist_ok=True)
        dumped_file = f"output/Unitree_motion/{cfg.robot.humanoid_type}/v1/GVHMR_output.pkl"
        print(dumped_file)
        # vis_mujoco(all_data[key_names[0]],cfg.robot.humanoid_type)         # for visualization
        joblib.dump(all_data, dumped_file)

    else:
        os.makedirs(f"output/Unitree_motion/{cfg.robot.humanoid_type}/v1/", exist_ok=True)
        joblib.dump(all_data, f"output/Unitree_motion/{cfg.robot.humanoid_type}/v1/amass_{cfg.robot.process_split}.pkl")
    


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()

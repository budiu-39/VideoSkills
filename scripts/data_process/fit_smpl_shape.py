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
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser, 
)
import mujoco

import time
import joblib
import torch
import torch.nn.functional as F
import math
from smpl_sim.utils.pytorch3d_transforms import axis_angle_to_matrix
from torch.autograd import Variable
from phc.utils.torch_humanoid_batch import Humanoid_Batch
from easydict import EasyDict
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES

smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]

@hydra.main(version_base=None, config_path="../../phc/data/cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    
    humanoid_fk = Humanoid_Batch(cfg.robot) # load forward kinematics model

    #### Define corresonpdances between h1 and smpl joints
    robot_joint_names_augment = humanoid_fk.body_names_augment 
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
    robot_joint_pick_idx = [ robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    #### Preparing fitting variabels
    device = torch.device("cpu")
    pose_aa_robot = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], humanoid_fk.num_bodies , axis = 2), 1, axis = 1)
    pose_aa_robot = torch.from_numpy(pose_aa_robot).float()

    #  ==== SMPL_Humanoid skeletion offset test (compare with the offset of smpl in zero pose ==== #

    # smpl_humanoid_fk = Humanoid_Batch(cfg.test)  # load forward kinematics model
    #
    # smpl_humanoid_joint_names_augment = smpl_humanoid_fk.body_names_augment
    # smpl_humanoid_joint_pick = SMPL_MUJOCO_NAMES
    # SMPL_joint_pick = SMPL_BONE_ORDER_NAMES
    # robot_joint_pick_idx = [smpl_humanoid_joint_names_augment.index(j) for j in smpl_humanoid_joint_pick]
    # smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in SMPL_joint_pick]
    #
    # pose_aa_stand = np.random.rand(1, 72)
    # pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)
    #
    # pose_aa_robot = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], smpl_humanoid_fk.num_bodies , axis = 2), 1, axis = 1)
    # pose_aa_robot = torch.from_numpy(pose_aa_robot).float()
    #
    # pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))
    # smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    #
    # trans = torch.zeros([1, 3])
    # beta = torch.zeros([1, 10])
    # verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta , trans)
    #
    # offset = joints[:, 0] - trans
    # root_trans_offset = trans + offset
    # pose_aa_robot[0,0,smpl_humanoid_joint_names_augment.index("Pelvis")] =  torch.tensor((sRot.from_rotvec(
    #     pose_aa_robot[0,0,smpl_humanoid_joint_names_augment.index("Pelvis")]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5])
    #                                                                          .inv()).as_rotvec())
    #
    # test_return = smpl_humanoid_fk.fk_batch(pose_aa_robot[None,], root_trans_offset[None, 0:1])
    # smpl_humanoid_joint_offset = torch.zeros_like(test_return.global_translation.squeeze(0))
    # smpl_joint_offset = torch.zeros_like(joints)
    # for i in range(len(smpl_humanoid_fk._parents[1:])):
    #     smpl_humanoid_joint_offset[:, i] = test_return.global_translation[:, :, i] - test_return.global_translation[:, :, smpl_humanoid_fk._parents[i]]
    #     smpl_joint_offset[:, i] = joints[:, i] - joints[:, smpl_parser_n.parents.cpu().numpy()[i]]
    # smpl_joint_offset = smpl_joint_offset[:, smpl_2_mujoco]
    #
    # transformed_humanoid_joint_offset = torch.matmul(torch.tensor(sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv().as_matrix(), dtype=torch.float32), smpl_humanoid_joint_offset.squeeze(0).T).T

    # ==== SMPL_Humanoid skeletion offset test (compare with the offset of smpl in zero pose ==== #

    #### prepare SMPL default pause for H1

    pose_aa_stand = np.zeros((1, 72))
    pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)

    for modifiers in cfg.robot.smpl_pose_modifier:
        modifier_key = list(modifiers.keys())[0]
        modifier_value = list(modifiers.values())[0]
        pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index(modifier_key)] = sRot.from_euler("xyz", eval(modifier_value),  degrees = False).as_rotvec()

    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))
    smpl_parser_n = SMPL_Parser(model_path="phc/data/smpl", gender="neutral")

    ###### Shape fitting
    trans = torch.zeros([1, 3])
    beta = torch.zeros([1, 10])
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta , trans)

    offset = joints[:, 0] - trans
    root_trans_offset = trans + offset

    fk_return = humanoid_fk.fk_batch(pose_aa_robot[None, ], root_trans_offset[None, 0:1])


    # === Unitree offset test === #
    # humanoid_joint_offset = torch.zeros_like(fk_return.global_translation_extend.squeeze(0)[:,robot_joint_pick_idx])
    # smpl_joint_offset = torch.zeros_like(joints[:,smpl_joint_pick_idx])
    #
    # pick_parent_idx = torch.zeros_like(torch.tensor(robot_joint_pick_idx))
    # for i in range(1, len(pick_parent_idx)):
    #     parent = humanoid_fk._parents[robot_joint_pick_idx[i]]
    #     while parent not in robot_joint_pick_idx:
    #         parent = humanoid_fk._parents[parent]
    #     pick_parent_idx[i] = parent
    #
    # smpl_pick_parent_idx = torch.zeros_like(torch.tensor(smpl_joint_pick_idx))
    # for i in range(1, len(smpl_joint_pick_idx)):
    #     parent = smpl_parser_n.parents[smpl_joint_pick_idx[i]]
    #     while parent not in smpl_joint_pick_idx:
    #         parent = smpl_parser_n.parents[parent]
    #     smpl_pick_parent_idx[i] = parent
    #
    # for i,id in enumerate(robot_joint_pick_idx):
    #     humanoid_joint_offset[:, i] = fk_return.global_translation_extend[:, :, id] - fk_return.global_translation_extend[:, :, pick_parent_idx[i]]
    # for i,id in enumerate(smpl_joint_pick_idx):
    #     smpl_joint_offset[:, i] = joints[:, id] - joints[:, smpl_pick_parent_idx[i]]

    # === Unitree offset test === #


    shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)
    optimizer_shape = torch.optim.Adam([shape_new, scale],lr=0.1)
    
    train_iterations=3000
    print("start fitting shapes")
    pbar = tqdm(range(train_iterations))
    for iteration in pbar:
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1]) # fitted smpl shape

        root_pos = joints[:, 0]

        # ==== Visualization ==== #

        # View h1 in mujoco
        # xml_path = "/home/miku/Documents/PHC/phc/data/assets/robot/unitree_h1/h1.xml"
        #
        # mj_model = mujoco.MjModel.from_xml_path(xml_path)
        # mj_data = mujoco.MjData(mj_model)
        # mj_data.qpos[:3] = root_trans_offset[0] + np.array([0, 0, 1])
        # mj_data.qpos[3:7] = (sRot.from_rotvec(pose_aa_robot.reshape(-1, 3)[0]) * sRot.from_euler('XYZ', [3.14,0,0])).as_quat()[[3,0,1,2]]
        # mj_data.qpos[7:] = sRot.from_rotvec(pose_aa_robot.reshape(-1, 3)[1:]).magnitude()
        # (sRot.from_quat([0,0,0,1]).inv() * sRot.from_quat([1,0,0,0])).as_euler("XYZ").flatten()

        # # View smpl in mujoco   #
        xml_path = f"phc/data/assets/mjcf/smpl_humanoid.xml"

        # The reference coordinate systems of **SMPL** and **MuJoCo** are different. Therefore transformation are needed.
        # pose_aa_stand[0, 3 * 3:4 * 3] = torch.tensor(
        #     (sRot.from_rotvec(pose_aa_stand.reshape(-1, 3)[3]) * sRot.from_euler("XYZ", [1.57, 0, 0])).as_rotvec())

        # pose_quat = (sRot.from_quat(
        #         [0.5, 0.5, 0.5, 0.5]) * sRot.from_rotvec(
        #         pose_aa_stand.reshape(-1,3)[smpl_2_mujoco]) * sRot.from_quat(
        #         [0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(-1, 4)
        #
        # root_quat =(sRot.from_quat(
        #     [0.5, 0.5, 0.5, 0.5]).inv() * sRot.from_quat(pose_quat[0])).as_quat()
        # #
        # mj_model = mujoco.MjModel.from_xml_path(xml_path)
        # mj_data = mujoco.MjData(mj_model)
        # mj_data.qpos[:3] = root_trans_offset[0] + np.array([0, 0, 1])
        # mj_data.qpos[3:7] = sRot.from_rotvec(pose_aa_stand.reshape(-1,3)[smpl_2_mujoco]).as_quat()[0][[3, 0, 1, 2]]
        # mj_data.qpos[7:] = sRot.from_rotvec(pose_aa_stand.reshape(-1,3)[smpl_2_mujoco][1:]).as_euler(
        #     "XYZ").flatten()
        #
        # mj_data.qpos[3:7] = root_quat[[3, 0, 1, 2]]
        # mj_data.qpos[7:] = sRot.from_quat(pose_quat[1:]).as_euler(
        #     "XYZ").flatten()

        # # Launch the viewer
        # with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        #     while viewer.is_running():
        #         time.sleep(100)

        # ==== Visualization ===== #

        joints = (joints - joints[:, 0]) * scale + root_pos
        if len(cfg.robot.extend_config) > 0:
            diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            print(diff)
        else:
            diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]

        # robot_joints = fk_return.global_translation_extend[:, :, robot_joint_pick_idx]
        # smpl_joints = joints[:, smpl_joint_pick_idx]
        # smpl_humanoid_joints =
        # loss_g = diff.norm(dim = -1).mean()
        loss_g = diff.norm(dim = -1).square().sum()
        
        loss = loss_g
        pbar.set_description_str(f"{iteration} - Loss: {loss.item() * 1000}")

        optimizer_shape.zero_grad()
        loss.backward()
        optimizer_shape.step()

    # print the fitted shape and scale parameters
    print("shape:",shape_new.detach())
    print("scale:",scale)

    if cfg.get("vis", False):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        import matplotlib.pyplot as plt
        
        j3d = fk_return.global_translation_extend[0, :, robot_joint_pick_idx, :].detach().numpy()
        j3d = j3d - j3d[:, 0:1]
        j3d_joints = joints[:, smpl_joint_pick_idx].detach().numpy()
        j3d_joints = j3d_joints - j3d_joints[:, 0:1]
        idx = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(90, 0)
        ax.scatter(j3d[idx, :,0], j3d[idx, :,1], j3d[idx, :,2], label='Humanoid Shape', c='blue')
        ax.scatter(j3d_joints[idx, :,0], j3d_joints[idx, :,1], j3d_joints[idx, :,2], label='Fitted Shape', c='red')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        drange = 1
        ax.set_xlim(-drange, drange)
        ax.set_ylim(-drange, drange)
        ax.set_zlim(-drange, drange)
        ax.legend()
        plt.show()
        print(robot_joints = fk_return.global_translation_extend[:, :, robot_joint_pick_idx])
        print(smpl_joints = joints[:, smpl_joint_pick_idx])

    os.makedirs(f"phc/data/{cfg.robot.humanoid_type}", exist_ok=True)
    joblib.dump((shape_new.detach(), scale), f"phc/data/{cfg.robot.humanoid_type}/shape_optimized_v1.pkl") # V2 has hip joints


if __name__ == "__main__":
    main()

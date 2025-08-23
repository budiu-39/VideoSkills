import os
import sys

sys.path.append(os.getcwd())
from videoskills.utils.torch_humanoid_batch import Humanoid_Batch
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import joblib
from tqdm import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
import torch
from torch.autograd import Variable
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
from videoskills import LEGGED_GYM_ROOT_DIR
from videoskills.envs.g1.g1_config import G1RoughCfg

smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]

from dataclasses import dataclass
from typing import List

@dataclass
class ExtendCfgEntry:
    joint_name: str
    parent_name: str
    pos: List[float]
    rot: List[float]

# 用在初始化时

def main() -> None:
    robot_cfg = G1RoughCfg()
    robot_cfg.asset.file = robot_cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    retarget_cfg = robot_cfg.retarget
    retarget_cfg.file = robot_cfg.asset.file
    retarget_cfg.extend_config = [ExtendCfgEntry(**d) for d in retarget_cfg.extend_config]

    humanoid_fk = Humanoid_Batch(retarget_cfg)  # load forward kinematics model

    #### Define corresonpdances between h1 and smpl joints
    robot_joint_names_augment = humanoid_fk.body_names_augment
    # TODO：这里需要 先把骨架弄出来做个 match，注意是 joint match
    robot_joint_pick = [i[0] for i in retarget_cfg.joint_matches]
    smpl_joint_pick = [i[1] for i in retarget_cfg.joint_matches]
    robot_joint_pick_idx = [robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    #### Preparing fitting variabels
    device = torch.device("cpu")
    pose_aa_robot = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None,], humanoid_fk.num_bodies, axis=2),
                              1, axis=1)
    pose_aa_robot = torch.from_numpy(pose_aa_robot).float()

    pose_aa_stand = np.zeros((1, 72))
    pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)

    for key, value in retarget_cfg.smpl_pose_modifier.items():
        modifier_key = key
        modifier_value = value
        pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index(modifier_key)] = sRot.from_euler("xyz", modifier_value,
                                                                                      degrees=False).as_rotvec()
    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))
    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")

    ###### Shape fitting
    trans = torch.zeros([1, 3])
    beta = torch.zeros([1, 10])
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta, trans)

    offset = joints[:, 0] - trans
    root_trans_offset = trans + offset

    fk_return = humanoid_fk.fk_batch(pose_aa_robot[None,], root_trans_offset[None, 0:1])

    shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)
    optimizer_shape = torch.optim.Adam([shape_new, scale], lr=0.1)

    train_iterations = 3000
    print("start fitting shapes")
    pbar = tqdm(range(train_iterations))
    for iteration in pbar:
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])  # fitted smpl shape

        root_pos = joints[:, 0]

        # joints = (joints - joints[:, 0]) * scale + root_pos
        if len(retarget_cfg.extend_config) > 0:
            diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            # print(diff)
        else:
            diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]

        loss_g = diff.norm(dim=-1).square().sum()

        loss = loss_g
        pbar.set_description_str(f"{iteration} - Loss: {loss.item() * 1000}")

        optimizer_shape.zero_grad()
        loss.backward()
        optimizer_shape.step()

    # print the fitted shape and scale parameters
    print("shape:", shape_new.detach())
    print("scale:", scale)

    if retarget_cfg.vis:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        import matplotlib.pyplot as plt

        j3d = fk_return.global_translation_extend[0, :, robot_joint_pick_idx, :].detach().numpy()
        j3d = j3d - j3d[:, 0:1]
        j3d_joints = joints[:, smpl_joint_pick_idx].detach().numpy()
        j3d_joints = j3d_joints - j3d_joints[:, 0:1]
        idx = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=0, azim=0)
        ax.scatter(j3d[idx, :, 0], j3d[idx, :, 1], j3d[idx, :, 2], label='Humanoid Shape', c='blue')
        ax.scatter(j3d_joints[idx, :, 0], j3d_joints[idx, :, 1], j3d_joints[idx, :, 2], label='Fitted Shape', c='red')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        drange = 1
        ax.set_xlim(-drange, drange)
        ax.set_ylim(-drange, drange)
        ax.set_zlim(-drange, drange)
        ax.legend()
        plt.show()
        # print(robot_joints=fk_return.global_translation_extend[:, :, robot_joint_pick_idx])
        # print(smpl_joints=joints[:, smpl_joint_pick_idx])

    os.makedirs(f"data/retarget/{retarget_cfg.humanoid_type}", exist_ok=True)
    joblib.dump((shape_new.detach(), scale), f"data/retarget/{retarget_cfg.humanoid_type}/shape_optimized_v1.pkl")  # V2 has hip joints


if __name__ == "__main__":
    main()

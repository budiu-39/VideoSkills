from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_joint_names import SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from scipy.spatial.transform import Rotation as sRot
import torch

from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState

mujoco_2_smpl = [SMPL_MUJOCO_NAMES.index(q) for q in SMPL_BONE_ORDER_NAMES if q in SMPL_MUJOCO_NAMES]
mujoco_2_smplh = [SMPLH_MUJOCO_NAMES.index(q) for q in SMPLH_BONE_ORDER_NAMES if q in SMPLH_MUJOCO_NAMES]

# from global root frame body quat to smpl pose
def humanoid2smpl(body_quat, root_trans, skeleton_trees, is_smplh=False):

    offset = skeleton_trees.local_translation[0].cpu()
    transl = root_trans - offset
    pre_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
    N = body_quat.shape[0]
    pose_quat = (sRot.from_quat(body_quat.reshape(-1, 4).numpy()) * pre_rot).as_quat().reshape(N, -1, 4)
    new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees,
                                                                    torch.from_numpy(pose_quat),
                                                                    root_trans.cpu(), is_local=False)
    local_rot = new_sk_state.local_rotation
    pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(N, -1, 3)
    if is_smplh:
        pose_aa = torch.from_numpy(pose_aa[:, mujoco_2_smplh, :].reshape(N, -1))
    else:
        pose_aa = torch.from_numpy(pose_aa[:, mujoco_2_smpl, :].reshape(N, -1))

    return pose_aa, transl
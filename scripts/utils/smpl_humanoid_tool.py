from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_joint_names import SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from scipy.spatial.transform import Rotation as sRot
import torch


from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState

mujoco_2_smpl = [SMPL_MUJOCO_NAMES.index(q) for q in SMPL_BONE_ORDER_NAMES if q in SMPL_MUJOCO_NAMES]
mujoco_2_smplh = [SMPLH_MUJOCO_NAMES.index(q) for q in SMPLH_BONE_ORDER_NAMES if q in SMPLH_MUJOCO_NAMES]

@torch.jit.script
def my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

# from global root frame body quat to smpl pose
def humanoid2smpl(body_quat, root_trans, skeleton_trees, is_smplh=False):
    # 这里只是利用相对关系，如果利用 skeletonstate, pos 是错的，其实改成全局旋转更加合适

    # new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees,
    #                                                                 body_quat,
    #                                                                 root_trans.cpu(), is_local=False)
    # joint_pos = new_sk_state.global_translation
    # # 需要做与 global_rot 相同的坐标系旋转
    # pre_rot_quat = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device=joint_pos.device)
    #
    # q_rep = pre_rot_quat.expand(joint_pos.shape[0], joint_pos.shape[1], 4)
    # joint_pos = my_quat_rotate(q_rep.reshape(-1, 4), joint_pos.reshape(-1, 3)).reshape_as(joint_pos)

    offset = skeleton_trees.local_translation[0].cpu()
    pre_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
    transl = root_trans - offset
    N = body_quat.shape[0]
    pose_quat = (sRot.from_quat(body_quat.reshape(-1, 4).numpy()) * pre_rot).as_quat().reshape(N, -1, 4)
    new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees,
                                                                    torch.from_numpy(pose_quat),
                                                                    root_trans.cpu(), is_local=False)
    local_rot = new_sk_state.local_rotation
    pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(N, -1, 3)
    if is_smplh:
        pose_aa = torch.from_numpy(pose_aa[:, mujoco_2_smplh, :].reshape(N, -1)).float()
    else:
        pose_aa = torch.from_numpy(pose_aa[:, mujoco_2_smpl, :].reshape(N, -1)).float()
        # joint_pos = joint_pos[:, mujoco_2_smpl, :]

    return pose_aa, transl  # , joint_pos
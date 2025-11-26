from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
from scripts.utils.face_z_align_util import *
import copy

import torch

def yup2zup(trans, rotvec):
    """
    rotvec: (N,3) or (3,)
    trans:  (N,3) or (3,)
    """
    mat = torch.tensor([[1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]], dtype=rotvec.dtype, device=rotvec.device)  # (3,3)
    # (1) rotvec → rotation matrix
    R_obj = axis_angle_to_matrix(rotvec)     # (N,3,3)

    # (2) R_new = R_ws * R_obj
    R_new = mat @ R_obj                     # broadcasting works

    # (3) back to rotvec
    rotvec_new = matrix_to_axis_angle(R_new)

    # (4) trans_new = R_ws @ trans
    trans_new = (mat @ trans.unsqueeze(-1)).squeeze(-1)

    return trans_new.float(), rotvec_new.float()

def smpl_to_272d(trans, pose_aa, beta, smpl_parser_n):
    seq_len = pose_aa.shape[0]
    root_first_frame_root_orient = pose_aa[0, 0]
    root_first_frame_root_orient_quat = expmap_to_quaternion(root_first_frame_root_orient)
    root_first_frame_root_orient_quat_xyzw = root_first_frame_root_orient_quat[[1, 2, 3, 0]]
    root_first_frame_root_orient_quat_xyzw = torch.from_numpy(
        root_first_frame_root_orient_quat_xyzw).float().unsqueeze(0)
    heading_inv, axis = calc_heading_quat_inv(root_first_frame_root_orient_quat_xyzw)
    heading_inv_axis_angle = heading_inv * axis
    heading_inv_axis_angle = heading_inv_axis_angle.numpy()
    q_diff = expmap_to_quaternion(heading_inv_axis_angle)
    result_root_orient_quaternion = qmul_np(q_diff.reshape(1, -1).repeat(seq_len, axis=0),
                                            expmap_to_quaternion(pose_aa[:, 0]))
    result_root_orient_axis_angle = quaternion_to_axis_angle(
        torch.from_numpy(result_root_orient_quaternion)).numpy()

    trans = qrot_np(q_diff.reshape(1, -1).repeat(seq_len, axis=0), trans)
    result_pose_body = np.concatenate(
        [result_root_orient_axis_angle, pose_aa[:, 1:].reshape(seq_len, -1), trans, beta], axis=-1)

    # smpl_face z to smpl joint
    data = torch.from_numpy(result_pose_body).float().cuda()


    joints = smpl_parser_n(body_pose=data[:, 3:66], betas=data[:, 75:],
                           transl=data[:, 72:75], global_orient=data[:, :3]).joints

    position_data = joints[:, :22, :3].cpu().numpy()
    nfrm, njoint, _ = position_data.shape

    # smpl and joint to 272 representation
    root_idx = 0
    rotation_smpl_axis_angle = result_pose_body
    rotations_wxyz = expmap_to_quaternion(rotation_smpl_axis_angle[:, :66].reshape(nfrm, njoint, 3))

    rotations_matrix = quaternion_to_matrix_np(rotations_wxyz)  # nframe, njoint, 3, 3

    # put on floor and put root on origin for the first frame
    ori = copy.deepcopy(position_data[0, root_idx])  # first frame root position
    y_min = np.min(position_data[:, :, 1])
    ori[1] = y_min
    position_data = position_data - ori
    velocities_root = position_data[1:, root_idx, :] - position_data[:-1, root_idx, :]

    # calculate local position, all frames on xz origin
    position_data[:, :, 0] -= position_data[:, 0:1, 0]
    position_data[:, :, 2] -= position_data[:, 0:1, 2]

    # calculate heading
    global_heading = - np.arctan2(rotations_matrix[:, root_idx, 0, 2], rotations_matrix[:, root_idx, 2, 2])
    global_heading_rot = np.array([rot_yaw(x) for x in global_heading])
    global_heading_diff = global_heading[1:] - global_heading[:-1]
    global_heading_diff_rot = np.array([rot_yaw(x) for x in global_heading_diff])

    # calculate positions no heading
    positions_no_heading = np.matmul(np.repeat(global_heading_rot[:, None, :, :], njoint, axis=1),
                                     position_data[..., None]).squeeze(-1)

    # calculate velocity no heading
    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]

    # calculate root velocity_xz_no_heading
    velocities_root_xy_no_heading = np.matmul(global_heading_rot[:-1], velocities_root[:, :, None]).squeeze()[
        ..., [0, 2]]

    # calculate rotations no heading
    rotations_matrix[:, 0, ...] = np.matmul(global_heading_rot, rotations_matrix[:, 0, ...])

    # concat all
    size_frame = 8 + njoint * 3 + njoint * 3 + njoint * 6
    final_x = np.zeros((nfrm, size_frame))

    # set the first frame of the root rotation to identity
    final_x[0, 2] = 1
    final_x[0, 6] = 1
    final_x[1:, 2:8] = matrix_to_rotation_6d(torch.from_numpy(global_heading_diff_rot)).numpy()  # take 6D rotation
    final_x[1:, :2] = velocities_root_xy_no_heading
    final_x[:, 8:8 + 3 * njoint] = np.reshape(positions_no_heading, (nfrm, -1))
    final_x[1:, 8 + 3 * njoint:8 + 6 * njoint] = np.reshape(velocities_no_heading, (nfrm - 1, -1))
    final_x[:, 8 + 6 * njoint:8 + 12 * njoint] = np.reshape(rotations_matrix[..., :, :2, :],
                                                            (nfrm, -1))  # take 6D rotation
    return final_x
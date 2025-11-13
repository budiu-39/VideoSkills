'''
This file is used to transform all smpl to z+ direction.
'''

from scripts.preprocess.face_z_align_util import *
import os
import torch
import copy
from tqdm import tqdm
import argparse
import glob
from smplx import SMPL
from scipy import ndimage
from scripts.preprocess.body_model_smplx import BodyModelSMPLX
import numpy as np
import joblib

def compute_canonical_transform(global_orient):
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=global_orient.dtype)
    global_orient_matrix = axis_angle_to_matrix(global_orient)
    global_orient_matrix = torch.matmul(rotation_matrix, global_orient_matrix)
    global_orient = matrix_to_axis_angle(global_orient_matrix)
    return global_orient

def transform_translation(trans):
    trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
    trans[:, 2] = trans[:, 2] * (-1)
    return trans


def process_pose(pose):
    pose_root = pose[:, :3]
    pose_root = compute_canonical_transform(torch.from_numpy(pose_root)).detach().cpu().numpy()
    pose[:, :3] = pose_root
    pose_trans = pose[:, 24*3:24*3+3]
    pose_trans = transform_translation(pose_trans)
    pose[:, 24*3:24*3+3] = pose_trans

    # return pose
    return np.float32(pose)

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path

def rot_yaw(yaw):
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    return np.array([[cs,0,sn],[0,1,0],[-sn,0,cs]])


def my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

    
def calc_heading(q):
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 2] = 1
    rot_dir = my_quat_rotate(q, ref_dir)
    heading = torch.atan2(rot_dir[..., 0], rot_dir[..., 2])
    return heading


def calc_heading_quat_inv(q):
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 1] = 1
    return -heading, axis

def _collect_amass_keys(amass_root: str):
    all_npzs = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.rstrip("/").split("/"))
    return {"0-" + "_".join(p.split("/")[split_len:]).replace(".npz", ""): p for p in all_npzs}

def foot_detect(global_positions, thres):
    """
        derived from https://github.com/orangeduck/Motion-Matching/blob/37df18afc44e8acca3af5e85dff96effa6a34b03/resources/generate_database.py#L160
    """
    left_foot = 10
    right_foot = 11
    global_velocities = global_positions[1:] - global_positions[:-1]
    contact_velocities = np.sqrt(np.sum(global_velocities[:, np.array([left_foot, right_foot])]**2, axis=-1))
    contacts = contact_velocities < thres
    # Median filter here acts as a kind of "majority vote", and removes
    # small regions  where contact is either active or inactive
    for ci in range(contacts.shape[1]):
        contacts[:,ci] = ndimage.median_filter(
            contacts[:,ci],
            size=6,
            mode='nearest')
    return contacts



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--filedir', type=str, required=True, help='Input directory path')
    args = parser.parse_args()

    amass_root = args.filedir
    all_npzs = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    amass_occlusion = joblib.load("data/amass_copycat_occlusion_v3.pkl")

    smpl_parser_n = BodyModelSMPLX(model_path='/mnt/lustre/work/ponsmoll/pba936/VideoSkills/data/SMPL', model_type= 'smplx')
    smpl_parser_n.cuda()
    smpl_parser_n.eval()
    bad_cnt = 0
    for data_path in tqdm(all_npzs):
        npz_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
        if not 'mocap_framerate' in npz_data:
            continue

        if 'trans' not in npz_data.keys():
            bad_cnt += 1
            continue
        bound = 0
        framerate = npz_data['mocap_framerate']

        skip = int(framerate / 30)
        root_trans = npz_data['trans'][::skip, :]
        body_pos = npz_data['poses'][::skip, :]
        rel = os.path.relpath(data_path, amass_root)
        key_name = rel.split("/")
        key_name_dump = "0-" + "_".join(key_name).replace(".npz", "")
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

        if "0-KIT_442_PizzaDelivery02_poses" == key_name_dump:
            bound = -2

        if bound == 0:
            bound = root_trans.shape[0]

        root_trans = root_trans[:bound]
        body_pos = body_pos[:bound]

        smpl_data = np.concatenate([body_pos[..., :66], np.zeros((root_trans.shape[0], 6)),
                                           root_trans, np.zeros((root_trans.shape[0], 10))], axis=-1)

        smpl_data = process_pose(smpl_data)
        smpl_data = smpl_data
        seq_len = smpl_data.shape[0]

        if seq_len > 5000:
            with open("discarded_sequences.txt", "a") as f:
                f.write(f"{data_path}\tseq_len={seq_len}\n")
            bad_cnt += 1
            continue

        if seq_len > 0:
            pose_body = smpl_data[:, :72].reshape(seq_len, -1, 3)
        else:  
            bad_cnt += 1
            continue

        # smpl to smpl face_z
        trans= smpl_data[:, 72:75]
        beta = smpl_data[:, 75:]
        root_first_frame_root_orient = pose_body[0,0]
        root_first_frame_root_orient_quat = expmap_to_quaternion(root_first_frame_root_orient)
        root_first_frame_root_orient_quat_xyzw = root_first_frame_root_orient_quat[[1, 2, 3, 0]]
        root_first_frame_root_orient_quat_xyzw = torch.from_numpy(root_first_frame_root_orient_quat_xyzw).float().unsqueeze(0)
        heading_inv, axis = calc_heading_quat_inv(root_first_frame_root_orient_quat_xyzw)
        heading_inv_axis_angle = heading_inv * axis
        heading_inv_axis_angle = heading_inv_axis_angle.numpy()
        q_diff = expmap_to_quaternion(heading_inv_axis_angle)
        result_root_orient_quaternion = qmul_np(q_diff.reshape(1, -1).repeat(seq_len, axis=0), expmap_to_quaternion(pose_body[:,0]))
        result_root_orient_axis_angle = quaternion_to_axis_angle(torch.from_numpy(result_root_orient_quaternion)).numpy()

        trans = qrot_np(q_diff.reshape(1, -1).repeat(seq_len, axis=0), trans)
        result_pose_body = np.concatenate([result_root_orient_axis_angle, pose_body[:,1:].reshape(seq_len, -1), trans, beta], axis=-1)

        # smpl_face z to smpl joint
        data = torch.from_numpy(result_pose_body).float().cuda()
        joints = smpl_parser_n(body_pose = data[:, 3:66], betas = data[:, 75:],
                                          transl = data[:, 72:75], global_orient = data[:, :3]).joints

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

        # smpl unit is m and 0.15 is given as cm, may need to change depending on the datasets
        contacts = foot_detect(position_data, 0.15 / 100)

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
        try:
            final_x[1:, 2:8] = matrix_to_rotation_6d(
                torch.from_numpy(global_heading_diff_rot)).numpy()  # take 6D rotation
        except:
            bad_cnt += 1
            continue
        final_x[1:, :2] = velocities_root_xy_no_heading
        final_x[:, 8:8 + 3 * njoint] = np.reshape(positions_no_heading, (nfrm, -1))
        final_x[1:, 8 + 3 * njoint:8 + 6 * njoint] = np.reshape(velocities_no_heading, (nfrm - 1, -1))
        final_x[:, 8 + 6 * njoint:8 + 12 * njoint] = np.reshape(rotations_matrix[..., :, :2, :],
                                                                (nfrm, -1))  # take 6D rotation

        # TODO: 要改的！
        output_path = data_path.replace(amass_root, amass_root + '_272')
        save_path = output_path.replace(".npz", ".npy")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(save_path, final_x)
    print(f"bad_cnt: {bad_cnt}")
    print(f"Processed files are saved in {amass_root}_272")
    



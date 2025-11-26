'''
This file is used to transform all smpl to z+ direction.
'''

from scripts.utils.face_z_align_util import *
import os
from tqdm import tqdm
import argparse
import glob
from scipy import ndimage
from scripts.utils.body_model_smplx import BodyModelSMPLX
import json
import os.path as osp
from scipy.spatial.transform import Rotation as sRot
from scripts.rep_272.d272_to_smpl import pose_272_to_smpl
from scripts.render.smpl_render_utils import smpl_visualize_and_render
from scripts.rep_272.smpl_to_d272 import smpl_to_272d


def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


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
    contact_velocities = np.sqrt(np.sum(global_velocities[:, np.array([left_foot, right_foot])] ** 2, axis=-1))
    contacts = contact_velocities < thres
    # Median filter here acts as a kind of "majority vote", and removes
    # small regions  where contact is either active or inactive
    for ci in range(contacts.shape[1]):
        contacts[:, ci] = ndimage.median_filter(
            contacts[:, ci],
            size=6,
            mode='nearest')
    return contacts

def apply_cam2world_rotvec_trans(rotvec, trans, R3x3):
    r_new = sRot.from_matrix(R3x3) * sRot.from_rotvec(rotvec)
    t_new = (R3x3 @ trans.T).T
    return r_new.as_rotvec().astype(np.float32), t_new.astype(np.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--filedir', type=str, required=True, help='Input directory path')
    args = parser.parse_args()

    smpl_parser_n = BodyModelSMPLX(model_path='data/SMPL', model_type= 'smplx')
    smpl_parser_n.cuda()
    smpl_parser_n.eval()
    bad_cnt = 0
    visualize = False

    folder_path = "dataset/motionx++/kungfu"
    # folder_path = "dataset/humanml3d"
    output_dir = "dataset/smpl_motion/MotionX++/kungfu"
    # npy_files = sorted(glob.glob(os.path.join(folder_path,'*.npy')))
    npy_files = sorted(glob.glob(os.path.join(folder_path, '*.json'), recursive=True))
    render_out_dir = "output/render_out/MotionX++_kungfu"

    for fpath in tqdm(npy_files):
        with open(fpath, 'r') as f:
            motion_data = json.load(f)
        rel_path = os.path.relpath(fpath, folder_path)

        save_path = osp.join(output_dir, rel_path).replace(".json", ".npy")
        # motion = np.load(fpath, allow_pickle=True)

        # 加入降采样代码
        root_trans = np.array([np.array(i['smplx_params']['trans'], dtype=np.float32)
                               for i in motion_data['annotations']])

        pose_aa = np.array([np.concatenate([np.array(i['smplx_params']['root_orient'], dtype=np.float32),
                                            np.array(i['smplx_params']['pose_body'], dtype=np.float32)])
                            for i in motion_data['annotations']])

        seq_len = pose_aa.shape[0]

        if seq_len > 5000:
            # with open("discarded_sequences.txt", "a") as f:
            #     f.write(f"{data_path}\tseq_len={seq_len}\n")
            bad_cnt += 1
            continue

        if seq_len > 0:
            pose_body = pose_aa[:, :72].reshape(seq_len, -1, 3)
        else:
            bad_cnt += 1
            continue

        pose_aa_smpl = np.zeros((seq_len, 24, 3), dtype=pose_aa.dtype)
        pose_aa_smpl[:, :22, :] = pose_aa.reshape(seq_len, 22, 3)

        # x272 = smpl_to_272d(root_trans, pose_aa_smpl, np.zeros((seq_len, 10)), smpl_parser_n)

        R_cam2world = [[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]
        pose_aa_smpl[:, 0], root_trans = apply_cam2world_rotvec_trans(pose_aa_smpl[:, 0], root_trans, R_cam2world)

        smpl_data = np.concatenate([pose_aa_smpl.reshape(seq_len, -1)[..., :66], np.zeros((root_trans.shape[0], 6)),
                                           root_trans, np.zeros((root_trans.shape[0], 10))], axis=-1)

        x272 = smpl_to_272d(smpl_data[:, 72:75], smpl_data[:, :72].reshape(seq_len, -1, 3), smpl_data[:, 75:], smpl_parser_n)



        if visualize:
            pose_aa = np.zeros((seq_len, 24, 3))
            trans, pose_aa[:, :22] = pose_272_to_smpl(x272)
            os.makedirs(render_out_dir, exist_ok=True)
            out_path = os.path.join(render_out_dir, os.path.basename(fpath).replace(".json", ".avi"))
            smpl_visualize_and_render(trans, pose_aa, out_path)

        output_path = fpath.replace('motionx++', 'motionx++' + '_272')
        save_path = output_path.replace(".json", ".npy")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(save_path, seq_len)
    print(f"bad_cnt: {bad_cnt}")
    print(f"Processed files are saved in motionx++_272")




'''
This file is used to transform all smpl to z+ direction.
'''

from scripts.utils.face_z_align_util import *
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
import os
import torch
import copy
from tqdm import tqdm
import argparse
import glob
from smplx import SMPL
from scipy import ndimage
from scripts.utils.body_model_smplx import BodyModelSMPLX
import numpy as np
import joblib
from scripts.rep_272.smpl_to_d272 import smpl_to_272d, yup2zup
from scripts.render.smpl_render_utils import smpl_visualize_and_render
from scripts.rep_272.d272_to_smpl import pose_272_to_smpl


def _collect_amass_keys(amass_root: str):
    all_npzs = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.rstrip("/").split("/"))
    return {"0-" + "_".join(p.split("/")[split_len:]).replace(".npz", ""): p for p in all_npzs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--filedir', type=str, required=True, help='Input directory path')
    args = parser.parse_args()
    visualize = False
    render_out_dir = "output/render_out/AMASS"

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
        seq_len = len(body_pos)

        root_trans, body_pos[:, :3] = yup2zup(torch.tensor(root_trans), torch.tensor(body_pos[:, :3]))

        smpl_data = np.concatenate([body_pos[..., :66], np.zeros((root_trans.shape[0], 6)),
                                           root_trans, np.zeros((root_trans.shape[0], 10))], axis=-1)

        x272 = smpl_to_272d(smpl_data[:, 72:75], smpl_data[:, :72].reshape(seq_len, -1, 3), smpl_data[:, 75:], smpl_parser_n)

        smpl_data = smpl_data

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

        if visualize:
            pose_aa = np.zeros((seq_len, 24, 3))
            trans, pose_aa[:, :22] = pose_272_to_smpl(x272)
            os.makedirs(render_out_dir, exist_ok=True)
            out_path = os.path.join(render_out_dir, key_name_dump + ".avi")
            smpl_visualize_and_render(trans, pose_aa, out_path)


        output_path = data_path.replace(amass_root, amass_root + '_272')
        save_path = output_path.replace(".npz", ".npy")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(save_path, x272)
    print(f"bad_cnt: {bad_cnt}")
    print(f"Processed files are saved in {amass_root}_272")
    



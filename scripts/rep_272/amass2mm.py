'''
This file is used to transform all smpl to z+ direction.
'''

import os
import torch
from tqdm import tqdm
import argparse
import glob
from scripts.utils.body_model_smplx import BodyModelSMPLX
import numpy as np
import joblib
from scripts.rep_272.smpl_to_d272 import smpl_to_272d, yup2zup
from scripts.render.smpl_render_utils import smpl_visualize_and_render
from scripts.rep_272.d272_to_smpl import pose_272_to_smpl
from scripts.rep_272.recover_visualize import recover_from_local_position
from scripts.rep_272.plot_3d_global import draw_to_batch


def _collect_amass_keys(amass_root: str):
    all_npzs = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.rstrip("/").split("/"))
    return {"0-" + "_".join(p.split("/")[split_len:]).replace(".npz", ""): p for p in all_npzs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--filedir', type=str, required=True, help='Input directory path')
    parser.add_argument('--keylist', type=str, default=None,
                        help='Path to a txt file containing key_name_dump, one per line')
    args = parser.parse_args()
    visualize = False
    render_out_dir = "output/render_out/AMASS"
    out_dir = "dataset/272_rep/AMASS_272"
    os.makedirs(out_dir, exist_ok=True)

    amass_root = args.filedir
    all_npzs = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    amass_occlusion = joblib.load("data/amass_copycat_occlusion_v3.pkl")

    allowed_keys = None
    if args.keylist is not None:
        with open(args.keylist, "r") as f:
            allowed_keys = {line.strip() for line in f if line.strip()}



    smpl_parser_n = BodyModelSMPLX(model_path='data/SMPL', model_type= 'smplx')
    smpl_parser_n.cuda()
    smpl_parser_n.eval()
    bad_cnt = 0
    # START = 9780
    for data_path in tqdm(all_npzs):
        # if idx < START:
        #     continue
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
        key_name_dump = "-".join(key_name).replace(".npz", "")
        out_path = os.path.join(out_dir, key_name_dump + ".npy")
        if allowed_keys is not None and key_name_dump not in allowed_keys:
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

        if "0-KIT_442_PizzaDelivery02_poses" == key_name_dump:
            bound = -2

        if bound == 0:
            bound = root_trans.shape[0]


        root_trans = root_trans[:bound]
        body_pos = body_pos[:bound]
        N = body_pos.shape[0]
        if N < 10:
            continue
        seq_len = len(body_pos)
        # print(body_pos.shape)

        # if seq_len < 10:
        #     bad_cnt += 1
        #     continue

        if seq_len > 5000:
            # with open("discarded_sequences.txt", "a") as f:
            #     f.write(f"{data_path}\tseq_len={seq_len}\n")
            bad_cnt += 1
            continue

        root_trans, body_pos[:, :3] = yup2zup(torch.tensor(root_trans), torch.tensor(body_pos[:, :3]))

        smpl_data = np.concatenate([body_pos[..., :66], np.zeros((root_trans.shape[0], 6)),
                                           root_trans, np.zeros((root_trans.shape[0], 10))], axis=-1)

        x272 = smpl_to_272d(smpl_data[:, 72:75], smpl_data[:, :72].reshape(seq_len, -1, 3), smpl_data[:, 75:], smpl_parser_n)

        smpl_data = smpl_data



        # if seq_len > 0:
        #     pose_body = smpl_data[:, :72].reshape(seq_len, -1, 3)
        # else:
        #     bad_cnt += 1
        #     continue

        # if visualize:
        #     pose_aa = np.zeros((seq_len, 24, 3))
        #     trans, pose_aa[:, :22] = pose_272_to_smpl(x272)
        #
        #     out_path = os.path.join(render_out_dir, key_name_dump + ".avi")
        #     smpl_visualize_and_render(trans, pose_aa, out_path)

        if visualize:
            pred_xyz = recover_from_local_position(x272, 22)
            xyz = pred_xyz.reshape(1, -1, 22, 3)
            os.makedirs(render_out_dir, exist_ok=True)
            pose_vis = draw_to_batch(xyz, outname = [f'{render_out_dir}/{key_name_dump}.mp4'], fps=30)

        # output_path = data_path.replace(amass_root, amass_root + '_272')
        # save_path = output_path.replace(".npz", ".npy")
        x272 = x272.astype(np.float32)
        np.save(out_path, x272)
    print(f"bad_cnt: {bad_cnt}")
    print(f"Processed files are saved in {amass_root}_272")
    



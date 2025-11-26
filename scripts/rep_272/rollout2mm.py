'''
This file is used to transform all smpl to z+ direction.
'''

from scripts.utils.face_z_align_util import *
import os
import torch
from tqdm import tqdm
import glob
from scripts.utils.body_model_smplx import BodyModelSMPLX
from scripts.poselib.skeleton.skeleton3d import SkeletonTree
import joblib
from scripts.utils.smpl_humanoid_tool import humanoid2smpl
from scripts.rep_272.smpl_to_d272 import smpl_to_272d, yup2zup
from scripts.rep_272.d272_to_smpl import pose_272_to_smpl
from scripts.render.smpl_render_utils import smpl_visualize_and_render


def rot_yaw_z(yaw):
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    return np.array([[cs, -sn, 0],
                     [sn,  cs, 0],
                     [ 0,   0, 1]], dtype=np.float32)


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


if __name__ == '__main__':
    folder_path = "/logs/smpl_ppo/amass_rollout/refine_results/succeed"
    # folder_path = "dataset/humanml3d"
    output_dir = "/logs/smpl_ppo/amass_rollout/292_w_action"
    # npy_files = sorted(glob.glob(os.path.join(folder_path,'*.npy')))
    pkl_files = sorted(glob.glob(os.path.join(folder_path, '*.pkl'), recursive=True))
    os.makedirs(output_dir, exist_ok=True)
    bad_cnt = 0
    visualize = False
    smpl_parser_n = BodyModelSMPLX(model_path='data/SMPL',
                                   model_type='smplx')
    skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/smpl_humanoid.xml")

    for fpath in tqdm(pkl_files):
        motion_data = joblib.load(fpath)
        position_data = motion_data['pred_pos'][:-1]
        pred_rot = motion_data['pred_rot'][:-1]

        nfrm, njoint, _ = position_data.shape
        root_idx = 0

        if nfrm > 5000:
            bad_cnt += 1
            continue

        # ================ 检测最后一步是否有 RESET ==================

        # root_pos = position_data[:, root_idx]  # (T, 3)
        # root_pos_diff = np.linalg.norm(root_pos[1:] - root_pos[:-1], axis=-1)  # (T-1,)
        #
        # # 计算 root 在相邻帧间的旋转差（用四元数）
        # root_quat = pred_rot[:, root_idx]  # (T, 4)  这里假设格式是 [x, y, z, w]
        # rot_all = R.from_quat(root_quat)   # (T,)
        # rel_rot = rot_all[1:] * rot_all[:-1].inv()   # 相对旋转 (T-1,)
        # rel_angle = rel_rot.magnitude()             # 每一步的旋转角度（弧度） (T-1,)
        #
        # # 最后一步的跳变
        # last_pos_jump = root_pos_diff[-1]
        # last_rot_jump_deg = np.degrees(rel_angle[-1])
        #
        # print(f"[{os.path.basename(fpath)}] last step: "
        #       f"Δpos={last_pos_jump:.3f} m, Δrot={last_rot_jump_deg:.1f}°")
        #
        # # 简单阈值判断（可以根据你的数据再调）
        # pos_thr = 2.0       # 2 米以上认为可疑
        # rot_thr_deg = 90.0  # 90 度以上认为可疑
        #
        # last_is_reset = (last_pos_jump > pos_thr) or (last_rot_jump_deg > rot_thr_deg)
        # if last_is_reset:
        #     print(f"  -> detected possible RESET at last frame (index {nfrm-1})")

        # 先从 SMPL robot 转成 SMPL

        pred_rot_np = torch.from_numpy(pred_rot)
        pos_np =  torch.from_numpy(position_data[:, 0])

        pose_aa, transl = humanoid2smpl(pred_rot_np, pos_np, skeleton_tree, is_smplh=False)

        # gif_outdir = os.path.dirname(fpath).replace("refine_results", "272_plot")
        # os.makedirs(gif_outdir, exist_ok=True)
        # gif_outpath = os.path.join(gif_outdir, os.path.basename(fpath).replace(".pkl", ".gif"))
        # draw_to_batch(joint_pos.unsqueeze(0).numpy(), outname = [gif_outpath])
        # 再由 SMPL 转换到 272 维表示
        smpl_parser_n.cuda()
        smpl_parser_n.eval()
        beta = np.zeros((nfrm, 10), dtype=np.float32)

        transl, pose_aa[:, :3] = yup2zup(transl, pose_aa[:, :3])

        smpl_data = np.concatenate([pose_aa.reshape(nfrm, -1)[..., :66], np.zeros((transl.shape[0], 6)),
                                           transl, np.zeros((transl.shape[0], 10))], axis=-1)

        x272 = smpl_to_272d(smpl_data[:, 72:75], smpl_data[:, :72].reshape(nfrm, -1, 3), smpl_data[:, 75:], smpl_parser_n)

        # 加上运动，记得错位（就是把动作的第一帧丢弃，然后整体往前移动一帧，含义是在当前 state 下执行的动作）
        action = motion_data['action'][1:]
        x272_action = np.concatenate([x272, action], axis=-1)  # (T, 272 + action_dim)


        if visualize:
            pose_aa = np.zeros((nfrm, 24, 3))
            trans, pose_aa[:, :22] = pose_272_to_smpl(x272_action[:,:272])
            out_path = os.path.dirname(fpath).replace("refine_results", "smpl_rendered_video")
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, os.path.basename(fpath).replace(".pkl", ".avi"))
            smpl_visualize_and_render(trans, pose_aa, out_path)

        file_name = os.path.basename(fpath)
        save_path = file_name.replace(".pkl", ".npy")
        output_path = os.path.join(output_dir, save_path)
        np.save(output_path, x272_action)
    print(f"bad_cnt: {bad_cnt}")
    print(f"Processed files are saved in 292 dim representation.")




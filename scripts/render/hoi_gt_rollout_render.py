import shutil
import smplx
import joblib
import pytorch3d.transforms as transforms
import os
import sys
import os.path as osp
import subprocess
import glob
import trimesh
import torch
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm
import argparse

sys.path.append(os.getcwd())

# 导入你的项目模块
from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPLH_MUJOCO_NAMES, SMPLX_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_parser import SMPLX_Parser

from scripts.smpl2sim.hoi.mujoco_contact_inference import (
    build_local_templates_by_body, prepare_batch_body_cloud,
    get_aligned_vhacd_hulls, compute_hull_planes,
    quick_viz_ig_cg, build_sk2mj_index,
    quick_viz_penetration_vhacd, penetration_depth_vhacd_cuda,
    contacts_from_xml_pointcloud,
    build_qpos_seq_from_state,
    debug_viz_vhacd_cuda_step
)
from scripts.smpl2sim.hoi.omomo_utils import rotate_at_frame_w_obj, get_smpl_parents
from scripts.render.mujoco_render import export_mujoco_video_hoi, create_temp_xml_with_object
from scripts.render.hoi_render import render_smpl_hoi_video_yup
from scripts.smpl2sim.hoi.smpl2sim_utils import from_yup_to_simulation, tranfrom_to_yup, from_retarget_to_simulation


def get_omomo_data(raw_p_path):
    # print(f"Loading OMOMO data from {raw_p_path}...")
    return joblib.load(raw_p_path)


def omomo_preprocess(seq_data):
    seq_name = seq_data['seq_name']
    object_name = seq_name.split("_")[1]
    trans2joint = seq_data['trans2joint']
    rest_human_offsets = seq_data['rest_offsets']
    betas = seq_data['betas'][0]
    gender = seq_data['gender']
    trans = seq_data['trans']
    frame_times = len(trans)
    global_orient = seq_data['root_orient']
    body_pose = seq_data['pose_body'].reshape(-1, 21, 3)
    obj_trans = seq_data['obj_trans'][:, :, 0]
    obj_rot = seq_data['obj_rot']
    obj_com_pos = seq_data['obj_com_pos']
    padding_zeros_hand = np.zeros((frame_times, 90))

    joint_aa_rep = torch.cat((torch.from_numpy(global_orient).float()[:, None, :], \
                              torch.from_numpy(body_pose).float()), dim=1)
    X = torch.from_numpy(rest_human_offsets).float()[None].repeat(joint_aa_rep.shape[0], 1, 1).detach().cpu().numpy()
    X[:, 0, :] = trans
    local_rot_mat = transforms.axis_angle_to_matrix(joint_aa_rep)
    Q = transforms.matrix_to_quaternion(local_rot_mat).detach().cpu().numpy()

    obj_x = obj_trans.copy()
    obj_rot_mat = torch.from_numpy(obj_rot).float()
    obj_q = transforms.matrix_to_quaternion(obj_rot_mat).detach().cpu().numpy()
    parents = get_smpl_parents()
    _, _, new_obj_x, new_obj_q = rotate_at_frame_w_obj(X[np.newaxis], Q[np.newaxis], \
                                                       obj_x[np.newaxis], obj_q[np.newaxis], \
                                                       trans2joint[np.newaxis], parents, n_past=1, floor_z=True)

    X, Q, new_obj_com_pos, _ = rotate_at_frame_w_obj(X[np.newaxis], Q[np.newaxis], \
                                                     obj_com_pos[np.newaxis], obj_q[np.newaxis], \
                                                     trans2joint[np.newaxis], parents, n_past=1, floor_z=True)

    new_seq_root_trans = X[0, :, 0, :]
    new_local_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(Q[0]).float())
    new_local_aa_rep = transforms.matrix_to_axis_angle(new_local_rot_mat)
    new_seq_root_orient = new_local_aa_rep[:, 0, :]
    new_seq_pose_body = new_local_aa_rep[:, 1:, :]
    new_obj_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(new_obj_q[0]).float())
    new_obj_trans = new_obj_x[0]

    poses = np.concatenate((new_seq_root_orient, new_seq_pose_body.reshape(-1, 63), padding_zeros_hand), axis=1)

    obj = {
        'rot': np.array(new_obj_rot_mat),
        'trans': np.array(new_obj_trans),
        'name': object_name,
    }

    human = {
        'poses': np.array(poses),
        'betas': np.array(betas),
        'trans': np.array(new_seq_root_trans),
        'gender': gender,
    }
    return human, obj


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--src", type=str, default="", help="Path to BEHAVE dataset")
    parser.add_argument("--obj_root", type=str, default="./data/omomo/objects", help="Path to OMOMO objects")
    # parser.add_argument("--render", action="store_true", default=False, help="Render comparison video.")
    # parser.add_argument("--dst", type=str, default="dataset/smplx_motion/behave_small", help="Output path")
    parser.add_argument("--penetration_check", action="store_true", default=False, help="Perform penetration check.")
    # parser.add_argument("--result_src", type=str, required=True,
    #                     help="Path to Omniretarget .npz results (needed for scale).")
    # [新增] Eval 结果路径
    parser.add_argument("--eval_src", type=str, required=True,
                        help="Path to Eval .pkl results (e.g. refine_results/succeed)")

    args = parser.parse_args()
    # output_dir = args.dst

    skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/smplx_humanoid_hand.xml")
    all_sequences = glob.glob(f"{args.src}/**/", recursive=True)

    smplx_parser_n = SMPLX_Parser(
        model_path='data/SMPL/smplx',
        gender='neutral',
        use_pca=False,
        create_transl=False,
        flat_hand_mean=True,
        num_betas=20
    )

    OBJECT_PATH = "data/omomo/objects/objects"
    dataset_name = "omomo_index_action_rate"
    render_outdir = f"renders/{dataset_name}"

    data_dict = get_omomo_data(args.src)
    data_dict_seq = {data_dict[k]['seq_name']: data_dict[k] for k in data_dict}
    target_keys = list(data_dict_seq.keys())

    OFFSET_FILE = "center_offset.npy"

    if not os.path.exists(OBJECT_PATH):
        os.makedirs(os.path.dirname(OBJECT_PATH), exist_ok=True)
        print("Preprocessing objects: Scaling and Centering...")
        scale_map = {v['seq_name'].split("_")[1]: v['obj_scale'][0] for v in data_dict_seq.values()}
        for obj_name in os.listdir(args.obj_root):
            obj_name = obj_name.replace("_cleaned_simplified.obj", "")
            obj_dir = os.path.join(OBJECT_PATH, obj_name)
            os.makedirs(obj_dir, exist_ok=True)
            raw_path = os.path.join(args.obj_root, f"{obj_name}_cleaned_simplified.obj")

            if obj_name not in scale_map or not os.path.exists(raw_path):
                continue

            mesh = trimesh.load(raw_path, force='mesh')
            mesh.vertices *= scale_map[obj_name]
            center = mesh.vertices.mean(axis=0)
            mesh.vertices -= center

            mesh.export(os.path.join(obj_dir, f"{obj_name}.obj"))
            np.save(os.path.join(obj_dir, OFFSET_FILE), center)
            print(f"Processed {obj_name}: Offset={center}")

    MODEL_PATH = "data/SMPL"
    smpl_model_male = smplx.create(model_path=MODEL_PATH, model_type='smplx', gender='male', use_pca=False,
                                   num_betas=16, ext='pkl')
    smpl_model_female = smplx.create(model_path=MODEL_PATH, model_type='smplx', gender='female', use_pca=False,
                                     num_betas=16, ext='pkl')
    smpl = {'male': smpl_model_male, 'female': smpl_model_female}

    camera_config = {
        'distance': 4.5,
        'azimuth': 0,
        'elevation': -15,
        'lookat_offset': np.array([0, 0, 0.7])
    }

    retarget_seq_list = glob.glob(osp.join(args.eval_src, "*.npz"))

    eval_list = glob.glob(osp.join(args.eval_src, "*.pkl"))
    eval_keys = [osp.basename(p).replace(".pkl", "") for p in eval_list]

    for seq_key in tqdm(target_keys):
        # 导入 Omniretarget 数据 (主要是为了复用其计算出的 scale 和 robot XML)
        if seq_key not in eval_keys:
            # print(f"Skipping {seq_key}, no retargeted result found.")
            continue
        # else:
        #     npz_path = osp.join(args.result_src, f"{seq_key}_original.npz")

        human, obj = omomo_preprocess(data_dict_seq[seq_key])
        entry = data_dict_seq[seq_key]

        key_str = data_dict_seq[seq_key]['seq_name']
        pose_aa_smpl = human['poses']
        trans_smpl = human['trans']
        betas_smpl = human['betas'][np.newaxis, :]
        gender = str(human['gender'])

        seq_name = entry['seq_name']
        obj_name = obj['name']
        obj_angles = sRot.from_matrix(obj['rot']).as_rotvec()
        obj_trans_raw = obj['trans']

        obj_dir_path = osp.join(OBJECT_PATH, obj_name)
        obj_mesh_path = osp.join(obj_dir_path, f"{obj_name}.obj")
        offset_path = osp.join(obj_dir_path, "center_offset.npy")

        mesh_obj = trimesh.load(obj_mesh_path, force='mesh')

        if os.path.exists(offset_path):
            center_offset = np.load(offset_path)
            rot_obj = sRot.from_rotvec(obj_angles)
            offset_world = rot_obj.apply(center_offset)
            obj_trans = obj_trans_raw + offset_world
        else:
            print(f"Warning: Offset file not found for {obj_name}, using raw trans.")
            obj_trans = obj_trans_raw

        obj_file = os.path.dirname(obj_mesh_path)
        points_cache_path = os.path.join(obj_file, 'sampled_points.pt')
        if os.path.exists(points_cache_path):
            object_points = torch.load(points_cache_path, map_location=device)
        else:
            pts, _ = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2025)
            object_points = torch.from_numpy(pts).float().to(device)
            torch.save(object_points, points_cache_path)

        obj = {'angles': obj_angles, 'trans': obj_trans, 'name': obj_name}
        human = {'poses': pose_aa_smpl, 'betas': betas_smpl, 'trans': trans_smpl, 'gender': gender}

        human_yup, obj_yup = tranfrom_to_yup(smpl[gender], human, obj, mesh_obj, origin_format='zup')

        body_clouds, body_geoms, mj_model = build_local_templates_by_body(
            "data/robots/smpl/smplx_humanoid_hand.xml",
            samples_per_geom=500)
        mj_data = mujoco.MjData(mj_model)
        sk2mj, mj2sk = build_sk2mj_index(mj_model, skeleton_tree, drop_world=True)

        # # 核心：获取 smpl_scale (虽然我们不用 retarget 的 motion，但需要它的 scale)
        # new_sk_state, object_dict, _, smpl_scale = from_retarget_to_simulation(
        #     npz_path, mj_model, mj_data, skeleton_tree, device=device
        # )
        # object_dict['name'] = obj_name
        #
        # motion_dict = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30).to_dict()
        #
        # # 计算接触（保留，用于 .npy 生成逻辑，如果不跑 npy 可注释）
        # # print(f"Computing Contacts via Robot Geometry (Z-up)...")
        # # 简化代码，这里主要为了 Render
        # obj_pos_seq = object_dict['obj_pos']
        # obj_rot_seq = object_dict['obj_rot']
        #
        # # ... (接触计算部分略，如果只是为了渲染视频可以不跑 contacts_from_xml_pointcloud，但为了完整性保留) ...
        # # 为了速度，如果只是渲染，可以注释掉下面这块 contacts_from_xml_pointcloud
        # # 如果需要生成最终的 .npy 供训练使用，则保留
        # cg_mj = np.zeros((len(obj_pos_seq), 52))
        # ig_mj = np.zeros((len(obj_pos_seq), 52, 3))
        # ig_idx_mj = np.zeros((len(obj_pos_seq), 52))
        # collision_tag = np.zeros((len(obj_pos_seq),), dtype=bool)

        # =========================================================
        # RENDER BLOCK
        # =========================================================
        # -----------------------------------------------------
        # 1. 渲染 Reference (GT)
        # -----------------------------------------------------
        output_mesh_video = f"{render_outdir}/origin/{seq_name}.mp4"
        os.makedirs(os.path.dirname(output_mesh_video), exist_ok=True)

        if not os.path.exists(output_mesh_video):
            render_smpl_hoi_video_yup(
                smpl_model=smpl[gender].to(device),
                human=human_yup,
                obj=obj_yup,
                output_path=output_mesh_video,
                fps=30,
                camera_cfg=camera_config,
                obj_mesh_path=obj_mesh_path
            )

        # -----------------------------------------------------
        # 2. 渲染 Simulation (Eval Result)
        # -----------------------------------------------------
        eval_outdir = f'{render_outdir}/eval_sim'
        os.makedirs(eval_outdir, exist_ok=True)
        eval_video_path = osp.join(eval_outdir, f"{key_str}.mp4")

        # 修复路径拼接逻辑
        pkl_path = os.path.join(args.eval_src, f"{key_str}.pkl")
        if not os.path.exists(pkl_path):
            pkl_path = os.path.join(args.eval_src, f"{seq_name}.pkl")

        if os.path.exists(pkl_path):
            print(f"Loading Eval data: {pkl_path}")
            eval_data = joblib.load(pkl_path)

            sim_traj = {}
            # IsaacGym 输出 [T, Num_Bodies, 3/4]
            N = eval_data['pred_pos'].shape[0]
            sim_traj['root_trans_offset'] = eval_data['pred_pos'][:, 0, :]
            sim_traj['root_rotation'] = eval_data['pred_rot'][:, 0, :]
            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,
                torch.tensor(eval_data['pred_rot']),  # 全局旋转
                torch.tensor(eval_data['pred_pos'][:, 0, :]),  # 根节点位移
                is_local=False  # 指定输入的是全局坐标
            )

            sim_traj['dof'] = sRot.from_quat(new_sk_state.local_rotation[:, 1:].reshape(-1, 4)).as_euler('XYZ').reshape(N, -1, 3)
            # eval_data['pred_dof_pos']

            # 兼容不同 key
            sim_obj_pos = eval_data.get('pred_obj_pos')
            sim_obj_rot = eval_data.get('pred_obj_rot')

            # 创建 XML (传入正确的 scale)
            temp_xml = create_temp_xml_with_object(
                "data/robots/smpl/smplx_humanoid_hand.xml",
                obj_mesh_path,
                smpl_scale=1
            )

            # MuJoCo 渲染
            export_mujoco_video_hoi(
                sim_traj,
                obj_pos=sim_obj_pos,
                obj_quat_xyzw=sim_obj_rot,
                camera_cfg=camera_config,
                xml_path=temp_xml,
                output_path=eval_video_path
            )

            # -----------------------------------------------------
            # 3. FFmpeg 合成
            # -----------------------------------------------------
            compare_outdir = f'{render_outdir}/comparison'
            os.makedirs(compare_outdir, exist_ok=True)
            comparison_video_path = osp.join(compare_outdir, f"{key_str}.mp4")

            print(f"Synthesizing comparison: {comparison_video_path}...")
            filter_str = (
                "[0:v]crop=720:720:(in_w-720)/2:(in_h-720)/2,setsar=1,format=yuv420p[v0];"
                "[1:v]crop=720:720:(in_w-720)/2:(in_h-720)/2,setsar=1,format=yuv420p[v1];"
                "[v0][v1]hstack"
            )

            cmd = [
                'ffmpeg', '-y',
                '-r', '30', '-i', output_mesh_video,  # Input 0: GT
                '-r', '30', '-i', eval_video_path,  # Input 1: Eval Sim (Corrected)
                '-filter_complex', filter_str,
                '-r', '30',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                comparison_video_path
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"✅ Comparison video saved: {comparison_video_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ FFmpeg Error for {key_str}")
                if e.stderr:
                    print(e.stderr)
        else:
            print(f"⚠️ Eval .pkl not found: {pkl_path}")

        # Debug & Save NPY Logic (Keep original logic if needed)
        # ... (Debug / NPY saving code omitted for brevity as Render is the focus) ...
        # If you need NPY saving, uncomment the contact calculation block above.
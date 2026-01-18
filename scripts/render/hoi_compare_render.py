
import shutil
import smplx
import joblib
import pytorch3d.transforms as transforms



import os
import sys
import os.path as osp

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
from tqdm import tqdm
import argparse
import glob
import trimesh
import torch
import mujoco
import numpy as np
import smplx
import subprocess

from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPLH_MUJOCO_NAMES, SMPLX_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_parser import SMPLX_Parser

from scripts.smpl2sim.hoi.mujoco_contact_inference import (build_local_templates_by_body, prepare_batch_body_cloud,
                                                           get_aligned_vhacd_hulls, compute_hull_planes,
                                                           quick_viz_ig_cg, build_sk2mj_index,
                                                           quick_viz_penetration_vhacd, penetration_depth_vhacd_cuda)
from scripts.smpl2sim.hoi.omomo_utils import rotate_at_frame_w_obj, get_smpl_parents
from scripts.render.mujoco_render import export_mujoco_video_hoi_wxyz, create_temp_xml_with_object, export_mujoco_video_hoi
from scripts.render.hoi_render import render_smpl_hoi_video_yup
from scripts.smpl2sim.hoi.smpl2sim_utils import from_yup_to_simulation, tranfrom_to_yup, from_retarget_to_simulation
from scripts.smpl2sim.hoi.mujoco_contact_inference import (
    contacts_from_xml_pointcloud,
    build_qpos_seq_from_state,
    debug_viz_vhacd_cuda_step
)

sys.path.append(os.getcwd())

def get_omomo_data(raw_p_path):
    print(f"Loading OMOMO data from {raw_p_path}...")
    return joblib.load(raw_p_path)

def omomo_preprocess(seq_data):
    seq_name = seq_data['seq_name']

    object_name = seq_name.split("_")[1]

    trans2joint = seq_data['trans2joint']
    rest_human_offsets = seq_data['rest_offsets']

    betas = seq_data['betas'][0]  # 1 X 16
    gender = seq_data['gender']

    trans = seq_data['trans']  # T X 3
    frame_times = len(trans)
    global_orient = seq_data['root_orient']  # T X 3
    body_pose = seq_data['pose_body'].reshape(-1, 21, 3)  # T X 63

    obj_trans = seq_data['obj_trans'][:, :, 0]  # T X 3
    obj_rot = seq_data['obj_rot']  # T X 3 X 3
    obj_com_pos = seq_data['obj_com_pos']  # T X 3

    padding_zeros_hand = np.zeros((frame_times, 90))

    joint_aa_rep = torch.cat((torch.from_numpy(global_orient).float()[:, None, :], \
                              torch.from_numpy(body_pose).float()), dim=1)  # T X J X 3
    X = torch.from_numpy(rest_human_offsets).float()[None].repeat(joint_aa_rep.shape[0], 1,
                                                                  1).detach().cpu().numpy()  # T X J X 3
    X[:, 0, :] = trans
    local_rot_mat = transforms.axis_angle_to_matrix(joint_aa_rep)  # T X J X 3 X 3
    Q = transforms.matrix_to_quaternion(local_rot_mat).detach().cpu().numpy()  # T X J X 4

    obj_x = obj_trans.copy()  # T X 3
    obj_rot_mat = torch.from_numpy(obj_rot).float()  # T X 3 X 3
    obj_q = transforms.matrix_to_quaternion(obj_rot_mat).detach().cpu().numpy()  # T X 4
    parents = get_smpl_parents()
    _, _, new_obj_x, new_obj_q = rotate_at_frame_w_obj(X[np.newaxis], Q[np.newaxis], \
                                                       obj_x[np.newaxis], obj_q[np.newaxis], \
                                                       trans2joint[np.newaxis], parents, n_past=1, floor_z=True)
    # 1 X T X J X 3, 1 X T X J X 4, 1 X T X 3, 1 X T X 4

    X, Q, new_obj_com_pos, _ = rotate_at_frame_w_obj(X[np.newaxis], Q[np.newaxis], \
                                                     obj_com_pos[np.newaxis], obj_q[np.newaxis], \
                                                     trans2joint[np.newaxis], parents, n_past=1, floor_z=True)

    new_seq_root_trans = X[0, :, 0, :]  # T X 3
    new_local_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(Q[0]).float())  # T X J X 3 X 3
    new_local_aa_rep = transforms.matrix_to_axis_angle(new_local_rot_mat)  # T X J X 3
    new_seq_root_orient = new_local_aa_rep[:, 0, :]  # T X 3
    new_seq_pose_body = new_local_aa_rep[:, 1:, :]  # T X 21 X 3
    new_obj_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(new_obj_q[0]).float())
    new_obj_trans = new_obj_x[0]
    cano_obj_mat = torch.matmul(new_obj_rot_mat[0], obj_rot_mat[0].transpose(0, 1))

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
    parser.add_argument("--process_split", type=str, default="train", choices=["train", "test", "valid"])
    parser.add_argument("--render", action="store_true", default=False, help="Whether to render the \
                                                                        retargeted motion using scenepic animation.")
    parser.add_argument("--eval_src", type=str, default=None,
                        help="Path to the directory containing Eval .pkl results (generated by runner_eval.py)")
    args = parser.parse_args()
    output_dir = args.dst

    skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/smplx_humanoid_hand.xml")

    all_sequences = glob.glob(f"{args.src}/**/", recursive=True)
    behave_full_motion_dict = {}

    smplx_parser_n = SMPLX_Parser(
        model_path='data/SMPL/smplx',
        gender='neutral',
        use_pca=False,  # 关键：不用 PCA，接受 45D/手
        create_transl=False,
        flat_hand_mean=True,
        num_betas=20  # SMPL-X 20 维 beta
    )

    OBJECT_PATH = "data/omomo/objects/objects"
    dataset_name = args.dst.split("/")[-1]
    render_outdir = f"renders/{dataset_name}"

    data_dict = get_omomo_data(args.src)
    data_dict_seq = {data_dict[k]['seq_name']: data_dict[k] for k in data_dict}
    target_keys = list(data_dict_seq.keys())  # 默认处理所有


    OFFSET_FILE = "center_offset.npy"

    if not os.path.exists(OBJECT_PATH):
        # 1. 复制目录 (会自动创建 OBJECT_PATH)
        os.makedirs(os.path.dirname(OBJECT_PATH), exist_ok=True)
        print("Preprocessing objects: Scaling and Centering...")

        # 2. 快速构建 {obj_name: scale} 映射表
        scale_map = {v['seq_name'].split("_")[1]: v['obj_scale'][0] for v in data_dict_seq.values()}

        # 3. 遍历并处理
        for obj_name in os.listdir(args.obj_root):
            obj_name = obj_name.replace("_cleaned_simplified.obj", "")
            obj_dir = os.path.join(OBJECT_PATH, obj_name)
            os.makedirs(obj_dir, exist_ok=True)
            raw_path = os.path.join(args.obj_root, f"{obj_name}_cleaned_simplified.obj")

            # 校验：必须在数据字典中、必须有原始mesh文件
            if obj_name not in scale_map or not os.path.exists(raw_path):
                continue

            # 加载 -> 缩放 -> 计算中心 -> 中心化
            mesh = trimesh.load(raw_path, force='mesh')
            mesh.vertices *= scale_map[obj_name]
            center = mesh.vertices.mean(axis=0)
            mesh.vertices -= center

            mesh.export(os.path.join(obj_dir, f"{obj_name}.obj"))
            np.save(os.path.join(obj_dir, OFFSET_FILE), center)
            print(f"Processed {obj_name}: Offset={center}")

    MODEL_PATH = "data/SMPL"
    smpl_model_male = smplx.create(model_path=MODEL_PATH, model_type='smplx', gender='male', use_pca=False,
                                num_betas=16, ext='pkl') # .to(device)
    smpl_model_female = smplx.create(model_path=MODEL_PATH, model_type='smplx', gender='female', use_pca=False,
                                num_betas=16, ext='pkl') # .to(device)

    smpl = {'male': smpl_model_male, 'female': smpl_model_female}

    camera_config = {
        'distance': 4.5,  # 相机距离目标的距离
        'azimuth': 0,  # 水平旋转角度（度）
        'elevation': -15,  # 仰角（度），负值通常表示从上往下看
        'lookat_offset': np.array([0, 0, 0.7])  # 目标点相对于根节点的偏移（看人中心）
    }

    retarget_seq_list = glob.glob(osp.join(args.result_src, "*.npz"))

    for seq_key in tqdm(target_keys):
        # 导入 Omniretarget 数据
        if seq_key + "_original.npz" not in [osp.basename(p) for p in retarget_seq_list]:
            print(f"Skipping {seq_key}, no retargeted result found.")
            continue
        else:
            npz_path = osp.join(args.result_src, f"{seq_key}_original.npz")

        # if seq_key != 'sub2_woodchair_010':
        #     continue
        human, obj = omomo_preprocess(data_dict_seq[seq_key])
        entry = data_dict_seq[seq_key]

        # 导入人体 SMPL 数据
        key_str = data_dict_seq[seq_key]['seq_name']
        pose_aa_smpl = human['poses']  #pose zup 只是朝向为 zup，相对旋转还是 yup
        trans_smpl = human['trans']
        betas_smpl = human['betas'][np.newaxis, :]
        gender =  str(human['gender'])

        # 导入物体数据
        seq_name = entry['seq_name']
        obj_name = obj['name']
        obj_angles = sRot.from_matrix(obj['rot']).as_rotvec()
        obj_trans_raw = obj['trans']

        # 2. 加载中心化后的 Mesh 和 Offset
        obj_dir_path = osp.join(OBJECT_PATH, obj_name)
        obj_mesh_path = osp.join(obj_dir_path, f"{obj_name}.obj")
        offset_path = osp.join(obj_dir_path, "center_offset.npy")

        mesh_obj = trimesh.load(obj_mesh_path, force='mesh')  # 这是已经中心化的 Mesh

        # 3. 修正物体平移轨迹
        # 原理: T_new = T_raw + R * Offset
        if os.path.exists(offset_path):
            center_offset = np.load(offset_path)  # (3,)

            # 计算每一帧的旋转对 Offset 的影响
            # (T, 3, 3) * (3, 1) -> (T, 3)
            # 使用 scipy Rotation 的 apply 方法最快
            rot_obj = sRot.from_rotvec(obj_angles)
            offset_world = rot_obj.apply(center_offset)

            # 更新 obj_trans
            obj_trans = obj_trans_raw + offset_world
        else:
            print(f"Warning: Offset file not found for {obj_name}, using raw trans.")
            obj_trans = obj_trans_raw


        obj_file = os.path.dirname(obj_mesh_path)
        points_cache_path = os.path.join(obj_file, 'sampled_points.pt')
        if os.path.exists(points_cache_path):
            # 显式指定 map_location
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




        if args.render:
            # 渲染原始人体与物体交互视频 (Y up)
            output_mesh_video = f"{render_outdir}/origin/{seq_name}.mp4"
            os.makedirs(os.path.dirname(output_mesh_video), exist_ok=True)

            render_smpl_hoi_video_yup(
                smpl_model=smpl[gender].to(device), human=human_yup, obj=obj_yup, output_path=output_mesh_video,
                fps=30, camera_cfg=camera_config, obj_mesh_path=obj_mesh_path
            )

            # 渲染机器人与物体交互视频
            rollout_outdir = f'{render_outdir}/rollout'
            os.makedirs(rollout_outdir, exist_ok=True)
            rollout_video_path = osp.join(rollout_outdir, f"{key_str}.mp4")
            os.makedirs(render_outdir, exist_ok=True)
            temp_xml = create_temp_xml_with_object("data/robots/smpl/smplx_humanoid_hand.xml", obj_mesh_path)

            # camera_config['azimuth'] = 180
            eval_pkl_path = osp.join(args.eval_src, f"{key_str}.pkl")
            print(f"Loading Eval data from: {eval_pkl_path}")
            eval_data = joblib.load(eval_pkl_path)

            # --- 数据映射: Eval .pkl -> MuJoCo Render Format ---
            # 假设 runner_eval.py 导出的 pkl 包含以下 keys (基于你之前的 runner_eval 代码):
            # 'pred_dof_pos', 'pred_pos' (root在其中), 'pred_rot', 'obj_pos', 'obj_rot' (或 'pred_obj_pos')

            # 1. 提取物体数据
            # 优先查找 HOI 相关的 key，如果没有则回退到 'obj_pos'
            sim_obj_pos = eval_data.get('pred_obj_pos', eval_data.get('obj_pos'))
            sim_obj_rot = eval_data.get('pred_obj_rot', eval_data.get('obj_rot'))  # 假设是 xyzw 或 wxyz，需确认

            # 2. 提取机器人数据
            # 需要构建 motion_traj 字典
            sim_traj = {}

            # 解析 Root State (通常 body_pos 的第0个索引是 Root)
            # 注意：IsaacGym 的 body_pos 通常是全局坐标
            if 'pred_pos' in eval_data and 'pred_rot' in eval_data:
                sim_traj['root_trans_offset'] = eval_data['pred_pos'][:, 0, :]  # (T, 3)
                sim_traj['root_rotation'] = eval_data['pred_rot'][:, 0, :]  # (T, 4) xyzw

            # 解析 DOF (关节角)
            if 'pred_dof_pos' in eval_data:
                sim_traj['dof'] = eval_data['pred_dof_pos']
            else:
                print(f"Warning: 'pred_dof_pos' not found in {eval_pkl_path}, skipping eval render.")
                continue

            # 3. 设置路径
            eval_outdir = f'{render_outdir}/eval_sim'
            os.makedirs(eval_outdir, exist_ok=True)
            eval_video_path = osp.join(eval_outdir, f"{key_str}.mp4")

            # 4. 创建临时 XML (带物体)
            temp_xml = create_temp_xml_with_object("data/robots/smpl/smplx_humanoid_hand.xml", obj_mesh_path)

            # 5. 调用渲染
            # 注意：确保 sim_obj_rot 是 xyzw 格式，如果 export 函数需要 wxyz，请在此处转换
            export_mujoco_video_hoi(
                sim_traj,
                obj_pos=sim_obj_pos,
                obj_quat_xyzw=sim_obj_rot,  # 假设 eval 出来的是 xyzw
                camera_cfg=camera_config,
                xml_path=temp_xml,
                output_path=eval_video_path
            )

            compare_outdir = f'{render_outdir}/comparison'
            os.makedirs(compare_outdir, exist_ok=True)
            comparison_video_path = osp.join(compare_outdir, f"{key_str}.mp4")

            print(f"正在合成对比视频: {key_str}...")
            filter_str = (
                "[0:v]crop=720:720:(in_w-720)/2:(in_h-720)/2,setsar=1,format=yuv420p[v0];"
                "[1:v]crop=720:720:(in_w-720)/2:(in_h-720)/2,setsar=1,format=yuv420p[v1];"
                "[v0][v1]hstack"
            )

            cmd = [
                'ffmpeg', '-y',
                '-r', '30', '-i', output_mesh_video,   # 强制输入0帧率为30
                '-r', '30', '-i', retarget_video_path,  # 强制输入1帧率为30
                '-filter_complex', filter_str,
                '-r', '30',                            # 强制输出帧率为30
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                comparison_video_path
            ]

            try:
                # 注意：这里将 stderr 改为 PIPE，以便失败时查看报错
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"✅ 对比视频已保存至: {comparison_video_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ FFmpeg 合成失败: {key_str}")
                print("--- FFmpeg 错误日志 (Stderr) ---")
                print(e.stderr)
                print("-------------------------------")
                break

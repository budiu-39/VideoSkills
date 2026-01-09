
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
from scripts.smpl2sim.hoi.hoi_retarget_utils import (compute_cg_ig_via_smplh_contacts_yup, get_smpl_vert_part)

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
    parser.add_argument("--dst", type=str, default="dataset/smplx_motion/behave_small", help="Output path")
    parser.add_argument("--penetration_check", action="store_true", default=False,
                        help="Whether to perform penetration check.")
    parser.add_argument("--filter_file", type=str, default=None,
                        help="Path to a txt file containing specific sequences to process.")
    parser.add_argument("--result_src", type=str, required=True,
                        help="Path to the directory containing Omniretarget .npz results.")
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

    OBJECT_PATH = "data/omomo/objects_scaled/objects"
    render_outdir = "renders/OMOMO_Omniretargeted"

    data_dict = get_omomo_data(args.src)
    data_dict_seq = {data_dict[k]['seq_name']: data_dict[k] for k in data_dict}
    target_keys = list(data_dict_seq.keys())  # 默认处理所有

    if args.filter_file and os.path.exists(args.filter_file):
        print(f"Applying filter from: {args.filter_file}")
        with open(args.filter_file, 'r') as f:
            # 读取每一行，去除首尾空格，去除 .pt 后缀
            valid_names = set()
            for line in f:
                name = line.strip()
                if not name: continue  # 跳过空行
                if name.endswith('.pt'):
                    name = name[:-3]  # 去掉 .pt
                valid_names.add(name)

        # 取交集：只保留既在 data_dict 中又在 txt 中的序列
        target_keys = [k for k in target_keys if k in valid_names]
        print(f"Filter applied. Processing {len(target_keys)} sequences.")

        if len(target_keys) == 0:
            print("Warning: No matching sequences found in the filter list!")

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

        # 导入 Omniretarget 数据
        if seq_name + "_original.npz" not in [osp.basename(p) for p in retarget_seq_list]:
            print(f"Skipping {seq_name}, no retargeted result found.")
            continue
        else:
            npz_path = osp.join(args.result_src, f"{seq_name}_original.npz")

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

        # new_sk_state, object_dict = from_yup_to_simulation(human_yup, obj_yup, smpl[gender],
        #                                                   smplx_parser_n, skeleton_tree, mesh_obj)

        body_clouds, body_geoms, mj_model = build_local_templates_by_body(
            "data/robots/smpl/smplx_humanoid_hand.xml",
            samples_per_geom=500)
        mj_data = mujoco.MjData(mj_model)
        sk2mj, mj2sk = build_sk2mj_index(mj_model, skeleton_tree, drop_world=True)

        new_sk_state, object_dict, _, smpl_scale = from_retarget_to_simulation(
            npz_path, mj_model, mj_data, skeleton_tree, device=device
        )

        motion_dict = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30).to_dict()

        # 建立 mujoco 模型，用于穿模计算和可视化检查


        # 计算 cg 和 ig
        smpl_vert_part = get_smpl_vert_part(smpl[gender])
        smpl_2_mujoco = [SMPLX_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLX_BONE_ORDER_NAMES]
        cg_yup, ig_yup = compute_cg_ig_via_smplh_contacts_yup(
            smpl_model=smpl[gender],
            human=human_yup,
            obj=obj_yup,
            obj_mesh_path=obj_mesh_path,
            smpl_vert_part=smpl_vert_part
        )

        # 注意：from_yup_to_simulation 内部会计算 R_yup2zup_mat (RotX 90)
        # 交互信息 IG/CG 也需要相应旋转
        R_yup2zup = sRot.from_euler('x', 90, degrees=True).as_matrix()

        # IG 是方向向量，需要旋转；CG 是标签，不需要旋转
        ig_mj = ig_yup[:, smpl_2_mujoco, :] @ R_yup2zup.T
        cg_mj = cg_yup[:, smpl_2_mujoco]

        collision_tag = np.zeros((len(cg_mj),), dtype=bool)

        if args.penetration_check:
            # 1. [预处理] V-HACD (只做一次)
            obj_mesh = trimesh.load(obj_mesh_path, force='mesh')
            vhacd_cache = obj_mesh_path.replace(".obj", "_aligned_vhacd.pkl")
            hulls_template = get_aligned_vhacd_hulls(
                mesh_obj, vhacd_cache, resolution=300000, max_hulls=64, max_v_per_ch=64
            )
            hulls_planes_local = [compute_hull_planes(h) for h in hulls_template]

            # 2. [预处理] 计算物体的 AABB (Local Space)
            # 用所有凸包的顶点合并计算，比原始 Mesh 更准确对应 V-HACD 形状
            all_hull_verts = np.concatenate([h.vertices for h in hulls_template], axis=0)
            obj_aabb_min = all_hull_verts.min(axis=0)
            obj_aabb_max = all_hull_verts.max(axis=0)

            # 3. [预处理] 批处理人体点云
            # 将 dict 形式的点云转为大数组，方便后续向量化操作
            B_len = new_sk_state.global_translation.shape[1]
            batch_pts_local, batch_indices = prepare_batch_body_cloud(body_clouds, mj2sk, B_len)

            # 4. 准备循环数据
            T_len = len(object_dict['obj_pos'])
            pen_seq = np.zeros((T_len, B_len), dtype=np.float32)

            body_pos_all = new_sk_state.global_translation
            body_rot_all = new_sk_state.global_rotation


            # 物体数据 (numpy -> tensor)
            obj_pos_gpu = torch.from_numpy(object_dict['obj_pos']).to(device).float()
            obj_rot_gpu = torch.from_numpy(object_dict['obj_rot']).to(device).float()

            # 3. 调用 CUDA 函数 (一次性计算所有帧)
            print("Running CUDA Penetration Check...")
            pen_seq_gpu = penetration_depth_vhacd_cuda(
                body_pos_seq=body_pos_all.to(device),
                body_rot_seq=body_rot_all.to(device),
                all_body_local_pts=torch.from_numpy(batch_pts_local).to(device).float(),
                body_indices=torch.from_numpy(batch_indices).to(device).long(),
                hulls_planes_local=hulls_planes_local,  # 里面是 numpy，函数内部会转
                obj_pos_seq=obj_pos_gpu,
                obj_rot_seq=obj_rot_gpu,
                obj_aabb_min=obj_aabb_min,
                obj_aabb_max=obj_aabb_max,
                device=device
            )

            # 4. 转回 Numpy 用于保存
            pen_seq = pen_seq_gpu.cpu().numpy()
            collision_tag = (pen_seq > 0.01).any(axis=1)


        if args.render:
            # 渲染原始人体与物体交互视频 (Y up)
            output_mesh_video = f"{render_outdir}/origin/{seq_name}.mp4"
            os.makedirs(os.path.dirname(output_mesh_video), exist_ok=True)

            render_smpl_hoi_video_yup(
                smpl_model=smpl[gender].to(device), human=human_yup, obj=obj_yup, output_path=output_mesh_video,
                fps=30, camera_cfg=camera_config, obj_mesh_path=obj_mesh_path
            )

            # 渲染机器人与物体交互视频
            retarget_outdir = f'{render_outdir}/retarget'
            os.makedirs(retarget_outdir, exist_ok=True)
            retarget_video_path = osp.join(retarget_outdir, f"{key_str}.mp4")
            N = new_sk_state.local_rotation.shape[0]
            os.makedirs(render_outdir, exist_ok=True)
            temp_xml = create_temp_xml_with_object("data/robots/smpl/smplx_humanoid_hand.xml", obj_mesh_path)
            # ROBOT_XML = "data/robots/smpl/smplx_humanoid_hand.xml"
            # temp_xml = create_temp_xml_with_object(
            #     ROBOT_XML,
            #     obj_mesh_path,
            #     smpl_scale=smpl_scale  # <--- 传入 Scale
            # )

            motion_traj = {}
            motion_traj['root_trans_offset'] = new_sk_state.root_translation.numpy()
            motion_traj['root_rotation'] = new_sk_state.global_root_rotation.numpy()
            motion_traj['dof'] = sRot.from_quat(new_sk_state.local_rotation[:, 1:].reshape(-1, 4)).as_euler('XYZ').reshape(N, -1, 3)

            # camera_config['azimuth'] = 180
            export_mujoco_video_hoi(
                motion_traj,
                obj_pos=object_dict['obj_pos'],
                obj_quat_xyzw=object_dict['obj_rot'],
                camera_cfg=camera_config,
                xml_path=temp_xml,
                output_path=retarget_video_path
            )

            # export_mujoco_video_hoi_wxyz(
            #     motion_dict,
            #     obj_pos=object_dict['obj_pos'],
            #     obj_quat_wxyz=object_dict['obj_rot'],
            #     camera_cfg=camera_config,
            #     xml_path=temp_xml,
            #     output_path=retarget_video_path
            # )

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


        if args.debug:
            t = min(50, len(cg_mj) - 1)
            # 计算该帧物体点云 (Z-up)
            o_pos = object_dict['obj_pos'][t]
            o_quat = object_dict['obj_rot'][t]
            r_mat = sRot.from_quat(o_quat).as_matrix()
            # object_points 是循环前采样好的 (1024, 3) 局部点
            object_points_world = (object_points.cpu().numpy() @ r_mat.T) + o_pos

            hulls_world_viz = []
            obj_mat = np.eye(4)
            obj_mat[:3, :3] = r_mat
            obj_mat[:3, 3] = o_pos

            hulls_world_viz = []
            # 只有在 penetration_check 开启时才有 hulls_template
            if args.penetration_check:
                obj_mat = np.eye(4)
                obj_mat[:3, :3] = r_mat # 使用正确的旋转矩阵
                obj_mat[:3, 3] = o_pos
                # 使用循环外定义的 hulls_template
                for h in hulls_template:
                    hw = h.copy()
                    hw.apply_transform(obj_mat)
                    hulls_world_viz.append(hw)
            # 获取当前帧的穿模数据
            p_row = pen_seq[t] if 'pen_seq' in locals() else None

            quick_viz_penetration_vhacd(
                body_local_clouds=body_clouds,
                obj_pts=object_points_world,
                body_pos_frame=new_sk_state.global_translation[t].cpu().numpy(),
                body_rot_frame=new_sk_state.global_rotation[t].cpu().numpy(),
                mj2sk=mj2sk,
                pen_depth_row=p_row,
                hulls_world=hulls_world_viz, # 传入变换后的凸包
                title=f"VHACD Check: {key_str} (Frame {t})"
            )

            quick_viz_ig_cg(
                body_local_clouds=body_clouds,
                obj_pts=object_points_world,
                # 修复此处：直接从 new_sk_state 获取全局平移
                body_pos_frame=new_sk_state.global_translation[t].cpu().numpy(),
                body_rot_frame=new_sk_state.global_rotation[t].cpu().numpy(),
                mj2sk=mj2sk,  # 关键！必须传入映射表
                contact_row=cg_mj[t],  # cg_mj 已经是重排后的顺序
                ig_frame=ig_mj[t],     # ig_mj 已经是重排后的顺序
                title=f"Fixed Check: {key_str}"
            )

        bundle = {
            "motion": motion_dict,  # SkeletonMotion 的 dict（含关节、根姿态、fps 等）
            "object": object_dict,
            "interaction": {
                "ig": ig_mj,  # (T,52,3) float32（世界系）
                "contact_robot": cg_mj,  # (T,52)   0/1 float32
                "collision_tag": collision_tag
            }
        }

        output_dir_sequences = osp.join(output_dir)
        os.makedirs(output_dir_sequences, exist_ok=True)
        save_path = osp.join(output_dir_sequences, f"{key_str}.npy")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        np.save(save_path, bundle, allow_pickle=True)

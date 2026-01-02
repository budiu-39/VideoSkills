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
import json

from scripts.libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPLH_MUJOCO_NAMES, SMPLH_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_parser import SMPLX_Parser
from scripts.smpl2sim.hoi.mujoco_contact_inference import (build_local_templates_by_body, prepare_batch_body_cloud,
                                                           get_aligned_vhacd_hulls, compute_hull_planes,
                                                           quick_viz_ig_cg, build_sk2mj_index,
                                                           penetration_depth_vhacd_frame, quick_viz_penetration_vhacd,
                                                           penetration_depth_vhacd_cuda)
from scripts.render.mujoco_render import export_mujoco_video_hoi, create_temp_xml_with_object
from scripts.render.hoi_render import render_smpl_hoi_video_yup
from scripts.smpl2sim.hoi.smpl2sim_utils import from_yup_to_simulation, tranfrom_to_yup
from scripts.smpl2sim.hoi.hoi_retarget_utils import (compute_cg_ig_via_smplh_contacts_yup, get_smpl_vert_part)


def load_behave_sequence(sequence_path, parse_frame_times=True):
    # 辅助加载函数：读取 npz 或返回空字典
    def load_npz(name):
        p = osp.join(sequence_path, name)
        return np.load(p, allow_pickle=True) if osp.exists(p) else {}

    f_s = load_npz("smpl_fit_all.npz")
    f_o = load_npz("object_fit_all.npz")

    # 1. SMPL 数据：直接用 np.array(..., dtype=np.float32) 强制转换
    smpl = {k: np.array(f_s[k], dtype=np.float32) for k in ['poses', 'trans', 'betas'] if k in f_s}
    smpl['gender'] = str(f_s.get('gender', 'neutral'))

    smpl['betas'] = smpl['betas']

    # 2. Object 数据
    obj = {k: np.array(f_o[k], dtype=np.float32) for k in ['angles', 'trans'] if k in f_o}

    # 针对带有 't' 前缀的 frame_times 进行内联处理
    if 'frame_times' in f_o and parse_frame_times:
        ft = f_o['frame_times']
        # 如果是字符串数组（BEHAVE 常见情况），去掉 't' 再转 float；否则直接转
        obj['frame_times'] = np.array([str(x).lstrip('tT') for x in ft],
                                      dtype=np.float32) if ft.dtype.kind in 'SUO' else ft.astype(np.float32)

    # 3. Info JSON
    info_p = osp.join(sequence_path, "info.json")
    info = json.load(open(info_p)) if osp.exists(info_p) else {}

    # 4. 组装返回
    out = {
        'smpl': smpl,
        'object': obj,
        'info': {'gender': info.get('gender', 'neutral'), 'cat': info.get('cat', '')}
    }

    # 5. 最小化校验：确保关键数据存在
    if 'poses' not in out['smpl'] or 'trans' not in out['smpl']:
        raise ValueError(f"Data missing in {sequence_path}")

    return out


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
                        help="Whether to perform penetration check and save the results.")
    args = parser.parse_args()
    output_dir = args.dst

    # 初始化 SMPL 模型
    MODEL_PATH = "data/SMPL"
    smpl_model_male = smplx.create(MODEL_PATH, model_type='smplh', gender="male", use_pca=False, num_betas=10,
                                   ext='pkl')
    smpl_model_female = smplx.create(MODEL_PATH, model_type='smplh', gender="female", use_pca=False, num_betas=10,
                                     ext='pkl')
    smpl = {'male': smpl_model_male, 'female': smpl_model_female}

    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": True,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smplx",
    }
    skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/smplx_humanoid_hand.xml")
    smpl_local_robot = LocalRobot(robot_cfg, data_dir="data/SMPL/smplx")

    all_sequences = glob.glob(f"{args.src}/**/", recursive=True)
    behave_full_motion_dict = {}

    # 这个是给机器人用的 SMPLX 层
    smplx_parser_n = SMPLX_Parser(
        model_path='data/SMPL/smplx',
        gender='neutral',
        use_pca=False,
        create_transl=False,
        flat_hand_mean=True,
        num_betas=20  # SMPL-X 20 维 beta
    )

    camera_config = {
        'distance': 4.5,  # 相机距离目标的距离
        'azimuth': 0,  # 水平旋转角度（度）
        'elevation': -15,  # 仰角（度），负值通常表示从上往下看
        'lookat_offset': np.array([0, 0, 0.7])  # 目标点相对于根节点的偏移（看人中心）
    }

    render_outdir = "renders/Behave"

    for sequence_dir in tqdm(all_sequences):
        if not osp.exists(osp.join(sequence_dir, "smpl_fit_all.npz")) and not osp.exists(osp.join(sequence_dir, "human.npz")):
            continue

        norm_path = osp.normpath(sequence_dir)
        seq_name = norm_path.split(osp.sep)[-1]
        # 这样打印会在进度条上方输出，不会打断进度条动画
        tqdm.write(f"Processing {sequence_dir}")

        # 导入人体 SMPL 数据，此时是相机坐标系(Y down)，先把它变成 渲染通用的坐标系(Y up)，再转化成 Z up。
        # 只渲染 Y up 和 Z up
        sequence_data = load_behave_sequence(sequence_dir)
        smpl_data = sequence_data['smpl']
        pose_aa_smpl = smpl_data['poses']
        trans_smpl = smpl_data['trans']
        betas_smpl = smpl_data['betas']
        gender = str(smpl_data['gender'])

        # 导入物体数据
        object_data = sequence_data['object']
        obj_angles = object_data['angles']
        obj_trans = object_data['trans']
        obj_times = object_data['frame_times']
        key_str = os.path.basename(os.path.normpath(sequence_dir))
        obj_name = sequence_data['info']['cat']


        # 导入物体网格，并采样点云
        obj_mesh_path = osp.join(args.obj_root, "objects", obj_name, f"{obj_name}.obj")
        mesh_obj = trimesh.load(obj_mesh_path, force='mesh')

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

        human_yup, obj_yup = tranfrom_to_yup(smpl[gender], human, obj, mesh_obj, origin_format='ydown')

        new_sk_state, object_dict = from_yup_to_simulation(human_yup, obj_yup, smpl[gender],
                                                          smplx_parser_n, skeleton_tree, mesh_obj)


        motion_dict = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30).to_dict()

        # 建立 mujoco 模型，用于穿模计算和可视化检查
        body_clouds, body_geoms, mj_model = build_local_templates_by_body(
            "data/robots/smpl/smplx_humanoid_hand.xml",
            samples_per_geom=500)
        mj_data = mujoco.MjData(mj_model)
        sk2mj, mj2sk = build_sk2mj_index(mj_model, skeleton_tree, drop_world=True)

        # 计算 cg 和 ig
        smpl_vert_part = get_smpl_vert_part(smpl[gender])
        smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
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

            motion_traj = {}
            motion_traj['root_trans_offset'] = new_sk_state.root_translation.numpy()
            motion_traj['root_rotation'] = new_sk_state.global_root_rotation.numpy()
            motion_traj['dof'] = sRot.from_quat(new_sk_state.local_rotation[:, 1:].reshape(-1, 4)).as_rotvec().reshape(
                N, -1, 3)
            export_mujoco_video_hoi(
                motion_traj,
                obj_pos=object_dict['obj_pos'],
                obj_quat_xyzw=object_dict['obj_rot'],
                camera_cfg=camera_config,
                xml_path=temp_xml,
                output_path=retarget_video_path
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




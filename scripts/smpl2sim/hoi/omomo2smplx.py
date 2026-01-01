import os
import sys
import os.path as osp

sys.path.append(os.getcwd())
import subprocess
from scipy.spatial.transform import Rotation as sRot
import numpy as np
from tqdm import tqdm
import argparse
import glob
import trimesh
import torch
import mujoco
import numpy as np
import shutil
import smplx
import joblib
import pytorch3d.transforms as transforms

from scripts.libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPLH_MUJOCO_NAMES, SMPLH_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_parser import SMPLX_Parser
from scripts.smpl2sim.hoi.mujoco_contact_inference import build_local_templates_by_body
from scripts.smpl2sim.hoi.mujoco_contact_inference import quick_viz_frame, build_sk2mj_index
from scripts.render.mujoco_render import export_mujoco_video_hoi, create_temp_xml_with_object
from scripts.render.render_smplh_hoi import render_smplh_hoi_video
from scripts.smpl2sim.hoi.hoi_retarget_utils import (apply_cam2world_rotvec_trans, _quat_rotate_xyzw,
                                                     angular_velocity_world_from_quat_xyzw, penetration_depth_sequence_ig,
                                                     compute_cg_ig_via_smplh_contacts_yup, smplh_vert_part_from_custom_layer)
from human_body_prior.body_model.body_model import BodyModel
from scripts.smpl2sim.hoi.omomo_utils import rotate_at_frame_w_obj, get_smpl_parents
from scripts.smpl2sim.hoi.model_test import compare_root_joint_smplx_vs_bodymodel, check_equivalence_with_known_R
from scripts.render.hoi_render import render_smplx_hoi_video_zup
from scripts.smpl2sim.hoi.smpl2sim_utils import from_yup_to_simulation

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

def from_zup_to_yup(smpl_model, poses, trans, betas, obj_trans, obj_angles, mesh_obj):
    ''' 将 OMOMO 数据集的 Z up 坐标系转换到 Y up 坐标系，输入输出都是 numpy 格式S '''
    rotation_matrix_x = sRot.from_euler('x', -np.pi/2, degrees=False)

    with torch.no_grad():
        smplx_output = smpl_model(
            body_pose=torch.from_numpy(poses[:, 3:66]).float().to(device),
            global_orient=torch.from_numpy(poses[:, :3]).float().to(device),
            betas=torch.from_numpy(betas).float().to(device),
            transl=torch.from_numpy(trans).float().to(device)
        )
        pelvis = smplx_output.joints.detach().cpu().numpy()[:, 0, :]

    # --- 1. 坐标系旋转 (World Transformation) ---
    # 旋转人
    rotvecs = poses[:, :3]
    rotated_rotations = rotation_matrix_x * sRot.from_rotvec(rotvecs)
    poses[:, :3] = rotated_rotations.as_rotvec()
    trans = rotation_matrix_x.apply(trans)

    # 旋转物体 (保持相对 pelvis 关系)
    obj_trans_delta = rotation_matrix_x.apply(obj_trans - pelvis)

    rotated_rotations2 = rotation_matrix_x * sRot.from_rotvec(obj_angles)
    obj_angles = rotated_rotations2.as_rotvec()

    # --- 2. SMPL 前向计算 (Pass 2: 旋转后) ---
    # 这一步是为了拿到旋转后的人体顶点最低点
    with torch.no_grad():
        smplx_output = smpl_model(
            body_pose=torch.from_numpy(poses[:, 3:66]).float().to(device),
            global_orient=torch.from_numpy(poses[:, :3]).float().to(device),
            betas=torch.from_numpy(betas).float().to(device),
            transl=torch.from_numpy(trans).float().to(device)
        )
        verts = smplx_output.vertices.detach().cpu().numpy()
        pelvis = smplx_output.joints.detach().cpu().numpy()[:, 0, :]  # 更新后的 pelvis

    # 更新物体位置
    obj_trans = pelvis + obj_trans_delta

    # --- 3. 落地校准 (Ground Alignment) ---
    # 计算每一帧物体的顶点世界坐标，只取了前 30 帧用于计算最低点
    angle_matrix = sRot.from_rotvec(obj_angles).as_matrix()
    obj_verts_template = mesh_obj.vertices[None, ...]  # (1, V, 3)
    obj_verts_template -= np.mean(obj_verts_template, axis=1, keepdims=True)
    # R * V_T + T
    obj_verts_motion = np.matmul(obj_verts_template, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]

    # 计算 diff
    diff_fix = min(verts[:30, ..., 1].min(), obj_verts_motion[:30, ..., 1].min())

    # 应用落地修正
    obj_trans[..., 1] -= diff_fix
    trans[..., 1] -= diff_fix

    obj = {'angles': obj_angles, 'trans': obj_trans, 'name': obj_name}
    human = {'poses': poses, 'betas': betas, 'trans': trans, 'gender': gender}
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
    args = parser.parse_args()
    output_dir = args.dst

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

    smplx_parser_n = SMPLX_Parser(
        model_path='data/SMPL/smplx',
        gender='neutral',
        use_pca=False,  # 关键：不用 PCA，接受 45D/手
        create_transl=False,
        flat_hand_mean=True,
        num_betas=20  # SMPL-X 20 维 beta
    )

    OBJECT_PATH = "./data/omomo/objects_scaled_center"

    data_dict = get_omomo_data(args.src)

    if not os.path.exists(OBJECT_PATH):
        # os.rename(OBJECT_PATH_RAW, OBJECT_PATH)
        shutil.copytree(args.obj_root, OBJECT_PATH)

        for object in os.listdir(OBJECT_PATH):
            for index in data_dict:
                seq_name = data_dict[index]['seq_name']
                obj_name = seq_name.split("_")[1]
                if obj_name == object.split("_")[0]:
                    print(obj_name)
                    obj_scale = data_dict[index]['obj_scale']
                    mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}_cleaned_simplified.obj"),
                                            force='mesh')
                    mesh_obj.vertices *= obj_scale[0]
                    os.makedirs(os.path.join(OBJECT_PATH, f"{obj_name}"), exist_ok=True)
                    mesh_obj.export(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"))
                    break

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

    for seq_key in tqdm(data_dict.keys()):
        human, obj = omomo_preprocess(data_dict[seq_key])
        entry = data_dict[seq_key]

        # 导入人体 SMPL 数据
        key_str = data_dict[seq_key]['seq_name']
        pose_aa_smpl = human['poses']  #pose zup 只是朝向为 zup，相对旋转还是 yup
        trans_smpl = human['trans']
        betas_smpl = human['betas'][np.newaxis, :]
        gender =  str(human['gender'])

        # 导入物体数据
        seq_name = entry['seq_name']
        obj_name = obj['name']
        obj_angles = sRot.from_matrix(obj['rot']).as_rotvec()
        obj_trans = obj['trans']

        obj_mesh_path = osp.join(OBJECT_PATH, obj_name, f"{obj_name}.obj")
        mesh_obj = trimesh.load(obj_mesh_path, force='mesh')

        output_mesh_video = f"renders/origin/{seq_name}.mp4"
        os.makedirs(os.path.dirname(output_mesh_video), exist_ok=True)

        render_smplx_hoi_video_zup(
            smplx_model=smpl[gender].to(device),
            poses_zup=pose_aa_smpl,  # (T, 156)
            trans_zup=trans_smpl,  # (T, 3)
            betas=betas_smpl,  # (10,)
            obj_mesh_path=obj_mesh_path,
            obj_trans_zup=obj_trans,
            obj_rotmat_zup=obj['rot'],
            output_path=output_mesh_video,
            fps=30,
            camera_cfg=camera_config,
        )

        obj_file = os.path.dirname(obj_mesh_path)
        points_cache_path = os.path.join(obj_file, 'sampled_points.pt')
        if os.path.exists(points_cache_path):
            # 显式指定 map_location
            object_points = torch.load(points_cache_path, map_location=device)
        else:
            pts, _ = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2025)
            object_points = torch.from_numpy(pts).float().to(device)
            torch.save(object_points, points_cache_path)

        human_yup, obj_yup = from_zup_to_yup(smpl[gender], pose_aa_smpl, trans_smpl, betas_smpl,
                                               obj_trans, obj_angles, mesh_obj)

        new_sk_state, object_dict = from_yup_to_simulation(human_yup, obj_yup, smpl[gender],
                                                          smplx_parser_n, skeleton_tree)

        motion_dict = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30).to_dict()

        if args.render:
            render_outdir = 'renders/retarget'
            os.makedirs(render_outdir, exist_ok=True)
            retarget_video_path = osp.join(render_outdir, f"{key_str}.mp4")
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

            compare_outdir = 'renders/comparison'
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

        # # 9. 计算接触和交互信息
        # # compute_cg_ig_via_smplh_contacts_yup 期望输入在 Y-up/camera 坐标系下：
        # # - pose_aa_smpl / trans_smpl 已经是 Y-up
        # # - 这里把物体从 world(Z-up) 转回 camera(Y-up)，避免坐标系混用导致物体“横着/翻倒”
        # obj_angles_cam, obj_trans_cam = apply_cam2world_rotvec_trans(
        #     obj_angles, obj_trans, R_world2cam_mat
        # )
        # obj_quat_cam_xyzw = sRot.from_rotvec(obj_angles_cam).as_quat()
        #
        # cg_np, ig_np = compute_cg_ig_via_smplh_contacts_yup(
        #     smplh_layer=smplh_layer,
        #     pose_aa=pose_aa_smpl,  # (T,D)
        #     betas=betas_smpl,  # (10,) 或 (1,10)
        #     trans=trans_smpl,  # (T,3)
        #     obj_mesh_path=obj_mesh_path,  # 物体mesh（局部坐标）
        #     obj_pos_world=obj_trans_cam,  # (T,3) camera/Y-up
        #     obj_quat_xyzw=obj_quat_cam_xyzw,  # (T,4) xyzw camera/Y-up
        #     smplh_vert_part=smplh_vert_part_from_custom_layer(smplh_layer),
        #     contact_threshold=0.01,
        #     samples_per_object=1024,
        # )
        #
        # ig_mj = ig_np[:, smpl_2_mujoco, :]  # (T,52,3)
        # ig_mj = ig_mj @ np.array(R_cam2world_mat).T
        # cg_mj = cg_np[:, smpl_2_mujoco]  # (T,52)
        #
        # # 穿模计算
        # body_clouds, body_geoms, mj_model = build_local_templates_by_body(
        #     "data/robots/smpl/smplx_humanoid_hand.xml",
        #     samples_per_geom=500)
        # mj_data = mujoco.MjData(mj_model)
        # sk2mj, mj2sk = build_sk2mj_index(mj_model, skeleton_tree, drop_world=True)
        #
        # body_pos = new_sk_state.global_translation
        # obj_pts_world = _quat_rotate_xyzw(q_xyzw, object_points) + p_w.unsqueeze(1)
        # obj_pts_world_np = obj_pts_world.cpu().numpy().astype(np.float32)  # [T,P,3]
        #
        # pen_seq = penetration_depth_sequence_ig(
        #     mj_model,
        #     body_geoms,
        #     mj2sk,
        #     obj_pts_world_np,  # e.g. List[np.ndarray], len T, each (P_t,3)
        #     body_pos.numpy().astype(np.float32),  # (T, B, 3)
        #     new_sk_state.global_rotation.numpy().astype(np.float32),  # (T, B, 4) xyzw
        # )
        #
        # bundle = {
        #     "motion": motion_dict,  # SkeletonMotion 的 dict（含关节、根姿态、fps 等）
        #     "object": {
        #         "name": obj_name,
        #         "obj_pos": obj_pos,
        #         "obj_rot": obj_quat_xyzw,  # xyzw —— Isaac Gym 对齐
        #         "obj_pos_vel": obj_pos_vel,
        #         "obj_rot_vel": obj_rot_vel,
        #     },
        #     "interaction": {
        #         "ig": ig_mj,  # (T,52,3) float32（世界系）
        #         "contact_robot": cg_mj,  # (T,52)   0/1 float32
        #         "collision_tag": (pen_seq > 0).any(axis=1)
        #     }
        # }
        #
        #
        # quick_viz = False
        # if quick_viz:
        #     body_pos_t = new_sk_state.global_translation  # Tensor, 与上文一致
        #     t = min(100, obj_pts_world_np.shape[0] - 1)
        #
        #     body_pos_frame = body_pos_t[t].cpu().numpy().astype(np.float64)
        #     body_rot_frame = new_sk_state.global_rotation[t].cpu().numpy().astype(np.float64)
        #
        #     quick_viz_frame(
        #         mj_model, mj_data,
        #         body_local_clouds=body_clouds,
        #         obj_pts=obj_pts_world_np[t],
        #         body_rot_frame=body_rot_frame,
        #         mj2sk=mj2sk,
        #         title=f"seq:{key_str} t={t}",
        #         body_pos_frame=body_pos_frame,
        #         ig_frame=ig_mj[t],
        #         contact_row=cg_mj[t],
        #     )
        #
        # args.render = True
        # if args.render:
        #     output_dir_render = osp.join(output_dir, "rendered")
        #     os.makedirs(output_dir_render, exist_ok=True)
        #     temp_xml = create_temp_xml_with_object(
        #         "data/robots/smpl/smplx_humanoid_hand.xml",
        #         obj_mesh_path
        #     )
        #     motion_traj = {}
        #     motion_traj['root_trans_offset'] = new_sk_state.root_translation.numpy()
        #     motion_traj['root_rotation'] = new_sk_state.global_root_rotation.numpy()
        #     motion_traj['dof'] = sRot.from_quat(new_sk_state.local_rotation[:, 1:].reshape(-1, 4)).as_rotvec().reshape(N, -1, 3)
        #     export_mujoco_video_hoi(
        #         motion_traj,
        #         obj_pos=obj_pos,
        #         obj_quat_xyzw=obj_quat_xyzw,
        #         xml_path=temp_xml,
        #         output_path=osp.join(output_dir_render, f"{key_str}.mp4"),
        #     )
        #
        #     # output_mesh_video = osp.join(output_dir_render, f"{key_str}_origin.mp4")
        #     #
        #     # #
        #     # root_aa_cam_render, trans_cam_render = apply_cam2world_rotvec_trans(
        #     #     pose_aa[:, :3],  # world root
        #     #     final_root_trans,  # world root translation（已 height-fix）
        #     #     R_world2cam_mat
        #     # )
        #     # pose_cam_render = pose_aa.copy()
        #     # pose_cam_render[:, :3] = root_aa_cam_render  # 仍然只改 root，body pose 不动
        #     #
        #     # # 2) 物体：用最终世界系的 obj_angles + obj_pos 转回 cam/Y-up
        #     # obj_angles_cam_render, obj_pos_cam_render = apply_cam2world_rotvec_trans(
        #     #     obj_angles,  # world
        #     #     obj_pos,  # world（已跟随 pelvis + height-fix）
        #     #     R_world2cam_mat
        #     # )
        #     # obj_quat_cam_render = sRot.from_rotvec(obj_angles_cam_render).as_quat()
        #     #
        #     # # 3) 调用渲染
        #     # render_smplh_hoi_video(
        #     #     smplh_layer=smplh_layer,
        #     #     poses=pose_cam_render,  # ✅ cam/Y-up，且与最终 world 一致
        #     #     trans=trans_cam_render,  # ✅ cam/Y-up，且已包含 height-fix
        #     #     betas=betas_smpl[:10],
        #     #     obj_mesh_path=obj_mesh_path,
        #     #     obj_pos=obj_pos_cam_render,  # ✅ cam/Y-up（最终物体位置）
        #     #     obj_quat_xyzw=obj_quat_cam_render,  # ✅ cam/Y-up
        #     #     output_path=output_mesh_video,
        #     #     fps=30
        #     # )
        #
        # output_dir_sequences = osp.join(output_dir, "sequences")
        # os.makedirs(output_dir_sequences, exist_ok=True)
        # save_path = osp.join(output_dir_sequences, f"{key_str}.npy")
        # os.makedirs(osp.dirname(save_path), exist_ok=True)
        # np.save(save_path, bundle, allow_pickle=True)



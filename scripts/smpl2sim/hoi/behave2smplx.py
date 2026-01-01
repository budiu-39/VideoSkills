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

from scripts.libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPLH_MUJOCO_NAMES, SMPLH_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_parser import SMPLX_Parser
from scripts.smpl2sim.hoi.mujoco_contact_inference import build_local_templates_by_body
from scripts.smpl2sim.hoi.mujoco_contact_inference import quick_viz_frame, build_sk2mj_index
from scripts.render.mujoco_render import export_mujoco_video_hoi, create_temp_xml_with_object
from scripts.render.render_smplh_hoi import render_smplh_hoi_video
from scripts.smpl2sim.hoi.smpl2sim_utils import from_yup_to_simulation
from scripts.smpl2sim.hoi.hoi_retarget_utils import (load_behave_sequence, apply_cam2world_rotvec_trans,
                                                     _quat_rotate_xyzw,
                                                     angular_velocity_world_from_quat_xyzw, penetration_depth_sequence_ig,
                                                     compute_cg_ig_via_smplh_contacts_yup, smplh_vert_part_from_custom_layer)


def from_ydown_to_yup(smpl_model, poses, trans, betas, obj_trans, obj_angles, mesh_obj):
    ''' 将 BEHAVE 数据集的 Y down 坐标系转换到 Y up 坐标系，输入输出都是 numpy 格式S '''
    rotation_matrix_x = sRot.from_euler('x', -np.pi, degrees=False)

    smplx_output = smpl_model(
        body_pose=torch.from_numpy(poses[:, 3:66]).float(),
        global_orient=torch.from_numpy(poses[:, :3]).float(),
        betas=torch.from_numpy(betas).float(),
        transl=torch.from_numpy(trans).float()
    )
    pelvis = smplx_output.joints.detach().numpy()[:, 0, :]

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
    smplx_output = smpl_model(
        body_pose=torch.from_numpy(poses[:, 3:66]).float(),
        global_orient=torch.from_numpy(poses[:, :3]).float(),
        betas=torch.from_numpy(betas).float(),
        transl=torch.from_numpy(trans).float()
    )
    verts = smplx_output.vertices.detach().numpy()
    pelvis = smplx_output.joints.detach().numpy()[:, 0, :]  # 更新后的 pelvis

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
        use_pca=False,  # 关键：不用 PCA，接受 45D/手
        create_transl=False,
        flat_hand_mean=True,
        num_betas=20  # SMPL-X 20 维 beta
    )

    for sequence_dir in tqdm(all_sequences):
        if not osp.exists(osp.join(sequence_dir, "smpl_fit_all.npz")) and not osp.exists(osp.join(sequence_dir, "human.npz")):
            continue

        norm_path = osp.normpath(sequence_dir)
        print("Processing", sequence_dir)

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

        human_yup, obj_yup = from_ydown_to_yup(smpl[gender], pose_aa_smpl, trans_smpl, betas_smpl,
                                               obj_trans, obj_angles, mesh_obj)

        new_sk_state, object_dict = from_yup_to_simulation(human_yup, obj_yup, smpl[gender],
                                                          smplx_parser_n, skeleton_tree)

        motion_dict = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30).to_dict()

        if args.render:
            render_outdir = 'renders'
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
                xml_path=temp_xml,
                output_path=osp.join(render_outdir, f"{key_str}.mp4"),
            )

        # 9. 计算接触和交互信息(在 yup 坐标系下计算，然后旋转到 Zup)

        # cg_np, ig_np = compute_cg_ig_via_smplh_contacts_yup(
        #     smplh_layer=smplh_layer,
        #     pose_aa=pose_aa_smpl,  # (T,D)
        #     betas=betas_smpl,  # (10,) 或 (1,10)
        #     trans=trans_smpl,  # (T,3)
        #     obj_mesh_path=obj_mesh_path,  # 物体mesh（局部坐标）
        #     obj_pos_world=obj_trans,  # (T,3)
        #     obj_quat_xyzw=sRot.from_rotvec(obj_angles).as_quat(),  # (T,4) xyzw
        #     smplh_vert_part=smplh_vert_part_from_custom_layer(smplh_layer),
        #     contact_threshold=0.01,
        #     samples_per_object=1024,
        # )
        #
        # ig_mj = ig_np[:, smpl_2_mujoco, :]  # (T,52,3)
        # ig_mj = ig_mj @ np.array(R_cam2world_mat).T
        # cg_mj = cg_np[:, smpl_2_mujoco]  # (T,52)

        # 穿模计算
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

        # quick_viz = True
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

        # render_smplh_hoi_video(
        #     smplh_layer=smplh_layer,
        #     poses=pose_aa_smpl,  # 原始
        #     trans=trans_smpl,
        #     betas=betas_smpl[0:1, :10],
        #     obj_mesh_path=obj_mesh_path,
        #     obj_pos=obj_trans,
        #     obj_quat_xyzw=sRot.from_rotvec(obj_angles).as_quat(),
        #     output_path=output_mesh_video,
        #     fps=30
        # )
        #
        # output_mesh_video = osp.join(output_dir_render, f"{key_str}_origin.mp4")
        #
        bundle = {
            "motion": motion_dict,  # SkeletonMotion 的 dict（含关节、根姿态、fps 等）
            "object": object_dict,
            "interaction": {
                "ig": ig_mj,  # (T,52,3) float32（世界系）
                "contact_robot": cg_mj,  # (T,52)   0/1 float32
                "collision_tag": (pen_seq > 0).any(axis=1)
            }
        }

        output_dir_sequences = osp.join(output_dir, "sequences")
        os.makedirs(output_dir_sequences, exist_ok=True)
        save_path = osp.join(output_dir_sequences, f"{key_str}.npy")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        np.save(save_path, bundle, allow_pickle=True)




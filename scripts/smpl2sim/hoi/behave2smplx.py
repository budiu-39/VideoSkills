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

from scripts.libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPLH_MUJOCO_NAMES, SMPLH_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_parser import SMPLX_Parser
from scripts.smpl2sim.hoi.mujoco_contact_inference import build_local_templates_by_body
from scripts.smpl2sim.hoi.mujoco_contact_inference import quick_viz_frame, build_sk2mj_index
from scripts.render.mujoco_render import export_mujoco_video_hoi, create_temp_xml_with_object
from scripts.render.render_smplh_hoi import render_smplh_hoi_video
from scripts.smpl2sim.hoi.hoi_retarget_utils import (load_behave_sequence, apply_cam2world_rotvec_trans,
                                                     _quat_rotate_xyzw,
                                                     angular_velocity_world_from_quat_xyzw, penetration_depth_sequence_ig,
                                                     compute_cg_ig_via_smplh_contacts_yup, smplh_vert_part_from_custom_layer)



def from_yup_to_simulation():
    return

def from_ydown_to_yup(poses, trans, betas, obj_trans, obj_angles, obj_name):
    rotation_matrix_x = Rotation.from_euler('x', -np.pi, degrees=False)

    smpl_model = smpl[gender]
    smplx_output = smpl_model(
        body_pose=torch.from_numpy(poses[:, 3:66]).float(),
        global_orient=torch.from_numpy(poses[:, :3]).float(),
        betas=torch.from_numpy(betas).float(),
        transl=torch.from_numpy(trans).float()
    )
    pelvis = smplx_output.joints.detach().numpy()[:, 0, :]

    # --- 4. 坐标系旋转 (World Transformation) ---
    # 旋转人
    rotvecs = poses[:, :3]
    rotated_rotations = rotation_matrix_x * Rotation.from_rotvec(rotvecs)
    poses[:, :3] = rotated_rotations.as_rotvec()
    trans = rotation_matrix_x.apply(trans)

    # 旋转物体 (保持相对 pelvis 关系)
    obj_trans_delta = rotation_matrix_x.apply(obj_trans - pelvis)

    rotated_rotations2 = rotation_matrix_x * Rotation.from_rotvec(obj_angles)
    obj_angles = rotated_rotations2.as_rotvec()

    # --- 5. SMPL 前向计算 (Pass 2: 旋转后) ---
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

    # --- 6. 落地校准 (Ground Alignment) ---
    # 这里只需要读取 mesh 来计算顶点位置，千万不要再 export 了！
    mesh_obj_path = os.path.join(OBJECT_CENTERED_PATH, f"{obj_name}/{obj_name}.obj")
    # process=False 确保顶点顺序和 Phase 1 处理时一致
    mesh_obj = trimesh.load(mesh_obj_path, force='mesh', process=False)

    # 计算每一帧物体的顶点世界坐标
    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    obj_verts_template = mesh_obj.vertices[None, ...]  # (1, V, 3)
    obj_verts_template -= np.mean(obj_verts_template, axis=1, keepdims=True)
    # R * V_T + T
    obj_verts_motion = np.matmul(obj_verts_template, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]

    # 计算 diff
    diff_fix = min(verts[:30, ..., 1].min(), obj_verts_motion[:30, ..., 1].min())

    # 应用落地修正
    obj_trans[..., 1] -= diff_fix
    trans[..., 1] -= diff_fix

    # --- 7. 保存 ---
    obj = {'angles': obj_angles, 'trans': obj_trans, 'name': obj_name}
    human = {'poses': poses, 'betas': betas[0], 'trans': trans, 'gender': gender}
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

    for sequence_dir in tqdm(all_sequences):
        if not osp.exists(osp.join(sequence_dir, "smpl_fit_all.npz")) and not osp.exists(osp.join(sequence_dir, "human.npz")):
            continue

        norm_path = osp.normpath(sequence_dir)
        parent_name = osp.basename(osp.dirname(norm_path))

        print("Processing", sequence_dir)

        # 导入人体 SMPL 数据，此时是相机坐标系(Y down)，先把它变成 渲染通用的坐标系(Y up)，再转化成 Z up。
        # 只渲染 Y up 和 Z up
        sequence_data = load_behave_sequence(sequence_dir)
        smpl_data = sequence_data['smpl']
        pose_aa_smpl = smpl_data['poses']
        trans_smpl = smpl_data['trans']
        betas_smpl = torch.from_numpy(smpl_data.get('betas').copy())
        gender = sequence_data['smpl']['gender']

        # 初始化 SMPL-H 模型（人体点云）
        model_root = 'data/smplh'
        smplh_layer = SMPL_Layer(center_idx=0, gender=gender, num_betas=10,
                                 model_root=str(model_root), hands=True)#.to('cuda')

        # 导入物体数据
        obj_angles = sequence_data.get('object', {}).get('angles', None)
        obj_trans = sequence_data.get('object', {}).get('trans', None)
        obj_times = sequence_data.get('object', {}).get('frame_times', None)
        key_str = os.path.basename(os.path.normpath(sequence_dir))
        object_name_str = key_str.split('_')[2]

        obj_name = object_name_str
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


        # 先把 Behave 从 Y down 转成 Y up，并修正高度
        with torch.no_grad():
            # 准备全序列数据进行批处理
            p_all = torch.from_numpy(pose_aa_smpl).float()
            t_all = torch.from_numpy(trans_smpl).float()
            # 注意：SMPL-H 层通常需要 betas 维度匹配
            b_all = betas_smpl[0:1, :10].expand(p_all.shape[0], -1)

            # 得到相机系下的 (verts, joints) 轨迹
            # 根据之前的报错，确保解包正确
            out_cam = smplh_layer(p_all, th_betas=b_all, th_trans=t_all)
            joints_cam_all = out_cam[1]  # (T, J, 3)
            pelvis_cam_all = joints_cam_all[:, 0].cpu().numpy()  # (T, 3) 轨迹

        # 3. 计算每一帧的相对位移并旋转
        # rel_pelvis_obj_cam: 每一帧物体中心相对于当前 Pelvis 的向量
        rel_pelvis_ydonw = obj_trans - pelvis_cam_all  # (T, 3)

        R_ydown2yup = sRot.from_euler('x', np.pi, degrees=False).as_matrix()

        # 人体相对位移旋转
        obj_trans_delta = sRot.from_matrix(R_ydown2yup).apply(rel_pelvis_ydonw)

        pose_aa_yup = pose_aa_smpl.copy()
        pose_aa_yup[:, :3], tran_yup= apply_cam2world_rotvec_trans(
            pose_aa_ydown[:, :3], tran_ydown, R_ydown2yup
        )

        obj_trans_yup = tran_ydown + rel_pelvis_obj_yup
        obj_angels_yup = sRot.from_matrix(R_ydown2yup).apply(obj_angles)

        # 最后调整高度








        # 1. 定义坐标系变换矩阵
        # R_cam2world_mat = sRot.from_euler('x', -np.pi / 2, degrees=False).as_matrix()

        R_cam2world_mat = sRot.from_euler('x', -np.pi, degrees=False).as_matrix()

        # 2. 【关键修复】获取相机系下 Pelvis 的完整轨迹 (T, 3)
        with torch.no_grad():
            # 准备全序列数据进行批处理
            p_all = torch.from_numpy(pose_aa_smpl).float()
            t_all = torch.from_numpy(trans_smpl).float()
            # 注意：SMPL-H 层通常需要 betas 维度匹配
            b_all = betas_smpl[0:1, :10].expand(p_all.shape[0], -1)

            # 得到相机系下的 (verts, joints) 轨迹
            # 根据之前的报错，确保解包正确
            out_cam = smplh_layer(p_all, th_betas=b_all, th_trans=t_all)
            joints_cam_all = out_cam[1]  # (T, J, 3)
            pelvis_cam_all = joints_cam_all[:, 0].cpu().numpy()  # (T, 3) 轨迹

        # 3. 计算每一帧的相对位移并旋转
        # rel_pelvis_obj_cam: 每一帧物体中心相对于当前 Pelvis 的向量
        rel_pelvis_obj_cam = obj_trans - pelvis_cam_all  # (T, 3)
        # 将位移向量轨迹整体旋转到世界系
        rel_pelvis_obj_world = sRot.from_matrix(R_cam2world_mat).apply(rel_pelvis_obj_cam)

        # 4. 处理人体旋转（获取世界系下的 Origin 轨迹）
        pose_aa = pose_aa_smpl.copy()
        pose_aa[:, :3], root_trans_origin_world = apply_cam2world_rotvec_trans(
            pose_aa_smpl[:, :3], trans_smpl, R_cam2world_mat
        )

        # 6. 高度修正，这里直接使用 SMPL-X Parser（等效于 来计算最低点
        # actual_pelvis_offset = skeleton_tree.local_translation[0].numpy()  # 这个 offset 是 Z up 下的
        diff_fix = 0
        with torch.no_grad():
            f_check = min(100, pose_aa.shape[0])
            p_t = torch.from_numpy(pose_aa[:f_check]).float()
            t_t = torch.from_numpy(root_trans_origin_world[:f_check]).float()
            verts, _ = smplx_parser_n.get_joints_verts(p_t, torch.zeros((1, 20)), t_t)
            diff_fix = verts[..., -1].min().item()

        # 目前的思路是这样的
        # 1. 计算物体相对人体的旋转
        # 2. 把人体旋转到正确的坐标系（root 细节不重要）
        # 3. 把这个 root 直接作为新的 pelvis，然后由于 Motionstate 的 root 是 pelvis，所以直接相对 root 调整最低点到地面即可

        final_root_trans = root_trans_origin_world - diff_fix

        # final_obj_pos: 物体跟随 Pelvis，应用相同的地面修正
        obj_pos = final_root_trans + rel_pelvis_obj_world

        # 7. 物体旋转 (独立处理，只需旋转角度)
        obj_angles_w, _ = apply_cam2world_rotvec_trans(
            obj_angles, obj_trans, R_cam2world_mat
        )

        # 计算物体最终结果
        obj_quat_xyzw = sRot.from_rotvec(obj_angles_w).as_quat().astype(np.float32)
        q_xyzw = torch.from_numpy(obj_quat_xyzw.astype(np.float32)).to(device)  # 移动到 device
        p_w = torch.from_numpy(obj_pos.astype(np.float32)).to(device)  # 移动到 device

        dt = 1.0 / 30
        obj_pos_vel = np.zeros_like(obj_pos)
        obj_pos_vel[1:] = (obj_pos[1:] - obj_pos[:-1]) / dt
        obj_vel_np = obj_pos_vel.astype(np.float32)  # [T,3]
        obj_acc_np = np.zeros_like(obj_vel_np, dtype=np.float32)
        obj_acc_np[1:] = (obj_vel_np[1:] - obj_vel_np[:-1]) / dt

        obj_rot_vel = angular_velocity_world_from_quat_xyzw(obj_quat_xyzw, dt)

        # 8. 格式化成合适的 SkeletonMotion 格式
        N = pose_aa.shape[0]
        pose_aa_mj = pose_aa.reshape(N, 52, 3)
        smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa_mj[:, smpl_2_mujoco]

        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 52, 4)
        # 轴角 -> 四元数（注意 scipy 返回 [x,y,z,w]）
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            torch.from_numpy(final_root_trans).float(),  # 显式转为 Tensor
            is_local=True)

        # 局部坐标系旋转（这个是骨架区别）
        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) *
                                sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                                            torch.from_numpy(pose_quat_global),
                                                                            torch.from_numpy(final_root_trans).float(),
                                                                            is_local=False)

        motion_dict = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30).to_dict()

        # 9. 计算接触和交互信息
        cg_np, ig_np = compute_cg_ig_via_smplh_contacts_yup(
            smplh_layer=smplh_layer,
            pose_aa=pose_aa_smpl,  # (T,D)
            betas=betas_smpl,  # (10,) 或 (1,10)
            trans=trans_smpl,  # (T,3)
            obj_mesh_path=obj_mesh_path,  # 物体mesh（局部坐标）
            obj_pos_world=obj_trans,  # (T,3)
            obj_quat_xyzw=sRot.from_rotvec(obj_angles).as_quat(),  # (T,4) xyzw
            smplh_vert_part=smplh_vert_part_from_custom_layer(smplh_layer),
            contact_threshold=0.01,
            samples_per_object=1024,
        )

        ig_mj = ig_np[:, smpl_2_mujoco, :]  # (T,52,3)
        ig_mj = ig_mj @ np.array(R_cam2world_mat).T
        cg_mj = cg_np[:, smpl_2_mujoco]  # (T,52)

        # 穿模计算
        body_clouds, body_geoms, mj_model = build_local_templates_by_body(
            "data/robots/smpl/smplx_humanoid_hand.xml",
            samples_per_geom=500)
        mj_data = mujoco.MjData(mj_model)
        sk2mj, mj2sk = build_sk2mj_index(mj_model, skeleton_tree, drop_world=True)

        body_pos = new_sk_state.global_translation
        obj_pts_world = _quat_rotate_xyzw(q_xyzw, object_points) + p_w.unsqueeze(1)
        obj_pts_world_np = obj_pts_world.cpu().numpy().astype(np.float32)  # [T,P,3]

        pen_seq = penetration_depth_sequence_ig(
            mj_model,
            body_geoms,
            mj2sk,
            obj_pts_world_np,  # e.g. List[np.ndarray], len T, each (P_t,3)
            body_pos.numpy().astype(np.float32),  # (T, B, 3)
            new_sk_state.global_rotation.numpy().astype(np.float32),  # (T, B, 4) xyzw
        )

        bundle = {
            "motion": motion_dict,  # SkeletonMotion 的 dict（含关节、根姿态、fps 等）
            "object": {
                "name": obj_name,
                "obj_pos": obj_pos,
                "obj_rot": obj_quat_xyzw,  # xyzw —— Isaac Gym 对齐
                "obj_pos_vel": obj_pos_vel,
                "obj_rot_vel": obj_rot_vel,
            },
            "interaction": {
                "ig": ig_mj,  # (T,52,3) float32（世界系）
                "contact_robot": cg_mj,  # (T,52)   0/1 float32
                "collision_tag": (pen_seq > 0).any(axis=1)
            }
        }


        quick_viz = True
        if quick_viz:
            body_pos_t = new_sk_state.global_translation  # Tensor, 与上文一致
            t = min(100, obj_pts_world_np.shape[0] - 1)

            body_pos_frame = body_pos_t[t].cpu().numpy().astype(np.float64)
            body_rot_frame = new_sk_state.global_rotation[t].cpu().numpy().astype(np.float64)

            quick_viz_frame(
                mj_model, mj_data,
                body_local_clouds=body_clouds,
                obj_pts=obj_pts_world_np[t],
                body_rot_frame=body_rot_frame,
                mj2sk=mj2sk,
                title=f"seq:{key_str} t={t}",
                body_pos_frame=body_pos_frame,
                ig_frame=ig_mj[t],
                contact_row=cg_mj[t],
            )

        args.render = True
        if args.render:
            output_dir_render = osp.join(output_dir, "rendered")
            os.makedirs(output_dir_render, exist_ok=True)
            temp_xml = create_temp_xml_with_object(
                "data/robots/smpl/smplx_humanoid_hand.xml",
                obj_mesh_path
            )
            motion_traj = {}
            motion_traj['root_trans_offset'] = new_sk_state.root_translation.numpy()
            motion_traj['root_rotation'] = new_sk_state.global_root_rotation.numpy()
            motion_traj['dof'] = sRot.from_quat(new_sk_state.local_rotation[:, 1:].reshape(-1, 4)).as_rotvec().reshape(N, -1, 3)
            export_mujoco_video_hoi(
                motion_traj,
                obj_pos=obj_pos,
                obj_quat_xyzw=obj_quat_xyzw,
                xml_path=temp_xml,
                output_path=osp.join(output_dir_render, f"{key_str}.mp4"),
            )

            output_mesh_video = osp.join(output_dir_render, f"{key_str}_origin.mp4")

            # 调用函数
            render_smplh_hoi_video(
                smplh_layer=smplh_layer,
                poses=pose_aa_smpl,  # 原始
                trans=trans_smpl,
                betas=betas_smpl[0:1, :10],
                obj_mesh_path=obj_mesh_path,
                obj_pos=obj_trans,
                obj_quat_xyzw=sRot.from_rotvec(obj_angles).as_quat(),
                output_path=output_mesh_video,
                fps=30
            )

        output_dir_sequences = osp.join(output_dir, "sequences")
        os.makedirs(output_dir_sequences, exist_ok=True)
        save_path = osp.join(output_dir_sequences, f"{key_str}.npy")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        np.save(save_path, bundle, allow_pickle=True)




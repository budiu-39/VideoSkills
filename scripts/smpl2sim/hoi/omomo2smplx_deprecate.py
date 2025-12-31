import os
import sys
import os.path as osp

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm
import argparse

from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPLH_MUJOCO_NAMES, SMPLH_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_parser import SMPLX_Parser
from scripts.smpl2sim.hoi.mujoco_contact_inference import build_local_templates_by_body, contacts_from_xml_pointcloud
from scripts.smpl2sim.hoi.mujoco_contact_inference import penetration_depth_sequence_ig
from scripts.smpl2sim.hoi.mujoco_contact_inference import build_qpos_seq_from_state, quick_viz_frame, build_sk2mj_index
from scripts.render.mujoco_render import vis_mujoco_hoi, create_temp_xml_with_object

from scripts.smpl2sim.hoi.hoi_retarget_utils import (load_behave_sequence, apply_cam2world_rotvec_trans,
                                                     _quat_rotate_xyzw, create_temp_xml_with_object, vis_mujoco_hoi,
                                                     angular_velocity_world_from_quat_xyzw, penetration_depth_sequence_ig,
                                                     compute_cg_ig_via_smplh_contacts, smplh_vert_part_from_custom_layer)
import trimesh
import joblib
import torch
import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scripts.libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
import shutil




def get_omomo_data(raw_p_path):
    print(f"Loading OMOMO data from {raw_p_path}...")
    return joblib.load(raw_p_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--src", type=str, default="", help="Path to BEHAVE dataset")
    parser.add_argument("--obj_root", type=str, default="./data/omomo/objects", help="Path to OMOMO objects")
    parser.add_argument("--render", action="store_true", default=False, help="Whether to render the \
                                                                        retargeted motion using scenepic animation.")
    parser.add_argument("--dst", type=str, default="dataset/smplx_motion/behave_small", help="Output path")
    args = parser.parse_args()
    output_dir = args.dst

    OBJECT_PATH = "./data/omomo/objects_scaled"

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

    smplx_parser_n = SMPLX_Parser(
        model_path='data/SMPL/smplx',
        gender='neutral',
        use_pca=False,  # 关键：不用 PCA，接受 45D/手
        create_transl=False,
        flat_hand_mean=True,
        num_betas=20  # SMPL-X 20 维 beta
    )

    skipped_due_to_first_frame_collision = []

    sequences_folder_names = []

    for seq_key in tqdm(data_dict.keys()):
        entry = data_dict[seq_key]

        # norm_path = osp.normpath(sequence_dir)
        # parent_name = osp.basename(osp.dirname(norm_path))

        # print("Processing", sequence_dir)


        pose_aa_smpl = entry['pose_body']
        global_aa = entry['root_orient']
        hand_aa = np.zeros((pose_aa_smpl.shape[0], 90), dtype=np.float32)
        pose_aa_smpl = np.concatenate([global_aa, pose_aa_smpl, hand_aa], axis=1)
        trans_smpl = entry['obj_trans'][:, :, 0]
        betas_smpl = torch.from_numpy(entry['betas'][0][:10])
        gender = entry.get('gender', 'neutral')

        N = pose_aa_smpl.shape[0]
        if N < 10:
            print(f"Sequence too short ({N} frames), skipping")
            continue

        # 模型
        D = pose_aa_smpl.shape[1]

        # 基于 SMPL 的修正,
        # model = smplx.create(
        #     model_path='data/SMPL',  # 包含 SMPL/SMPLX 模型文件的目录
        #     model_type='smpl',  # 或 'smplx' / 'smplh'，取决于你的模型
        #     gender='neutral',  # male/female/neutral
        #     use_pca=False
        # )

        # 2. 创建全 0 参数
        betas = torch.zeros([1, 10])  # 形状参数（shape）
        # body_pose = torch.zeros([1, 69])  # 姿态参数（23*3）
        # global_orient = torch.zeros([1, 3])  # 全局旋转
        # transl = torch.zeros([1, 3])  # 平移
        # output = model(
        #     betas=betas,
        #     body_pose=body_pose,
        #     global_orient=global_orient,
        #     transl=transl
        # )

        # 4. 导出关节或顶点位置
        # joints = output.joints.detach().cpu().numpy()  # (1, N_joints, 3)
        # vertices = output.vertices.detach().cpu().numpy()  # (1, 6890, 3)


        # 世界坐标系旋转
        # R_cam2world = [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]
        # R_cam2world = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
        # R_cam2world = sRot.from_euler('x', -np.pi/2, degrees=False).as_matrix()
        # 从相机坐标（y 向下，z 向后） 到 y 变 -z（y 是高度，也就是说新坐标系里 z 是高度）, z 变 y 的坐标系（y 是前后）
        # R_cam2world = [[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]
        # R_cam2world = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        world_shift = np.zeros(3, dtype=np.float32)

        # 这个距离是 z up 时的 z 轴，所以应该在 z up 时做，而且如果后续有旋转的话，应该在旋转前做！
        pose_aa = pose_aa_smpl.copy()

        root_trans = trans_smpl - skeleton_tree_smpl.local_translation[0].numpy()
        # pose_aa[:, :3], root_trans_offset = apply_cam2world_rotvec_trans(pose_aa_smpl[:, :3], root_trans, R_cam2world)


        root_trans_offset = torch.from_numpy(root_trans).float()
        # 关节重排
        pose_aa_mj = pose_aa.reshape(N, 52, 3)
        smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa_mj[:, smpl_2_mujoco]

        # 轴角 -> 四元数（注意 scipy 返回 [x,y,z,w]）
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 52, 4)
        # root_trans_offset = torch.zeros(N, 3) + root_trans
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True)

        frame_check = 100
        height_tolorance = 0
        fix_height = True  # 开启更稳妥
        if fix_height:
            with torch.no_grad():
                frame_check = min(frame_check, N)
                pose_t = torch.from_numpy(pose_aa[:frame_check]).float()  # (F,72) 关键！
                beta_np = np.zeros(20, dtype=np.float32)  # 10 维 betas
                beta_t = torch.from_numpy(beta_np[None, ...]).float()  # (1,10)
                trans_t = root_trans_offset[:frame_check].float()  # (F,3)

                verts, joints = smplx_parser_n.get_joints_verts(pose_t, beta_t, trans_t)
                offset = joints[:, 0] - trans_t
                feet_z = (verts - offset[:, None])[..., -1]
                diff_fix = feet_z.min().item()
                root_trans_offset[..., -1] -= diff_fix
                world_shift -= np.array([0, 0, diff_fix], dtype=np.float32)

        # 局部坐标系旋转（这个是骨架区别）
        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) *
                                sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                                            torch.from_numpy(pose_quat_global),
                                                                            root_trans_offset, is_local=False)
        fps = 30
        motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30)

        # 物体修正(Behave 世界坐标系 -> Isaac Gym 世界坐标系)
        seq_name = entry['seq_name']
        obj_name = seq_name.split("_")[1]
        obj_angles = sRot.from_matrix(entry['obj_rot']).as_rotvec()
        obj_trans = entry['obj_trans'][:, :, 0]

        # key_str = os.path.basename(os.path.normpath(sequence_dir))
        # object_name_str = key_str.split('_')[2]

        model_root = 'data/smplh'
        smplh_layer = SMPL_Layer(center_idx=0, gender=gender, num_betas=10,
                               model_root=str(model_root), hands=True).to('cuda')

        # obj_name = object_name_str
        obj_mesh_path = osp.join(OBJECT_PATH, obj_name, f"{obj_name}.obj")
        mesh_obj = trimesh.load(obj_mesh_path, force='mesh')
        obj_points, _ = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2025)

        # obj_angles_w, obj_trans_w = apply_cam2world_rotvec_trans(obj_angles, obj_trans, R_cam2world)
        obj_quat_xyzw = sRot.from_rotvec(obj_angles).as_quat().astype(np.float32)


        # 位置
        obj_pos = obj_trans.astype(np.float32)
        obj_pos = (obj_pos + world_shift[None, :]).astype(np.float32)
        # 速度
        dt = 1.0 / fps
        obj_pos_vel = np.zeros_like(obj_pos)
        obj_pos_vel[1:] = (obj_pos[1:] - obj_pos[:-1]) / dt

        # 角速度
        obj_rot_vel = angular_velocity_world_from_quat_xyzw(obj_quat_xyzw, dt)

        # key_str = os.path.basename(os.path.normpath(sequence_dir))
        # object_name_str = key_str.split('_')[2]
        motion_dict = motion.to_dict()  # 与 motion.to_file() 里保存的结构一致

        # ==== 推理接触 ====
        # 1) 准备 body_pos (世界坐标): [T,52,3]
        body_pos_t = new_sk_state.global_translation  # Tensor, 与上文一致
        if not isinstance(body_pos_t, torch.Tensor):
            body_pos_t = torch.from_numpy(body_pos_t)
        body_pos_t = body_pos_t.to(torch.float32)  # [T,52,3]

        q_xyzw = torch.from_numpy(obj_quat_xyzw.astype(np.float32))  # [T,4]
        p_w = torch.from_numpy(obj_pos.astype(np.float32))  # [T,3]
        dt = 1.0 / fps

        obj_points_local = torch.from_numpy(
            trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2025)[0].astype(np.float32))
        obj_pts_world = _quat_rotate_xyzw(q_xyzw, obj_points_local) + p_w.unsqueeze(1)  # [T,P,3]

        with torch.no_grad():
            # obj_pts_world 已在上文计算
            ig_torch = compute_sdf(body_pos_t, obj_pts_world)  # [T,52,3]
            ref_ig = ig_torch.cpu().numpy().astype(np.float32)

        body_clouds, body_geoms, mj_model = build_local_templates_by_body("data/robots/smpl/smplx_humanoid_hand.xml",
                                                                samples_per_geom=500)

        qpos_seq_np = build_qpos_seq_from_state(mj_model, new_sk_state)

        obj_pts_world_np = obj_pts_world.cpu().numpy().astype(np.float32)  # [T,P,3]
        obj_pos_np = p_w.cpu().numpy().astype(np.float32)  # [T,3]
        obj_vel_np = obj_pos_vel.astype(np.float32)  # [T,3]
        obj_acc_np = np.zeros_like(obj_vel_np, dtype=np.float32)
        obj_acc_np[1:] = (obj_vel_np[1:] - obj_vel_np[:-1]) / dt

        sk2mj, mj2sk = build_sk2mj_index(mj_model, skeleton_tree, drop_world=True)
        names = np.array([str(n) for n in skeleton_tree.node_names])  # or skeleton_tree.body_names
        # 你想标记为“地面/脚/胸”的集合（名字要与 XML/Skeleton 一致）
        mask_names = {"L_Ankle", "R_Ankle", "L_Toe", "R_Toe"}
        # Skeleton 顺序的布尔掩码：True 表示该 body 属于“接地”集合
        ground_mask_sk = np.isin(names, list(mask_names))
        # 调 contacts + ig
        # contact_robot, ig_np = contacts_from_xml_pointcloud(
        #     mj_model, body_clouds,
        #     qpos_seq=qpos_seq_np.astype(np.float32),
        #     obj_pts_world=obj_pts_world_np,
        #     obj_pos=obj_pos_np, vel=obj_vel_np, acc=obj_acc_np,
        #     sigma_pad=ADAPTIVE_PAD, sigma_no_interact=FIXED_SIGMA_NO_INTERACT,
        #     ground_height=0.0,
        #     sk2mj=sk2mj, mj2sk=mj2sk,
        #     ig_body_pos_world = body_pos_t.numpy().astype(np.float32),
        #     ig_body_rot_world = new_sk_state.global_rotation.numpy().astype(np.float32),
        #     ground_mask_sk=ground_mask_sk,
        # )

        pen_seq = penetration_depth_sequence_ig(
            mj_model,
            body_geoms,
            mj2sk,
            obj_pts_world_np,  # e.g. List[np.ndarray], len T, each (P_t,3)
            body_pos_t.numpy().astype(np.float32),  # (T, B, 3)
            new_sk_state.global_rotation.numpy().astype(np.float32),  # (T, B, 4) xyzw
        )

        first_frame_collided = (pen_seq[0] > 0).any()
        if first_frame_collided:
            # key_str = os.path.basename(os.path.normpath(sequence_dir))  # 样本名
            print(f"[SKIP-FIRST-FRAME-COLLISION] {seq_key}")
            skipped_due_to_first_frame_collision.append(seq_key)
            continue  # 直接跳到下一个 sequence


        cg_np, ig_np = compute_cg_ig_via_smplh_contacts(
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
        # ig_mj = ig_mj @ np.array(R_cam2world).T
        cg_mj = cg_np[:, smpl_2_mujoco]  # (T,52)


        # body_ids_wo_foot_ankel = np.where(~ground_mask_sk)[0]

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

        # === 可视化
        t = min(100, obj_pts_world_np.shape[0] - 1)
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = qpos_seq_np[t]
        mujoco.mj_forward(mj_model, mj_data)

        body_pos_frame = body_pos_t[t].cpu().numpy().astype(np.float64)
        body_rot_frame = new_sk_state.global_rotation[t].cpu().numpy().astype(np.float64)
        # body_pos_frame = mj_data.xpos[sk2mj].copy()  # 严格用 MJCF body 原点
        quick_viz = True

        if quick_viz:
            quick_viz_frame(
                mj_model, mj_data,
                body_local_clouds=body_clouds,
                obj_pts=obj_pts_world_np[t],
                contact_row=cg_mj[t],
                body_rot_frame=body_rot_frame,
                mj2sk=mj2sk,
                title=f"seq:{seq_key} t={t}",
                body_pos_frame=body_pos_frame,
                ig_frame=ig_mj[t],
            )

        if args.render:
            temp_xml = create_temp_xml_with_object(
                "data/robots/smpl/smplx_humanoid_hand.xml",
                obj_mesh_path
            )
            motion_traj = {}
            motion_traj['root_trans_offset'] = new_sk_state.root_translation.numpy()
            motion_traj['root_rotation'] = new_sk_state.global_root_rotation.numpy()
            motion_traj['dof'] = sRot.from_quat(new_sk_state.local_rotation[:, 1:].reshape(-1, 4)).as_rotvec().reshape(N, -1, 3)
            vis_mujoco_hoi(
                motion_traj,
                obj_pos=obj_pos,
                obj_quat_xyzw=obj_quat_xyzw,
                xml_path=temp_xml
            )

        # Construct save path
        # rel_path = osp.relpath(args.src)
        save_path = osp.join(output_dir, f"{seq_key}.npy")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        np.save(save_path, bundle, allow_pickle=True)

    #     sequences_folder_names.append(parent_name)
    #
    # with open("parent_dirs.txt", "w") as f:
    #     # 如果只需要唯一的目录名，可以使用 set(parent_folder_names)
    #     for item in sequences_folder_names:
    #         f.write(f"{item}\n")


    print("Done")


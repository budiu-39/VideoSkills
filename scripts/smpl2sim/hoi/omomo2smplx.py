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
import shutil
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
from scripts.smpl2sim.hoi.render_test import render_smplx_hoi_video_zup

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
    obj_angles = sRot.from_matrix(obj_rot).as_rotvec()
    obj_scale = seq_data['obj_scale']  # T X 1
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

    # 初始化 SMPL-H 模型（人体点云）
    SMPLX_PATH = 'data/SMPL/smplx'
    dmpl_fname = None
    num_dmpls = None
    num_expressions = None
    num_betas = 16
    surface_model_male_fname = os.path.join(SMPLX_PATH, "SMPLX_MALE.npz")
    surface_model_female_fname = os.path.join(SMPLX_PATH, "SMPLX_FEMALE.npz")
    surface_model_neutral_fname = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")
    smpl_model_male = BodyModel(bm_fname=surface_model_male_fname,
                                num_betas=num_betas,
                                num_expressions=num_expressions,
                                num_dmpls=num_dmpls,
                                dmpl_fname=dmpl_fname)
    smpl_model_female = BodyModel(bm_fname=surface_model_female_fname,
                                  num_betas=num_betas,
                                  num_expressions=num_expressions,
                                  num_dmpls=num_dmpls,
                                  dmpl_fname=dmpl_fname)
    smpl_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
                                   num_betas=num_betas,
                                   num_expressions=num_expressions,
                                   num_dmpls=num_dmpls,
                                   dmpl_fname=dmpl_fname)
    smpl = {'male': smpl_model_male, 'female': smpl_model_female, 'neutral': smpl_model_neutral}

    for seq_key in tqdm(data_dict.keys()):
        human, obj = omomo_preprocess(data_dict[seq_key])
        entry = data_dict[seq_key]

        # 导入人体 SMPL 数据
        key_str = data_dict[seq_key]['seq_name']
        pose_zup = human['poses']  #pose zup 只是朝向为 zup，相对旋转还是 yup
        tran_zup = human['trans']
        betas_smpl = torch.as_tensor(human['betas'])
        gender =  str(human['gender'])


        # 导入物体数据
        seq_name = entry['seq_name']
        obj_name = obj['name']
        obj_angles = sRot.from_matrix(obj['rot']).as_rotvec()
        obj_trans = obj['trans']


        obj_mesh_path = osp.join(OBJECT_PATH, obj_name, f"{obj_name}.obj")
        mesh_obj = trimesh.load(obj_mesh_path, force='mesh')

        output_mesh_video = f"renders/{seq_name}_{obj_name}_zup.mp4"
        os.makedirs(os.path.dirname(output_mesh_video), exist_ok=True)

        render_smplx_hoi_video_zup(
            smplx_model=smpl[gender].to(device),
            poses_zup=pose_zup,  # (T, 156)
            trans_zup=tran_zup,  # (T, 3)
            betas=betas_smpl,  # (10,)
            obj_mesh_path=obj_mesh_path,
            obj_trans_zup=obj_trans,
            obj_rotmat_zup=obj['rot'],
            output_path=output_mesh_video,
            fps=30
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

        # 1. 定义坐标系变换矩阵
        R_cam2world_mat = sRot.from_euler('x', 90, degrees=True).as_matrix()
        R_world2cam_mat = sRot.from_euler('x', -90, degrees=True).as_matrix()

        # 2. 构造相机系 Pose (Y-up) —— 用于后续接触/交互计算（函数名 *_yup）
        #    注意：这里只转换 root（global_orient）和 trans，body pose 绝对不要做坐标系旋转。
        with torch.no_grad():
            root_aa_cam, trans_smpl = apply_cam2world_rotvec_trans(
                pose_zup[:, :3], tran_zup, R_world2cam_mat
            )
            pose_aa_smpl = pose_zup.copy()
            pose_aa_smpl[:, :3] = root_aa_cam

        model_root = 'data/smplh'
        smplh_layer = SMPL_Layer(center_idx=0, gender=gender, num_betas=10,
                                 model_root=str(model_root), hands=True).to('cuda')

        # 3. 在世界系(Z-up)下计算 Pelvis 轨迹，用于稳定地计算物体相对 Pelvis 的位移
        with torch.no_grad():
            p_world = torch.from_numpy(pose_zup).float().to(device)
            t_world = torch.from_numpy(tran_zup).float().to(device)
            b_world = betas_smpl[:10].unsqueeze(0).repeat(p_world.shape[0], 1).to(device)

            out_world = smplh_layer(p_world, th_betas=b_world, th_trans=t_world)
            pelvis_world_all = out_world[1][:, 0].detach().cpu().numpy()  # (T,3) # (T,3)

        # 4. 计算每一帧物体相对于 Pelvis 的位移（世界系/Z-up）
        rel_pelvis_obj_world = obj_trans - pelvis_world_all

        # 4. 处理人体旋转（获取世界系下的轨迹）
        # 现在 pose_aa_smpl 已定义，可以 copy 了
        pose_aa = pose_aa_smpl.copy()
        pose_aa[:, :3], root_trans_origin_world = apply_cam2world_rotvec_trans(
            pose_aa_smpl[:, :3], trans_smpl, R_cam2world_mat
        )

        # 5. 算出世界系下 Origin 到 Pelvis 的固定偏移（对于固定 beta，这是常量，取一帧即可）
        # 这里其实没有意义，不管怎么旋转 pelvis offset 都是固定的，吗？
        with torch.no_grad():
            p_w0 = torch.from_numpy(pose_aa[0:1]).float()
            t_w0 = torch.from_numpy(root_trans_origin_world[0:1]).float()
            _, joints_world = smplx_parser_n.get_joints_verts(p_w0, torch.zeros((1, 20)), t_w0)
            actual_pelvis_offset = (joints_world[0, 0] - t_w0[0]).cpu().numpy()

        # 6. 地面校准 (Height Fix)
        diff_fix = 0
        with torch.no_grad():
            f_check = min(100, pose_aa.shape[0])
            p_t = torch.from_numpy(pose_aa[:f_check]).float()
            t_t = torch.from_numpy(root_trans_origin_world[:f_check]).float()
            verts, _ = smplx_parser_n.get_joints_verts(p_t, torch.zeros((1, 20)), t_t)
            diff_fix = verts[..., -1].min().item()

        # 7. 应用最终修正：得到最终 Pelvis 和物体位置
        # final_root_trans: 机器人根节点 (Pelvis) 的最终世界坐标轨迹
        final_root_trans = (root_trans_origin_world + actual_pelvis_offset) - diff_fix

        # final_obj_pos: 物体跟随 Pelvis，应用相同的地面修正
        obj_pos = final_root_trans + rel_pelvis_obj_world

        # 计算物体最终结果
        obj_quat_xyzw = sRot.from_rotvec(obj_angles).as_quat().astype(np.float32)
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
        # compute_cg_ig_via_smplh_contacts_yup 期望输入在 Y-up/camera 坐标系下：
        # - pose_aa_smpl / trans_smpl 已经是 Y-up
        # - 这里把物体从 world(Z-up) 转回 camera(Y-up)，避免坐标系混用导致物体“横着/翻倒”
        obj_angles_cam, obj_trans_cam = apply_cam2world_rotvec_trans(
            obj_angles, obj_trans, R_world2cam_mat
        )
        obj_quat_cam_xyzw = sRot.from_rotvec(obj_angles_cam).as_quat()

        cg_np, ig_np = compute_cg_ig_via_smplh_contacts_yup(
            smplh_layer=smplh_layer,
            pose_aa=pose_aa_smpl,  # (T,D)
            betas=betas_smpl,  # (10,) 或 (1,10)
            trans=trans_smpl,  # (T,3)
            obj_mesh_path=obj_mesh_path,  # 物体mesh（局部坐标）
            obj_pos_world=obj_trans_cam,  # (T,3) camera/Y-up
            obj_quat_xyzw=obj_quat_cam_xyzw,  # (T,4) xyzw camera/Y-up
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


        quick_viz = False
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

            # output_mesh_video = osp.join(output_dir_render, f"{key_str}_origin.mp4")
            #
            # #
            # root_aa_cam_render, trans_cam_render = apply_cam2world_rotvec_trans(
            #     pose_aa[:, :3],  # world root
            #     final_root_trans,  # world root translation（已 height-fix）
            #     R_world2cam_mat
            # )
            # pose_cam_render = pose_aa.copy()
            # pose_cam_render[:, :3] = root_aa_cam_render  # 仍然只改 root，body pose 不动
            #
            # # 2) 物体：用最终世界系的 obj_angles + obj_pos 转回 cam/Y-up
            # obj_angles_cam_render, obj_pos_cam_render = apply_cam2world_rotvec_trans(
            #     obj_angles,  # world
            #     obj_pos,  # world（已跟随 pelvis + height-fix）
            #     R_world2cam_mat
            # )
            # obj_quat_cam_render = sRot.from_rotvec(obj_angles_cam_render).as_quat()
            #
            # # 3) 调用渲染
            # render_smplh_hoi_video(
            #     smplh_layer=smplh_layer,
            #     poses=pose_cam_render,  # ✅ cam/Y-up，且与最终 world 一致
            #     trans=trans_cam_render,  # ✅ cam/Y-up，且已包含 height-fix
            #     betas=betas_smpl[:10],
            #     obj_mesh_path=obj_mesh_path,
            #     obj_pos=obj_pos_cam_render,  # ✅ cam/Y-up（最终物体位置）
            #     obj_quat_xyzw=obj_quat_cam_render,  # ✅ cam/Y-up
            #     output_path=output_mesh_video,
            #     fps=30
            # )

        output_dir_sequences = osp.join(output_dir, "sequences")
        os.makedirs(output_dir_sequences, exist_ok=True)
        save_path = osp.join(output_dir_sequences, f"{key_str}.npy")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        np.save(save_path, bundle, allow_pickle=True)



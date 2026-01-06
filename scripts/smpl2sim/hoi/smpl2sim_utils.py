import os
import sys

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import torch
import numpy as np
import mujoco

from scripts.poselib.skeleton.skeleton3d import SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPLH_MUJOCO_NAMES, SMPLH_BONE_ORDER_NAMES

from scripts.smpl2sim.hoi.hoi_retarget_utils import angular_velocity_world_from_quat_xyzw


def tranfrom_to_yup(smpl_model, human, obj, mesh_obj, origin_format):
    ''' 将数据集的 Y down(相机），Z up (OMOMO/AMASS/动捕) 坐标系转换到 Y up 坐标系，输入输出都是 numpy 格式 '''
    device = next(smpl_model.parameters()).device  # 自动获取模型所在设备

    if origin_format == 'ydown':
        # BEHAVE 逻辑：
        # 1. RotX(-pi) 把 Y-down 转为 Y-up。此时人脸朝向从 -Z 变成了 +Z (背对相机)
        # 2. RotY(pi) 把人转 180 度，使其重新面向 -Z (正对相机)
        # 使用 euler('xyz', ...) 组合旋转，注意顺序
        rotation_matrix_x = sRot.from_euler('y', np.pi) * sRot.from_euler('x', -np.pi)

    elif origin_format == 'zup':
        rotation_matrix_x = sRot.from_euler('x', -np.pi / 2, degrees=False)
    else:
        raise NotImplementedError("Only 'ydown' origin_format is supported.")

    poses = human['poses']
    trans = human['trans']
    gender = human['gender']
    betas = human['betas']


    obj_trans = obj['trans']  # (T, 3)
    obj_angles = obj['angles']  # (T, 3)
    obj_name = obj['name']

    T = poses.shape[0]
    if betas.ndim == 1:
        # (10,) -> (T, 10)
        betas = np.tile(betas[None, :], (T, 1))
    elif betas.shape[0] == 1:
        # (1, 10) -> (T, 10)
        betas = np.repeat(betas, T, axis=0)
    elif betas.shape[0] != T:
        # 如果 betas 帧数不对（比如只有 10 帧但 poses 有 1000 帧），强制取第一帧并对齐
        betas = np.tile(betas[0:1], (T, 1))

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
    # obj_verts_template -= np.mean(obj_verts_template, axis=1, keepdims=True)
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



def from_yup_to_simulation(human, obj, smpl_model, smplx_parser_n, skeleton_tree, mesh_obj):
    rotation_matrix_x = sRot.from_euler('x', np.pi / 2)

    device = next(smpl_model.parameters()).device

    # 1. 准备数据
    poses = human['poses']  # (T, D)
    trans = human['trans']
    betas = human['betas']

    obj_trans = obj['trans']  # (T, 3)
    obj_angles = obj['angles']  # (T, 3)
    obj_name = obj['name']


    # 2. 获取 yup 坐标系下的 pelvis 轨迹， 计算物体相对 pelvis 的位置
    with torch.no_grad():
        smplx_output = smpl_model(
            body_pose=torch.from_numpy(poses[:, 3:66]).float().to(device),
            global_orient=torch.from_numpy(poses[:, :3]).float().to(device),
            betas=torch.from_numpy(betas).float().to(device),
            transl=torch.from_numpy(trans).float().to(device)
        )
        pelvis = smplx_output.joints.detach().cpu().numpy()[:, 0, :]

    obj_delta_yup = obj_trans - pelvis  # (T, 3)
    # 将位移向量轨迹整体旋转到世界系
    rel_pelvis_obj_world = rotation_matrix_x.apply(obj_delta_yup)

    # --- 4. 计算人体和物体在 Z-up 下的"原始"位置 (未落地修正) ---

    # 4.1 人体原始 Z-up
    pose_aa = poses.copy()
    rotated_root_orientation = rotation_matrix_x * sRot.from_rotvec(pose_aa[:, :3])
    pose_aa[:, :3] = rotated_root_orientation.as_rotvec()
    root_trans_origin_world = rotation_matrix_x.apply(trans)  # (T, 3)

    # 4.2 物体原始 Z-up (基于人体 Root + 相对位移)
    # 注意：这时候人体还没落地修正，所以物体也没落地
    obj_pos_raw = root_trans_origin_world + rel_pelvis_obj_world
    obj_angles_w = rotation_matrix_x * sRot.from_rotvec(obj_angles)

    # --- 5. 联合高度修正 (Joint Ground Alignment) ---

    # 设定检测帧数 (前100帧用于确定地面高度)
    f_check = min(100, pose_aa.shape[0])

    # A. 计算人体最低点
    with torch.no_grad():
        p_t = torch.from_numpy(pose_aa[:f_check]).float()
        t_t = torch.from_numpy(root_trans_origin_world[:f_check]).float()
        # 注意：这里需要传入空的 hand pose 占位，防止 smplx 报错，如果模型包含手
        # 假设 smplx_parser_n 内部处理好了，或者只用了 body_pose
        verts_h, _ = smplx_parser_n.get_joints_verts(p_t, torch.zeros((1, 20)), t_t)
        min_z_h = verts_h[..., -1].min().item()

    # B. 计算物体最低点
    # 准备物体模板顶点 (中心化，与 tranfrom_to_yup 逻辑保持一致)
    v_template = mesh_obj.vertices.copy()
    # v_template -= np.mean(v_template, axis=0)  # (V, 3)

    # 取前 f_check 帧的物体旋转矩阵和平移
    R_obj_check = obj_angles_w[:f_check].as_matrix()  # (T_check, 3, 3)
    T_obj_check = obj_pos_raw[:f_check]  # (T_check, 3)

    # 批量计算物体顶点世界坐标: V_world = V_local @ R.T + T
    # (T, 3, 3) @ (3, V) -> (T, 3, V) -> transpose -> (T, V, 3)
    # 使用 einsum 加速: 'tij, vj -> tvi'
    # (t:frames, i:row, j:col/vert_dim, v:vertices)
    # V_local @ R.T 等价于 (R @ V_local.T).T
    v_obj_world = np.matmul(v_template, np.transpose(R_obj_check, (0, 2, 1))) + T_obj_check[:, np.newaxis, :]
    min_z_o = v_obj_world[..., 2].min()

    # C. 决定修正值：谁更低就以谁为准
    diff_fix = min(min_z_h, min_z_o)

    # --- 6. 应用修正 ---

    # 修正人体 Root
    final_root_trans = root_trans_origin_world - diff_fix

    # 修正物体位置 (物体位置是基于人体计算的，人体降了，物体自然也降了，
    # 但我们需要重新计算基于 final_root_trans 的 obj_pos)
    obj_pos = final_root_trans + rel_pelvis_obj_world

    # 6. 计算物体额外信息（速度、角速度），默认 30 fps
    dt = 1.0 / 30
    obj_pos_vel = np.zeros_like(obj_pos)
    obj_pos_vel[1:] = (obj_pos[1:] - obj_pos[:-1]) / dt

    obj_quat = obj_angles_w.as_quat()  # (T,4) xyzw
    obj_rot_vel = angular_velocity_world_from_quat_xyzw(obj_quat, dt)

    # 7. 把 Zup SMPL (final_root_trans 和 pose_aa) 格式化成合适的 SkeletonMotion 格式
    # 7.1 先做骨骼重排序
    N = pose_aa.shape[0]
    pose_aa_mj = pose_aa.reshape(N, 52, 3)
    smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
    pose_aa_mj = pose_aa_mj[:, smpl_2_mujoco]

    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 52, 4)

    # 局部坐标系旋转（这个是骨架区别，因为 SMPL 的旋转坐标系是Yup, 而机器人是 Zup）
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat),
        torch.from_numpy(final_root_trans).float(),  # 显式转为 Tensor
        is_local=True)

    pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) *
                        sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
    new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                                    torch.from_numpy(pose_quat_global),
                                                                    torch.from_numpy(final_root_trans).float(),
                                                                    is_local=False)

    object = {"name": obj_name, "obj_pos": obj_pos, "obj_rot": obj_quat,
              "obj_pos_vel": obj_pos_vel, "obj_rot_vel": obj_rot_vel}

    return new_sk_state, object


def from_retarget_to_simulation(npz_path, mj_model, mj_data, skeleton_tree, device='cuda'):
    """
    读取 robot_retarget.py 的结果，并将其转换为仿真流程所需的格式。
    """
    # 1. 加载数据
    data = np.load(npz_path, allow_pickle=True)
    qpos = data["qpos"]  # (T, D)
    fps = int(data["fps"]) if "fps" in data else 30

    T = qpos.shape[0]

    if "smpl_scale" in data:
        # np.savez 存的是数组，如果是标量可能是 0-d array
        smpl_scale = float(data["smpl_scale"])
    else:
        smpl_scale = 1.0

    # 2. 解析 qpos (假设 MuJoCo 格式: Root(7) + Joints(N) + Object(7))
    # 机器人部分
    root_pos = qpos[:, 0:3]
    root_quat = qpos[:, 3:7]  # wxyz

    # 物体部分 (最后7位)
    obj_pos = qpos[:, -7:-4]
    obj_quat = qpos[:, -4:]  # wxyz

    # 关节部分
    joint_qpos = qpos[:, 7:-7]

    # 3. 构建 object_dict
    # 计算速度用于物理仿真（如果需要）
    dt = 1.0 / fps
    obj_pos_vel = np.zeros_like(obj_pos)
    obj_pos_vel[1:] = (obj_pos[1:] - obj_pos[:-1]) / dt

    # 简单的角速度计算 (wxyz -> 转换后计算)
    # 注意：angular_velocity_world_from_quat_xyzw 需要 xyzw
    # 这里为了简单，先暂存 wxyz
    object_dict = {
        'obj_pos': obj_pos,  # numpy (T, 3)
        'obj_rot': obj_quat,  # numpy (T, 4) [w, x, y, z]
        'obj_pos_vel': obj_pos_vel,
        # 'obj_rot_vel': ... (如果有需要再计算，注意四元数顺序)
        'name': 'unknown'  # 外部填充
    }

    # 4. 构建 motion_dict (用于直接保存和渲染)
    motion_dict = {
        'root_trans_offset': root_pos,
        'root_rotation': root_quat,  # [w, x, y, z]
        'dof': joint_qpos,
        'fps': fps
    }

    # 5. 使用 MuJoCo 进行前向运动学 (FK)
    global_trans_list = []
    global_rot_list = []

    # 遍历每一帧计算 FK
    for t in range(T):
        mj_data.qpos[:3] = root_pos[t]
        mj_data.qpos[3:7] = root_quat[t]
        mj_data.qpos[7:] = joint_qpos[t]  # 假设 xml 只有机器人，没有物体

        mujoco.mj_kinematics(mj_model, mj_data)

        # 获取所有 body 的位置和旋转
        # mj_data.xpos: (nbody, 3)
        # mj_data.xquat: (nbody, 4) [w, x, y, z]

        # --- FIX: 去掉 [0]，我们要所有 body 的位置 ---
        global_trans_list.append(mj_data.xpos.copy())

        # --- FIX: wxyz -> xyzw ---
        # PoseLib/SkeletonState 通常使用 [x, y, z, w]
        global_rot_list.append(mj_data.xquat.copy()[:, [1, 2, 3, 0]])

        # 转为 Tensor (T, N_body, 3/4)
    # 注意：SkeletonState 需要在 CPU 上构建，或者与 device 匹配
    global_trans_tensor = torch.tensor(np.array(global_trans_list), dtype=torch.float32)
    global_rot_tensor = torch.tensor(np.array(global_rot_list), dtype=torch.float32)

    # 6. 构建 SkeletonState
    # from_rotation_and_root_translation 需要:
    # rotations: (..., J, 4) - Global rotations (XYZW)
    # root_translation: (..., 3) - Global root translation

    # --- FIX: 只传入 Root 的 Translation ---
    # global_trans_tensor[:, 0, :] 假设第 0 个 body 是 root (通常 World 或 Pelvis)
    # 根据 SkeletonTree 的定义，通常索引 0 或 1 是 Root，取决于是否 drop_world
    # 为了保险，直接使用输入的 root_pos 即可
    root_pos_tensor = torch.tensor(root_pos, dtype=torch.float32)

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        global_rot_tensor,  # 全局旋转
        root_pos_tensor,  # 根节点位移
        is_local=False  # 指定输入的是全局坐标
    )

    return new_sk_state, object_dict, motion_dict, smpl_scale
import numpy as np
import torch
import trimesh
import pyrender
import imageio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot
from pyrender import MetallicRoughnessMaterial


def spherical_to_cartesian(target, dist, azim, elev):
    """
    将 MuJoCo 的球坐标参数转换为 Z-up 笛卡尔坐标
    MuJoCo 约定: azimuth=0 在 -Y 方向看往 +Y? 或者看 X?
    通常: azim=0 是从正前方看。
    """
    # 转换为弧度
    azim_rad = np.deg2rad(azim)
    elev_rad = np.deg2rad(elev)

    # 计算相对偏移 (Z-up 坐标系)
    # MuJoCo 的 elevation 是相对于水平面的角度
    x = dist * np.cos(elev_rad) * np.sin(azim_rad)
    y = -dist * np.cos(elev_rad) * np.cos(azim_rad)
    z = -dist * np.sin(elev_rad)  # 因为 elev 是负的，所以 -dist * sin 为正

    return target + np.array([x, y, z])

def look_at_zup(camera_pos, target, up=np.array([0.0, 0.0, 1.0])):
    """
    Z-up 坐标系下的 Camera Look-at
    """
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - camera_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        forward = np.array([0.0, 1.0, 0.0])
    else:
        forward = forward / forward_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        up2 = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up2)
        right = right / (np.linalg.norm(right) + 1e-12)
    else:
        right = right / right_norm

    true_up = np.cross(right, forward)

    R = np.eye(3)
    R[:, 0] = right
    R[:, 1] = true_up
    R[:, 2] = -forward

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = camera_pos
    return T


def render_smplx_hoi_video_zup(
        smplx_model,  # 现在是 smplx.create 生成的模型
        poses_zup,
        trans_zup,
        betas,
        obj_mesh_path,
        obj_trans_zup,
        obj_rotmat_zup,
        output_path,
        camera_cfg,
        fps=30,
        viewport=(1280, 720)
):
    device = next(smplx_model.parameters()).device  # 自动获取模型所在设备

    poses_zup = np.asarray(poses_zup)
    trans_zup = np.asarray(trans_zup)
    T_frames = trans_zup.shape[0]

    # 1. 修改点：获取面片信息，smplx 类使用 .faces
    faces = smplx_model.faces

    # 材质和地面设置保持不变...
    smpl_material = MetallicRoughnessMaterial(baseColorFactor=[0.7, 0.7, 0.9, 1.0], metallicFactor=0.05,
                                              roughnessFactor=0.9)
    obj_material = MetallicRoughnessMaterial(baseColorFactor=[0.8, 0.5, 0.5, 1.0], metallicFactor=0.2,
                                             roughnessFactor=0.6)
    floor_material = MetallicRoughnessMaterial(baseColorFactor=[0.85, 0.85, 0.85, 1.0], metallicFactor=0.0,
                                               roughnessFactor=1.0)
    floor_trimesh = trimesh.creation.box(extents=[10.0, 10.0, 0.001])
    floor_pose = np.eye(4);
    floor_pose[2, 3] = -0.0005
    floor_pyr = pyrender.Mesh.from_trimesh(floor_trimesh, material=floor_material)
    obj_base = trimesh.load(obj_mesh_path, force="mesh")

    W, H = viewport
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

    # 2. 修改点：确保 betas 维度匹配 (num_betas=16)
    if not torch.is_tensor(betas):
        betas = torch.from_numpy(betas).float()

    # 截断或填充到模型需要的维度 (例如 16)
    num_betas_model = smplx_model.num_betas
    if betas.shape[-1] != num_betas_model:
        betas = betas[..., :num_betas_model]

    betas = betas.view(1, -1).to(device)

    if obj_rotmat_zup.ndim == 2:
        obj_rotmat_zup = np.repeat(obj_rotmat_zup[None, :, :], T_frames, axis=0)

    with imageio.get_writer(output_path, fps=fps) as video:
        for t in tqdm(range(T_frames), desc="Render SMPL-X HOI (Z-up)",  mininterval=1):
            scene = pyrender.Scene(ambient_light=[0.15, 0.15, 0.15], bg_color=[240, 240, 240])
            scene.add(floor_pyr, pose=floor_pose)

            p_t = torch.from_numpy(poses_zup[t:t + 1]).float().to(device)
            tr_t = torch.from_numpy(trans_zup[t:t + 1]).float().to(device)

            # 3. 修改点：传参名变更
            with torch.no_grad():
                output = smplx_model(
                    global_orient=p_t[:, :3],  # 原 root_orient
                    body_pose=p_t[:, 3:66],  # 原 pose_body
                    left_hand_pose=p_t[:, 66:111],  # 原 pose_hand_l
                    right_hand_pose=p_t[:, 111:156],  # 原 pose_hand_r
                    betas=betas.expand(1, -1),  # 确保 batch 一致
                    transl=tr_t,  # 原 trans
                )
                # 4. 修改点：输出属性名变更，使用 .vertices
                verts = output.vertices[0].detach().cpu().numpy()

            smpl_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            scene.add(pyrender.Mesh.from_trimesh(smpl_mesh, material=smpl_material))

            # 物体渲染和相机设置逻辑保持不变...
            obj_T = np.eye(4);
            obj_T[:3, :3] = obj_rotmat_zup[t];
            obj_T[:3, 3] = obj_trans_zup[t]
            obj_mesh = obj_base.copy().apply_transform(obj_T)
            scene.add(pyrender.Mesh.from_trimesh(obj_mesh, material=obj_material))

            target = trans_zup[t] + camera_cfg['lookat_offset']
            camera_pos = spherical_to_cartesian(
                target,
                camera_cfg['distance'],
                camera_cfg['azimuth'],
                camera_cfg['elevation']
            )
            cam_pose = look_at_zup(camera_pos, target)
            scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=cam_pose)
            scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=cam_pose)

            color, _ = renderer.render(scene)
            video.append_data(color)

    renderer.delete()
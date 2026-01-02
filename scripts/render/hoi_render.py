import numpy as np
import torch
import trimesh
import pyrender
import imageio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot
from pyrender import MetallicRoughnessMaterial

def spherical_to_cartesian_yup(target, dist, azim, elev):
    """
    将 MuJoCo 风格的球坐标参数转换为 Y-up 笛卡尔坐标
    azim: 0 位于 -Z 轴, 正值向 +X 旋转
    elev: 0 位于 XZ 平面, 正值向 +Y 旋转
    """
    azim_rad = np.deg2rad(azim)
    elev_rad = np.deg2rad(elev)

    # Y-up 坐标计算
    x = dist * np.cos(elev_rad) * np.sin(azim_rad)
    y = -dist * np.sin(elev_rad) # 注意：如果 elev 是 -15，y 为正，即从上往下看
    z = dist * np.cos(elev_rad) * np.cos(azim_rad)

    return target + np.array([x, y, z])


def look_at_yup(camera_pos, target):
    """
    Y-up 坐标系下的 Camera Look-at (Up = [0, 1, 0])
    """
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)

    up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = true_up
    T[:3, 2] = -forward
    T[:3, 3] = camera_pos
    return T


def render_smpl_hoi_video_yup(
        smpl_model,
        human,
        obj,
        obj_mesh_path,
        output_path,
        camera_cfg,
        fps=30,
        viewport=(1280, 720)
):
    """
    专门接受 human_yup 和 obj_yup 字典输入的渲染函数
    """
    device = next(smpl_model.parameters()).device

    # 1. 解包数据
    poses = np.asarray(human['poses'])  # (T, 156)
    trans = np.asarray(human['trans'])  # (T, 3)
    betas = human['betas']  # (1, 16)

    obj_trans = np.asarray(obj['trans'])  # (T, 3)
    obj_angles = np.asarray(obj['angles'])  # (T, 3)

    T_frames = trans.shape[0]
    faces = smpl_model.faces

    # 2. 准备材质和静态物体
    smpl_material = MetallicRoughnessMaterial(baseColorFactor=[0.7, 0.7, 0.9, 1.0], metallicFactor=0.05,
                                              roughnessFactor=0.9)
    obj_material = MetallicRoughnessMaterial(baseColorFactor=[0.8, 0.5, 0.5, 1.0], metallicFactor=0.2,
                                             roughnessFactor=0.6)
    floor_material = MetallicRoughnessMaterial(baseColorFactor=[0.85, 0.85, 0.85, 1.0], metallicFactor=0.0,
                                               roughnessFactor=1.0)

    # 静态地面 (Y-up: 放在 Y=0, 稍微往下一点防止 z-fighting)
    floor_trimesh = trimesh.creation.box(extents=[20.0, 0.001, 20.0])
    floor_pose = np.eye(4)
    floor_pose[1, 3] = -0.0005
    floor_pyr = pyrender.Mesh.from_trimesh(floor_trimesh, material=floor_material)

    # 静态物体网格 (只上传 GPU 一次)
    mesh_obj = trimesh.load(obj_mesh_path, force="mesh")
    obj_pyr_mesh = pyrender.Mesh.from_trimesh(mesh_obj, material=obj_material)

    # 3. 处理 Betas Tensor
    if not isinstance(betas, torch.Tensor):
        betas = torch.from_numpy(np.asarray(betas)).float().to(device)
    if betas.ndim == 2 and betas.shape[0] != 1:
        betas = betas[0:1]  # 确保是 (1, N)

    num_betas_model = smpl_model.num_betas
    betas = betas[..., :num_betas_model]

    renderer = pyrender.OffscreenRenderer(viewport_width=viewport[0], viewport_height=viewport[1])

    with imageio.get_writer(output_path, fps=fps) as video:
        for t in tqdm(range(T_frames), desc="Rendering Y-up HOI", mininterval=2.0):
            scene = pyrender.Scene(ambient_light=[0.15, 0.15, 0.15], bg_color=[240, 240, 240])

            # A. 添加地面
            scene.add(floor_pyr, pose=floor_pose)

            # B. 添加物体 (仅更新 Pose)
            obj_T = np.eye(4)
            obj_T[:3, :3] = sRot.from_rotvec(obj_angles[t]).as_matrix()
            obj_T[:3, 3] = obj_trans[t]
            scene.add(obj_pyr_mesh, pose=obj_T)

            # C. 添加人体
            p_t = torch.from_numpy(poses[t:t + 1]).float().to(device)
            tr_t = torch.from_numpy(trans[t:t + 1]).float().to(device)

            with torch.no_grad():
                output = smpl_model(
                    global_orient=p_t[:, :3],
                    body_pose=p_t[:, 3:66],
                    left_hand_pose=p_t[:, 66:111],
                    right_hand_pose=p_t[:, 111:156],
                    betas=betas,
                    transl=tr_t,
                )
                verts = output.vertices[0].detach().cpu().numpy()

            smpl_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            scene.add(pyrender.Mesh.from_trimesh(smpl_mesh, material=smpl_material))

            # D. 相机跟随逻辑 (Y-up)
            # MuJoCo 的 lookat_offset 通常是 [0, 0, 0.7]，在 Yup 下我们需要它在 Y 轴上有偏移
            # 如果你的 config 没改，这里我们把 offset 转换一下
            y_offset = np.array([camera_cfg['lookat_offset'][0],
                                 camera_cfg['lookat_offset'][2],
                                 -camera_cfg['lookat_offset'][1]])

            target = trans[t] + y_offset
            camera_pos = spherical_to_cartesian_yup(
                target,
                camera_cfg['distance'],
                camera_cfg['azimuth'],
                camera_cfg['elevation']
            )
            cam_pose = look_at_yup(camera_pos, target)

            scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=cam_pose)
            scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=cam_pose)

            color, _ = renderer.render(scene)
            video.append_data(color)

    renderer.delete()
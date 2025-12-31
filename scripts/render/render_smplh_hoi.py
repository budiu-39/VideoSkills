import numpy as np
import torch
import trimesh
import pyrender
import imageio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot
from pyrender import MetallicRoughnessMaterial


def render_smplh_hoi_video(
        smplh_layer,
        poses, trans, betas,
        obj_mesh_path, obj_pos, obj_quat_xyzw,
        output_path, fps=30
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smplh_layer = smplh_layer.to(device)
    num_frames = len(trans)

    # 1. 获取面片信息
    if hasattr(smplh_layer, 'faces'):
        faces = smplh_layer.faces
    else:
        faces = smplh_layer.smpl_data['f']
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    # 2. 准备材质
    smpl_material = MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.7, 0.9, 1.0],
        metallicFactor=0.1, roughnessFactor=0.8
    )
    obj_material = MetallicRoughnessMaterial(
        baseColorFactor=[0.5, 0.8, 0.5, 1.0],
        metallicFactor=0.2, roughnessFactor=0.5
    )
    floor_material = MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.8, 0.8, 1.0],
        metallicFactor=0.0, roughnessFactor=0.9
    )

    # 3. 【修改】创建地板 Mesh (Y-Up 优化)
    # extents 在 Y 轴上非常薄
    floor_trimesh = trimesh.creation.box(extents=[10, 0.001, 10])
    floor_pyrender = pyrender.Mesh.from_trimesh(floor_trimesh, material=floor_material)

    obj_base_mesh = trimesh.load(obj_mesh_path, force='mesh')
    renderer = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=720)

    with imageio.get_writer(output_path, fps=fps) as video:
        for t in tqdm(range(num_frames)):
            # 环境光与背景
            scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1], bg_color=[230, 230, 230])

            # A. 添加地板 (保持在 Y=0)
            scene.add(floor_pyrender, pose=np.eye(4))

            # B. 计算 SMPL 顶点
            p_t = torch.from_numpy(poses[t:t + 1]).float().to(device)
            t_t = torch.from_numpy(trans[t:t + 1]).float().to(device)
            b_t = betas.to(device).view(1, -1).expand(1, -1)

            with torch.no_grad():
                output = smplh_layer(p_t, th_betas=b_t, th_trans=t_t)
                smpl_verts = output[0][0].cpu().numpy()

            smpl_mesh = trimesh.Trimesh(vertices=smpl_verts, faces=faces)
            scene.add(pyrender.Mesh.from_trimesh(smpl_mesh, material=smpl_material))

            # C. 添加物体
            obj_mat = np.eye(4)
            obj_mat[:3, :3] = sRot.from_quat(obj_quat_xyzw[t]).as_matrix()
            obj_mat[:3, 3] = obj_pos[t]
            curr_obj_mesh = obj_base_mesh.copy()
            curr_obj_mesh.apply_transform(obj_mat)
            scene.add(pyrender.Mesh.from_trimesh(curr_obj_mesh, material=obj_material))

            # ------------------------------------------------------
            # D. 【关键修改】相机跟随 (Y-Up 模式)
            # ------------------------------------------------------
            # 目标点：人的躯干 (Y 是高度)
            target = trans[t].copy()
            target[1] += 0.9  # 向上偏移到胸部

            # 相机位置：从 X 和 Z 方向偏移，Y轴略微抬高
            # 相机放在人的 侧前方 (x=3, z=3)，高度 y=1.5
            camera_offset = np.array([3.0, 0.6, 3.0])
            camera_pos = target + camera_offset

            # 计算相机姿态：
            # 在 Y-up 下，相机默认看向 -Z 方向。
            # 旋转逻辑：绕 Y 轴转 45 度（水平转身），绕 X 轴转 -15 度（向下低头）
            cam_rot = sRot.from_euler('YXZ', [45, -15, 0], degrees=True).as_matrix()

            cam_pose = np.eye(4)
            cam_pose[:3, :3] = cam_rot
            cam_pose[:3, 3] = camera_pos

            scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=cam_pose)

            # 光照跟随相机
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            scene.add(light, pose=cam_pose)
            # ------------------------------------------------------

            # E. 渲染
            color, _ = renderer.render(scene)
            video.append_data(color)

    renderer.delete()
    print(f"✅ 视频保存至: {output_path}")
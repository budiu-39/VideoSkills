import numpy as np
import torch
import trimesh
import pyrender
import imageio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot
from pyrender import MetallicRoughnessMaterial


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
    smplx_model,
    poses_zup,
    trans_zup,
    betas,
    obj_mesh_path,
    obj_trans_zup,
    obj_rotmat_zup,
    output_path,
    fps=30,
    viewport=(1280, 720)
):

    device = torch.device("cuda")

    poses_zup = np.asarray(poses_zup)
    trans_zup = np.asarray(trans_zup)
    T_frames = trans_zup.shape[0]

    # 获取面片信息 (SMPL-X BodyModel 属性为 .f)
    faces = smplx_model.f
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    # 材质设置
    smpl_material = MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.7, 0.9, 1.0],
        metallicFactor=0.05, roughnessFactor=0.9
    )
    obj_material = MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.5, 0.5, 1.0],  # 稍微改下颜色区分
        metallicFactor=0.2, roughnessFactor=0.6
    )
    floor_material = MetallicRoughnessMaterial(
        baseColorFactor=[0.85, 0.85, 0.85, 1.0],
        metallicFactor=0.0, roughnessFactor=1.0
    )

    # 地面
    floor_trimesh = trimesh.creation.box(extents=[10.0, 10.0, 0.001])
    floor_pose = np.eye(4)
    floor_pose[2, 3] = -0.0005
    floor_pyr = pyrender.Mesh.from_trimesh(floor_trimesh, material=floor_material)

    # 物体基准 Mesh
    obj_base = trimesh.load(obj_mesh_path, force="mesh")

    W, H = viewport
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

    # 准备 Betas
    if not torch.is_tensor(betas):
        betas = torch.from_numpy(betas).float()
    betas = betas.view(1, -1).to(device)

    # 支持物体旋转矩阵 (T,3,3) 或 (3,3)
    if obj_rotmat_zup.ndim == 2:
        obj_rotmat_zup = np.repeat(obj_rotmat_zup[None, :, :], T_frames, axis=0)

    with imageio.get_writer(output_path, fps=fps) as video:
        for t in tqdm(range(T_frames), desc="Render SMPL-X HOI (Z-up)"):
            scene = pyrender.Scene(ambient_light=[0.15, 0.15, 0.15], bg_color=[240, 240, 240])
            scene.add(floor_pyr, pose=floor_pose)

            # --- SMPL-X 参数切片 ---
            # 假设 poses_zup 结构符合 OMOMO: root(3) + body(63) + hand(90) = 156
            p_t = torch.from_numpy(poses_zup[t:t + 1]).float().to(device)
            tr_t = torch.from_numpy(trans_zup[t:t + 1]).float().to(device)

            # 提取 SMPL-X 需要的各个部分
            root_orient = p_t[:, :3]
            pose_body = p_t[:, 3:66]
            # SMPL-X BodyModel 通常需要 pose_hand (L/R)
            # 如果是 156 维，则 66:111 是左手，111:156 是右手
            pose_lhand = p_t[:, 66:111]
            pose_rhand = p_t[:, 111:156]

            with torch.no_grad():
                # 调用 BodyModel
                # 注意：如果你的模型有 jaw/eye/pca，可以在这里添加对应参数
                output = smplx_model(
                    root_orient=root_orient,
                    pose_body=pose_body,
                    pose_hand_l=pose_lhand,
                    pose_hand_r=pose_rhand,
                    betas=betas,
                    trans=tr_t,
                    return_verts=True
                )
                # BodyModel 输出的顶点在 .v 属性中
                verts = output.v[0].detach().cpu().numpy()

            smpl_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            scene.add(pyrender.Mesh.from_trimesh(smpl_mesh, material=smpl_material))

            # --- 物体渲染 ---
            obj_T = np.eye(4)
            obj_T[:3, :3] = obj_rotmat_zup[t]
            obj_T[:3, 3] = obj_trans_zup[t]
            obj_mesh = obj_base.copy()
            obj_mesh.apply_transform(obj_T)
            scene.add(pyrender.Mesh.from_trimesh(obj_mesh, material=obj_material))

            # --- 相机设置 ---
            target = trans_zup[t].copy()
            target[2] += 0.9  # 聚焦在人体中心位置
            camera_pos = target + np.array([2.5, -2.5, 1.2])
            cam_pose = look_at_zup(camera_pos, target)

            cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            scene.add(cam, pose=cam_pose)

            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            scene.add(light, pose=cam_pose)

            color, _ = renderer.render(scene)
            video.append_data(color)

    renderer.delete()
    print(f"✅ Video saved to: {output_path}")
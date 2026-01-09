import os
import time
import mujoco
import imageio
from scipy.spatial.transform import Rotation as sRot
import tqdm
import numpy as np

def export_mujoco_video(
        motion_traj: dict,
        output_path: str,
        fps: int,
        xml_path,
        width: int = 1280,
        height: int = 720,
):
    """
    Render a motion sequence (new format) to MP4 via MuJoCo off-screen renderer.

    Parameters
    ----------
    motion_traj : dict
        Should contain keys "root_trans_offset", "pose_aa", "dof", "fps".
    output_path : str
        e.g. 'output/render_out/videos/XXX.mp4'
    humanoid_type : str
        Determines XML path: data/robots/g1/g1_29dof.xml
    fps : int | None
        Frames per second for output video.  If None use motion_traj["fps"].
    width / height : int
        Resolution of rendered frames.
    """

    # (below here keep your original code)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if fps is None:
        fps = int(motion_traj.get("fps", 30))

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)

    root_traj = motion_traj["root_trans_offset"]        # (N,3)
    rotvec_traj = motion_traj["pose_aa"][:, 0]          # (N,3)
    dof_traj  = motion_traj["dof"].reshape(len(root_traj), -1)

    num_frames = len(root_traj)
    assert rotvec_traj.shape[0] == num_frames

    # ----------- 3. 初始化渲染器 & 相机 -----------
    renderer = mujoco.Renderer(mj_model, width=width, height=height)
    cam = mujoco.MjvCamera()

    # 把相机对准首帧、保持随动简单易调
    cam.lookat[:]   = root_traj[0]
    cam.distance    = 3.0
    cam.elevation   = -10.0
    heading_vec     = sRot.from_rotvec(rotvec_traj[0]).apply([0, 0, 0])
    cam.azimuth     = -90

    # ----------- 4. 写视频 -----------
    writer = imageio.get_writer(output_path, fps=fps)

    for t in range(num_frames):
        # qpos = [root_xyz, root_quat(wxyz), dof...]
        mj_data.qpos[:3]  = root_traj[t]
        mj_data.qpos[3:7] = sRot.from_rotvec(rotvec_traj[t]).as_quat()[[3, 0, 1, 2]]
        mj_data.qpos[7:]  = dof_traj[t]

        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam)
        frame = renderer.render()
        writer.append_data(frame)

    writer.close()
    print(f" video saved →  {output_path}")

def vis_mujoco(motion_traj, xml_path, humanoid_type='g1'):

    print(mujoco.__version__)  # 应该输出 3.2.3
    print(hasattr(mujoco, "viewer"))

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    num_frames = len(motion_traj['root_trans_offset'])

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for t in range(num_frames):
            mj_data.qpos[:3] = motion_traj['root_trans_offset'][t]
            mj_data.qpos[3:7] = motion_traj['root_rotation'][t][[3, 0, 1, 2]]  # Convert from wxyz to xyzw
            mj_data.qpos[7:] = motion_traj['dof'][t].flatten()
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            time.sleep(1 / 30)

def vis_mujoco_hoi(motion_traj, obj_pos, obj_quat_xyzw, xml_path):
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)
    num_frames = len(motion_traj['root_trans_offset'])

    def xyzw_to_wxyz(q): return q[[3,0,1,2]]

    obj_mocap_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object_body")
    obj_mocap_id  = obj_mocap_bid - mj_model.nbody + mj_model.nmocap

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for t in range(num_frames):
            mj_data.qpos[:3]  = motion_traj['root_trans_offset'][t]
            mj_data.qpos[3:7] = motion_traj['root_rotation'][t][[3,0,1,2]]
            mj_data.qpos[7:]  = motion_traj['dof'][t].flatten()

            mj_data.mocap_pos[obj_mocap_id]  = obj_pos[t]
            mj_data.mocap_quat[obj_mocap_id] = xyzw_to_wxyz(obj_quat_xyzw[t])

            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            time.sleep(1/30)


def export_mujoco_video_hoi_wxyz(motion_traj, obj_pos, obj_quat_wxyz, xml_path, camera_cfg,
                            output_path="output.mp4", fps=30):
    """
    导出 MuJoCo 视频，专门处理 motion_dict 格式输入。

    Args:
        motion_traj (dict): 包含以下键的字典:
            - 'root_trans_offset': (T, 3) 全局位移
            - 'root_rotation': (T, 4) 全局旋转 [w, x, y, z] (MuJoCo格式)
            - 'dof': (T, 29) 关节角度 (标量)
        obj_pos (np.ndarray): (T, 3) 物体位置
        obj_quat_wxyz (np.ndarray): (T, 4) 物体旋转 [w, x, y, z] (MuJoCo格式)
        xml_path (str): 模型 XML 路径
        camera_cfg (dict): 相机配置
    """
    # 1. 初始化模型
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML not found: {xml_path}")

    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # --- 修复 Offscreen 缓冲区报错 (尤其是大分辨率下) ---
    mj_model.vis.global_.offwidth = 1280
    mj_model.vis.global_.offheight = 720
    # -----------------------------------------------

    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=720, width=1280)

    # 2. 相机设置
    camera = mujoco.MjvCamera()
    # 默认距离和角度，可通过 camera_cfg 覆盖
    camera.distance = camera_cfg.get('distance', 3.0) + 1.0
    camera.azimuth = camera_cfg.get('azimuth', 90) - 90
    camera.elevation = camera_cfg.get('elevation', -15)
    # 视点偏移 (相对于机器人根节点)
    lookat_offset = camera_cfg.get('lookat_offset', np.zeros(3))

    num_frames = len(motion_traj['root_trans_offset'])

    # 3. 获取物体 Mocap ID (增加健壮性检查)
    obj_mocap_id = -1
    try:
        # 查找名为 "object_body" 的 body ID
        obj_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object_body")
        if obj_bid != -1:
            # 获取对应的 mocap ID
            obj_mocap_id = mj_model.body_mocapid[obj_bid]
        else:
            print("⚠️ Warning: 'object_body' not found in XML. Object will not be rendered.")
    except Exception as e:
        print(f"⚠️ Warning: Error checking object body: {e}")

    print(f"🎬 正在导出视频到: {output_path} (Frames: {num_frames})")

    with imageio.get_writer(output_path, fps=fps) as video:
        for t in tqdm.tqdm(range(num_frames), mininterval=1.0):

            # --- A. 更新机器人状态 ---
            mj_data.qpos[:3] = motion_traj['root_trans_offset'][t]
            mj_data.qpos[3:7] = motion_traj['root_rotation'][t]
            mj_data.qpos[7:] = motion_traj['dof'][t].reshape(-1)

            # --- B. 更新物体状态 ---
            if obj_mocap_id != -1:
                mj_data.mocap_pos[obj_mocap_id] = obj_pos[t]
                mj_data.mocap_quat[obj_mocap_id] = obj_quat_wxyz[t]

            # --- C. 物理计算 (正向运动学) ---
            # 必须调用此函数以更新几何体位置 (xpos, xquat 等)
            mujoco.mj_forward(mj_model, mj_data)

            # --- D. 更新相机视角 ---
            # 让相机 LookAt 始终跟随机器人的根节点位置 + 偏移量
            camera.lookat[:] = mj_data.qpos[:3] + lookat_offset

            # --- E. 渲染 ---
            renderer.update_scene(mj_data, camera=camera)
            frame = renderer.render()
            video.append_data(frame)

    print(f"✅ 视频导出成功: {output_path}")

# def export_mujoco_video_hoi(motion_traj, obj_pos, obj_quat_xyzw, xml_path, camera_cfg,
#                             output_path="output.mp4", fps=30):
#     # 1. 初始化模型
#     mj_model = mujoco.MjModel.from_xml_path(xml_path)
#
#     # --- 修复 Offscreen 缓冲区报错 ---
#     mj_model.vis.global_.offwidth = 1280
#     mj_model.vis.global_.offheight = 720
#     # ------------------------------
#
#     mj_data = mujoco.MjData(mj_model)
#     renderer = mujoco.Renderer(mj_model, height=720, width=1280)
#
#     # 3. 相机设置
#     # 不使用可能报错的 camera.type = ...，改用手动 lookat 跟随
#     camera = mujoco.MjvCamera()
#     camera.distance = camera_cfg['distance'] + 1.0
#     camera.azimuth = camera_cfg['azimuth'] + 90
#     camera.elevation = camera_cfg['elevation']
#
#     num_frames = len(motion_traj['root_trans_offset'])
#
#     def xyzw_to_wxyz(q): return q[[3, 0, 1, 2]]
#
#     # 4. 获取物体 mocap ID
#     obj_mocap_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object_body")
#     obj_mocap_id = mj_model.body_mocapid[obj_mocap_bid]
#
#     print(f"正在导出视频到: {output_path}...")
#
#     with imageio.get_writer(output_path, fps=fps) as video:
#         for t in tqdm.tqdm(range(num_frames), mininterval=1.0):
#             # 更新人体姿态
#             mj_data.qpos[:3] = motion_traj['root_trans_offset'][t]
#             mj_data.qpos[3:7] = xyzw_to_wxyz(motion_traj['root_rotation'][t])
#             mj_data.qpos[7:] = motion_traj['dof'][t].flatten()
#
#             # 更新物体位置
#             mj_data.mocap_pos[obj_mocap_id] = obj_pos[t]
#             mj_data.mocap_quat[obj_mocap_id] = xyzw_to_wxyz(obj_quat_xyzw[t])
#
#             # 物理计算更新
#             mujoco.mj_forward(mj_model, mj_data)
#
#             # --- 关键：摄像机跟随逻辑 ---
#             # 每一帧都让相机盯住人物当前的根节点位置 (root_xyz)
#             camera.lookat[:] = mj_data.qpos[:3] + camera_cfg['lookat_offset']
#             # --------------------------
#
#             # 渲染并写入
#             renderer.update_scene(mj_data, camera=camera)
#             frame = renderer.render()
#             video.append_data(frame)
#
#     print(f"✅ 视频已保存至: {output_path}")

def export_mujoco_video_hoi(motion_traj, obj_pos, obj_quat_xyzw, xml_path, camera_cfg,
                            output_path="output.mp4", fps=30):
    # 1. 初始化模型
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # --- 修复 Offscreen 缓冲区报错 ---
    mj_model.vis.global_.offwidth = 1280
    mj_model.vis.global_.offheight = 720
    # ------------------------------

    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=720, width=1280)

    # 3. 相机设置
    # 不使用可能报错的 camera.type = ...，改用手动 lookat 跟随
    camera = mujoco.MjvCamera()
    camera.distance = camera_cfg['distance'] + 1.0
    camera.azimuth = camera_cfg['azimuth'] + 90
    camera.elevation = camera_cfg['elevation']

    num_frames = len(motion_traj['root_trans_offset'])

    def xyzw_to_wxyz(q): return q[[3, 0, 1, 2]]

    # 4. 获取物体 mocap ID
    obj_mocap_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object_body")
    obj_mocap_id = mj_model.body_mocapid[obj_mocap_bid]

    print(f"正在导出视频到: {output_path}...")

    with imageio.get_writer(output_path, fps=fps) as video:
        for t in tqdm.tqdm(range(num_frames), mininterval=1.0):
            # 更新人体姿态
            mj_data.qpos[:3] = motion_traj['root_trans_offset'][t]
            mj_data.qpos[3:7] = xyzw_to_wxyz(motion_traj['root_rotation'][t])
            mj_data.qpos[7:] = motion_traj['dof'][t].flatten()

            # 更新物体位置
            mj_data.mocap_pos[obj_mocap_id] = obj_pos[t]
            mj_data.mocap_quat[obj_mocap_id] = xyzw_to_wxyz(obj_quat_xyzw[t])

            # 物理计算更新
            mujoco.mj_forward(mj_model, mj_data)

            # --- 关键：摄像机跟随逻辑 ---
            # 每一帧都让相机盯住人物当前的根节点位置 (root_xyz)
            camera.lookat[:] = mj_data.qpos[:3] + camera_cfg['lookat_offset']
            # --------------------------

            # 渲染并写入
            renderer.update_scene(mj_data, camera=camera)
            frame = renderer.render()
            video.append_data(frame)

    print(f"✅ 视频已保存至: {output_path}")

# def vis_mujoco(motion_traj, xml_path, humanoid_type = 'g1'):
#
#     print(mujoco.__version__)  # 应该输出 3.2.3
#     print(hasattr(mujoco, "viewer"))
#
#     mj_model = mujoco.MjModel.from_xml_path(xml_path)
#     mj_data = mujoco.MjData(mj_model)
#     num_frames = len(motion_traj['root_trans_offset'])
#
#     with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
#         # 设置摄像机参数
#         viewer.cam.lookat[:] = motion_traj['root_trans_offset'][0]  # 朝向机器人初始位置
#         viewer.cam.distance = 3.0  # 相机距离，可调整
#         viewer.cam.azimuth = -90  # 方位角（左侧视图）
#         viewer.cam.elevation = -15  # 仰角（往下看）
#
#         for t in range(num_frames):
#             mj_data.qpos[:3] = motion_traj['root_trans_offset'][t]
#             mj_data.qpos[3:7] = sRot.from_rotvec(motion_traj['pose_aa'][t][0]).as_quat()[[3, 0, 1, 2]]
#             mj_data.qpos[7:] = motion_traj['dof'][t].flatten()
#             mujoco.mj_forward(mj_model, mj_data)
#             viewer.sync()
#             time.sleep(1 / 30)

import os
import tempfile
from lxml import etree


def create_temp_xml_with_object(base_xml_path, obj_mesh_path, smpl_scale=1.0):
    """
    在现有 robot xml 中注入 object mesh 和 body。

    Args:
        base_xml_path: 机器人 XML 路径
        obj_mesh_path: 物体 Mesh 路径
        smpl_scale (float): 统一缩放比例 (应用到 x, y, z)
    """
    # 解析 XML
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(base_xml_path, parser)
    root = tree.getroot()

    # 1. 准备 Mesh 路径和名称
    if obj_mesh_path is None or not os.path.exists(obj_mesh_path):
        # 如果没有物体，直接返回原文件或不做修改
        return base_xml_path

    mesh_name = os.path.splitext(os.path.basename(obj_mesh_path))[0]
    abs_mesh_path = os.path.abspath(obj_mesh_path)

    # 构造缩放字符串 "s s s"
    scale_str = f"{smpl_scale} {smpl_scale} {smpl_scale}"

    # 2. 确保 <asset> 标签存在
    asset = root.find("asset")
    if asset is None:
        asset = etree.Element("asset")
        root.insert(0, asset)

    # 3. 注册 Mesh 资源 (Asset 去重 + 添加 Scale)
    existing_mesh = asset.find(f"mesh[@name='{mesh_name}']")

    if existing_mesh is not None:
        # 如果已存在，更新路径和缩放
        existing_mesh.set('file', abs_mesh_path)
        existing_mesh.set('scale', scale_str)
    else:
        # 创建新的 mesh asset，包含 scale 属性
        etree.SubElement(
            asset, "mesh",
            name=mesh_name,
            file=abs_mesh_path,
            scale=scale_str  # <--- 关键修改：设置缩放
        )

    # 4. 确保 <worldbody> 存在
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = etree.SubElement(root, "worldbody")

    # 5. 添加 Object Body (Body 去重)
    existing_body = worldbody.find("body[@name='object_body']")
    if existing_body is not None:
        worldbody.remove(existing_body)

    # 添加新的 body
    body_el = etree.SubElement(worldbody, "body", name="object_body", mocap="true")

    # 添加 geom 引用 mesh
    etree.SubElement(
        body_el, "geom",
        name="object_geom",
        type="mesh",
        mesh=mesh_name,
        rgba="0.2 0.8 0.2 0.6",
        contype="0",
        conaffinity="0"
    )

    # 6. 保存
    dir_name = os.path.dirname(os.path.abspath(base_xml_path))
    base_name = os.path.basename(base_xml_path)
    # 文件名加上 scale 标识防止混淆，或者直接覆盖
    output_xml_path = os.path.join(dir_name, base_name.replace(".xml", f"_{mesh_name}_scaled.xml"))

    tree.write(output_xml_path, pretty_print=True, encoding="utf-8", xml_declaration=True)

    return output_xml_path
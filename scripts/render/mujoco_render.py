import os
import time
import mujoco
import imageio
from scipy.spatial.transform import Rotation as sRot
import tqdm

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


def export_mujoco_video_hoi(motion_traj, obj_pos, obj_quat_xyzw, xml_path, output_path="output.mp4", fps=30):
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
    camera.distance = 4.0  # 相机距离人的距离
    camera.elevation = -20  # 视角高度（仰角）
    camera.azimuth = 45  # 视角方位角

    num_frames = len(motion_traj['root_trans_offset'])

    def xyzw_to_wxyz(q): return q[[3, 0, 1, 2]]

    # 4. 获取物体 mocap ID
    obj_mocap_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object_body")
    obj_mocap_id = mj_model.body_mocapid[obj_mocap_bid]

    print(f"正在导出视频到: {output_path}...")

    with imageio.get_writer(output_path, fps=fps) as video:
        for t in tqdm.tqdm(range(num_frames)):
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
            camera.lookat[:] = mj_data.qpos[:3]
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

def create_temp_xml_with_object(base_xml_path, obj_mesh_path=None):
    """
    基于现有 humanoid xml，添加一个 mocap 物体 body。
    若 obj_mesh_path=None，则创建一个 box 占位。
    返回：新 xml 的路径（临时文件，可直接传给 mujoco.MjModel.from_xml_path）
    """
    tree = etree.parse(base_xml_path)
    root = tree.getroot()

    # 确保有 worldbody 和 asset
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = etree.SubElement(root, "worldbody")

    asset = root.find("asset")
    if asset is None:
        asset = etree.SubElement(root, "asset")

    # ---- 添加 mesh asset（可选） ----
    geom_type = "box"
    mesh_name = "MY_OBJ_MESH"
    obj_mesh_path = os.path.abspath(obj_mesh_path) if obj_mesh_path is not None else None
    if obj_mesh_path is not None and os.path.exists(obj_mesh_path):
        geom_type = "mesh"
        mesh_el = etree.SubElement(asset, "mesh", name=mesh_name, file=obj_mesh_path)

    # ---- 添加 mocap body ----
    body_el = etree.SubElement(worldbody, "body", name="object_body", mocap="true")

    if geom_type == "mesh":
        geom_el = etree.SubElement(
            body_el, "geom",
            name="object_geom", type="mesh", mesh=mesh_name,
            rgba="0.2 0.8 0.2 0.6", contype="0", conaffinity="0"
        )
    else:
        geom_el = etree.SubElement(
            body_el, "geom",
            name="object_geom", type="box", size="0.1 0.1 0.1",
            rgba="0.2 0.8 0.2 0.6", contype="0", conaffinity="0"
        )

    # ---- 写入临时文件 ----
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "humanoid_with_object.xml")
    tree.write(tmp_path, pretty_print=True, encoding="utf-8", xml_declaration=True)
    print(f"✅ 临时 XML 生成: {tmp_path}")
    return tmp_path

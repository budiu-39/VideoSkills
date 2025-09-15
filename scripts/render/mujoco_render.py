import os
import time
import mujoco
import imageio
from scipy.spatial.transform import Rotation as sRot


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

def vis_mujoco(motion_traj, xml_path, humanoid_type = 'g1'):

    print(mujoco.__version__)  # 应该输出 3.2.3
    print(hasattr(mujoco, "viewer"))

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    num_frames = len(motion_traj['root_trans_offset'])

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # 设置摄像机参数
        viewer.cam.lookat[:] = motion_traj['root_trans_offset'][0]  # 朝向机器人初始位置
        viewer.cam.distance = 3.0  # 相机距离，可调整
        viewer.cam.azimuth = -90  # 方位角（左侧视图）
        viewer.cam.elevation = -15  # 仰角（往下看）

        for t in range(num_frames):
            mj_data.qpos[:3] = motion_traj['root_trans_offset'][t]
            mj_data.qpos[3:7] = sRot.from_rotvec(motion_traj['pose_aa'][t][0]).as_quat()[[3, 0, 1, 2]]
            mj_data.qpos[7:] = motion_traj['dof'][t].flatten()
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            time.sleep(1 / 30)



import os
import sys
sys.path.append(os.getcwd())
import joblib
import mujoco.viewer
import argparse
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import imageio
import mediapy as media
import time

def vis_mujoco(motion_traj, humanoid_type = 'g1'):
    import mujoco
    import time
    print(mujoco.__version__)  # 应该输出 3.2.3
    print(hasattr(mujoco, "viewer"))
    print("MuJoCo version:", mujoco.__version__)
    print("mujoco has viewer:", hasattr(mujoco, "viewer"))
    print("mujoco loaded from:", mujoco.__file__)
    xml_path = f"phc/data/assets/robot/unitree_{humanoid_type}/{humanoid_type}.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    num_frames = len(motion_traj['root_states_seg'])
    opt = mujoco.MjvOption()

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # 设置相机视角
        viewer.cam.azimuth = -91.575
        viewer.cam.elevation = -10.6
        viewer.cam.distance = 3.126227
        viewer.cam.lookat[:] = np.array([-0.05651619, -0.00279599, 0.39617481])

        for t in range(num_frames):
            mj_data.qpos[:3] = motion_traj['root_states_seg'][t][:3]
            mj_data.qpos[3:7] = (motion_traj['root_states_seg'][t][3:7])[[3, 0, 1, 2]]
            mj_data.qpos[7:] = motion_traj['dof_pos'][t].flatten()
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            time.sleep(1/30)

        # 当你按 Esc 后，窗口关闭，这里继续执行
        print("\n📸 当前相机参数：")
        print("azimuth =", viewer.cam.azimuth)
        print("elevation =", viewer.cam.elevation)
        print("distance =", viewer.cam.distance)
        print("lookat =", viewer.cam.lookat)


import os
import os.path as osp
import joblib
import imageio
import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation as sRot
import mujoco

def _render_mujoco_offscreen_single(motion_traj: dict,
                                    output_video_path: str,
                                    humanoid_model_file: str,
                                    fps: int = 30,
                                    width: int = 1280,
                                    height: int = 720):
    """
    依据 motion_traj 渲染单个视频（离屏）。motion_traj 需包含：
      - 'root_states_seg': (N, ?) tensor/ndarray，[:3] 位置，[3:7] 根四元数(wxyz)
      - 'dof_pos':         (N, DoF, 1) 关节标量
    """
    os.makedirs(osp.dirname(output_video_path), exist_ok=True)

    # 1) 加载模型
    mj_model = mujoco.MjModel.from_xml_path(humanoid_model_file)
    mj_data  = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, width=width, height=height)

    root_states = motion_traj['root_states_seg']
    dof_pos     = motion_traj['dof_pos']
    # 兼容 torch/numpy
    if hasattr(root_states, "cpu"): root_states = root_states.cpu().numpy()
    if hasattr(dof_pos, "cpu"):     dof_pos     = dof_pos.cpu().numpy()

    num_frames = len(root_states)
    assert dof_pos.shape[0] == num_frames, "dof_pos 与 root_states_seg 帧数不一致"

    # 2) 设置相机（朝向初始前向）
    root_pos0      = root_states[0][:3]
    root_quat_wxyz = root_states[0][3:7]
    heading_vector = sRot.from_quat(root_quat_wxyz).apply([1, 0, 0])

    cam = mujoco.MjvCamera()
    cam.lookat[:]  = root_pos0
    cam.distance   = 3.0
    cam.elevation  = -10.0
    cam.azimuth    = np.rad2deg(np.arctan2(heading_vector[1], heading_vector[0])) + 180

    # 3) 写视频
    writer = imageio.get_writer(output_video_path, fps=fps)
    for t in range(num_frames):
        # qpos = [root_xyz(3), root_quat(wxyz->wxyz), dof...]
        mj_data.qpos[:3]  = root_states[t][:3]
        # MuJoCo 使用 wxyz；Renderer 里你之前转换到 [w, x, y, z] 顺序
        mj_data.qpos[3:7] = root_states[t][3:7][[0, 1, 2, 3]]  # 已是 wxyz 就无需重排
        mj_data.qpos[7:]  = dof_pos[t].reshape(-1)

        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam)
        frame = renderer.render()
        writer.append_data(frame)
    writer.close()


def render(ref_sim_data_path: str,
           output_path: str,
           use_offscreen: bool,
           raw_video_path: str = None,
           *,
           humanoid_model_file: str = "data/robots/g1_description/g1_29dof.xml",
           fps: int = 30):
    """
    与你 SMPL 版本 render 相同的 I/O 结构：
      - ref_sim_data_path: 单个 .pkl 或 目录（批量）
      - output_path:       输出目录（会在里面生成 <key>.mp4）
      - use_offscreen:     接口保持一致，这里总是使用 MuJoCo 离屏渲染以产出视频
      - raw_video_path:    （可选）若存在 <raw_video_path>/<key>/0_input_video.mp4，则与渲染结果拼接
      - humanoid_model_file: MuJoCo 模型 xml 路径
      - fps:               输出帧率
    约定：每个 pkl 里包含你训练时保存的 dict：
      - 'root_states_seg'： (N, 7+...) 其中 [:3] 位置；[3:7] 根四元数 (wxyz)
      - 'dof_pos'        ： (N, DoF, 1)
    """
    # 0) 收集文件
    if osp.isfile(ref_sim_data_path):
        pkl_files = [ref_sim_data_path]
        base_dir  = osp.dirname(ref_sim_data_path)
    elif osp.isdir(ref_sim_data_path):
        pkl_files = sorted([osp.join(ref_sim_data_path, f) for f in os.listdir(ref_sim_data_path) if f.endswith(".pkl")])
        base_dir  = ref_sim_data_path
    else:
        raise ValueError(f"Invalid path: {ref_sim_data_path}")

    os.makedirs(output_path, exist_ok=True)

    for pkl_file in pkl_files:
        motion_key = osp.splitext(osp.basename(pkl_file))[0]   # 不带后缀
        video_file = osp.join(output_path, f"{motion_key}.mp4")

        # 1) 读取 pkl
        ref_sim_data = joblib.load(pkl_file)
        # 允许数据直接就是 motion_traj，也允许嵌套存储
        if 'root_states_seg' in ref_sim_data and 'dof_pos' in ref_sim_data:
            motion_traj = ref_sim_data
        elif 'motion_traj' in ref_sim_data:
            motion_traj = ref_sim_data['motion_traj']
        else:
            raise KeyError(f"{pkl_file} 缺少 'root_states_seg' / 'dof_pos'。")

        # 2) 渲染（离屏）
        _render_mujoco_offscreen_single(
            motion_traj=motion_traj,
            output_video_path=video_file,
            humanoid_model_file=humanoid_model_file,
            fps=fps,
        )

        # 3) 可选与原视频拼接（保持你的 ffmpeg 逻辑）
        if raw_video_path:
            raw_dir   = Path(raw_video_path) / motion_key
            raw_input = raw_dir / "0_input_video.mp4"
            if raw_input.is_file():
                combo_video_file = osp.join(output_path, motion_key + "_stack.mp4")
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(raw_input),
                    "-i", str(video_file),
                    "-filter_complex",
                    "[0:v]scale=-2:1080,setsar=1[v0];[1:v]setsar=1[v1];[v0][v1]hstack=inputs=2[v]",
                    "-map", "[v]", "-map", "0:a?",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-r", str(fps), "-shortest",
                    str(combo_video_file),
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"Finished rendering {pkl_file} → {combo_video_file}")
                except subprocess.CalledProcessError as e:
                    print(f"[WARN] ffmpeg hstack failed for {motion_key}: {e}. Keep {video_file} only.")
                    print(f"Finished rendering {pkl_file} → {video_file}")
            else:
                print(f"[INFO] No raw video found at {raw_input}. Keep {video_file} only.")
                print(f"Finished rendering {pkl_file} → {video_file}")
        else:
            print(f"Finished rendering {pkl_file} → {video_file}")


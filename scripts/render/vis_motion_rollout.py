
import os
import sys
sys.path.append(os.getcwd())
import joblib

import argparse
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import imageio
import mediapy as media
import time

import os.path as osp
import joblib
import subprocess
from pathlib import Path
from datetime import datetime
import mujoco


def vis_mujoco(motion_traj, humanoid_type = 'g1'):
    import mujoco.viewer
    import time
    print(mujoco.__version__)  # 应该输出 3.2.3
    print(hasattr(mujoco, "viewer"))
    print("MuJoCo version:", mujoco.__version__)
    print("mujoco has viewer:", hasattr(mujoco, "viewer"))
    print("mujoco loaded from:", mujoco.__file__)
    xml_path = f"data//robots/{humanoid_type}/{humanoid_type}.xml"
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
        print("\n 当前相机参数：")
        print("azimuth =", viewer.cam.azimuth)
        print("elevation =", viewer.cam.elevation)
        print("distance =", viewer.cam.distance)
        print("lookat =", viewer.cam.lookat)

def make_root_states(pred_pos, pred_rot, quat_order="xyzw"):
    """
    用 pred_pos / pred_rot 的根关节 (joint 0) 构造 MuJoCo 的 root state.

    参数:
      pred_pos: (N, J, 3)
      pred_rot: (N, J, 4)
      quat_order: 'xyzw' 或 'wxyz'，取决于你 pred_rot 的存储顺序

    返回:
      root_states: (N, 7)，= [x, y, z, qw, qx, qy, qz] (MuJoCo 的 wxyz)
    """
    # 保证是 numpy
    if hasattr(pred_pos, "cpu"): pred_pos = pred_pos.cpu().numpy()
    if hasattr(pred_rot, "cpu"): pred_rot = pred_rot.cpu().numpy()

    # 第 0 个关节当根
    root_xyz = pred_pos[:, 0, :]   # (N,3)
    root_q   = pred_rot[:, 0, :]   # (N,4)

    if quat_order == "xyzw":
        # 转成 wxyz
        root_q_wxyz = np.stack([root_q[:, 3], root_q[:, 0], root_q[:, 1], root_q[:, 2]], axis=1)
    elif quat_order == "wxyz":
        root_q_wxyz = root_q
    else:
        raise ValueError("quat_order 必须是 'xyzw' 或 'wxyz'")

    root_states = np.concatenate([root_xyz, root_q_wxyz], axis=1)  # (N,7)
    return root_states


def _render_mujoco_offscreen_single(motion_traj: dict,
                                    output_video_path: str,
                                    humanoid_model_file: str,
                                    fps: int = 30,
                                    width: int = 1080,
                                    height: int = 1080):
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

    root_states = make_root_states(motion_traj['pred_pos'], motion_traj['pred_rot'], quat_order="xyzw")
    # 兼容 torch/numpy
    dof_pos = motion_traj['pred_dof_pos']


    num_frames = len(root_states)
    assert dof_pos.shape[0] == num_frames, "dof_pos 与 root_states_seg 帧数不一致"

    # 2) 设置相机（朝向初始前向）
    root_pos0      = root_states[0][:3]
    # root_quat_wxyz = root_states[0][3:]
    # heading_vector = sRot.from_quat(root_quat_wxyz).apply([1, 0, 0])

    cam = mujoco.MjvCamera()
    cam.lookat[:]  = root_pos0
    cam.distance   = 3.0
    cam.elevation  = -10.0
    cam.azimuth    = -90

    # 3) 写视频
    writer = imageio.get_writer(output_video_path, fps=fps)
    for t in range(num_frames):
        # qpos = [root_xyz(3), root_quat(wxyz->wxyz), dof...]
        mj_data.qpos[:3]  = root_states[t][:3]
        # MuJoCo 使用 wxyz；Renderer 里你之前转换到 [w, x, y, z] 顺序
        mj_data.qpos[3:7] = root_states[t][3:7]
        mj_data.qpos[7:]  = dof_pos[t].reshape(-1)

        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam)
        frame = renderer.render()
        writer.append_data(frame)
    writer.close()


def mujoco_render(ref_sim_data_path: str,
           output_path: str,
           use_offscreen: bool,
           raw_video_path: str = None,
           humanoid_model_file: str = "data/robots/g1/g1_29dof.xml",
           retargeted_video_path: str = None,
           fps: int = 50):

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
        if 'pred_pos' in ref_sim_data and 'pred_dof_pos' in ref_sim_data:
            motion_traj = ref_sim_data
        elif 'motion_traj' in ref_sim_data:
            motion_traj = ref_sim_data['motion_traj']
        else:
            raise KeyError(f"{pkl_file} need 'pred_pos' / 'pred_dof_pos'")

        # 2) 渲染（离屏）
        _render_mujoco_offscreen_single(
            motion_traj=motion_traj,
            output_video_path=video_file,
            humanoid_model_file=humanoid_model_file,
            fps=fps,
        )

        # 3) 可选与原视频拼接（保持你的 ffmpeg 逻辑）
        # 3) 可选与原视频拼接（raw | (retargeted?) | rollout）
        if raw_video_path:
            raw_dir   = Path(raw_video_path) / motion_key
            raw_input = raw_dir / "0_input_video.mp4"

            # 可选的 retargeted 视频：<retargeted_video_path>/<motion_key>.mp4
            retargeted_input = None
            if retargeted_video_path:
                cand = Path(retargeted_video_path) / f"{motion_key}.mp4"
                if cand.is_file():
                    retargeted_input = cand

            if raw_input.is_file():
                combo_video_file = osp.join(output_path, motion_key + "_stack.mp4")

                # 动态构造输入：总是把 rollout 放在最后
                inputs = [str(raw_input)]
                if retargeted_input:
                    inputs.append(str(retargeted_input))
                inputs.append(str(video_file))

                # 构造 -i 参数
                cmd = ["ffmpeg", "-y"]
                for iv in inputs:
                    cmd += ["-i", iv]

                # 构造 filter_complex：每路都 scale/setsar/format → [v0],[v1],...
                f_lines = []
                labels  = []
                for i in range(len(inputs)):
                    f_lines.append(f"[{i}:v]scale=-2:1080,setsar=1,format=yuv420p[v{i}]")
                    labels.append(f"[v{i}]")
                # 横向拼接
                f_lines.append(f"{''.join(labels)}hstack=inputs={len(inputs)}:shortest=1[v]")
                filter_complex = ";".join(f_lines)

                cmd += [
                    "-filter_complex", filter_complex,
                    "-map", "[v]",
                    "-map", "0:a?",          # 有音频就带上左路音频
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-vsync", "2",
                    "-r", str(fps),
                    "-shortest",
                    str(combo_video_file),
                ]

                try:
                    r = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    print(f"Finished rendering {pkl_file} → {combo_video_file}")
                except subprocess.CalledProcessError as e:
                    print(f"[FFMPEG ERROR] {e.stderr}")
                    print(f"[WARN] ffmpeg stack failed for {motion_key}. Keep {video_file} only.")
                    print(f"Finished rendering {pkl_file} → {video_file}")
            else:
                print(f"[INFO] No raw video found at {raw_input}. Keep {video_file} only.")
                print(f"Finished rendering {pkl_file} → {video_file}")
        else:
            print(f"Finished rendering {pkl_file} → {video_file}")


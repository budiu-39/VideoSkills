import os
import sys
import os.path as osp

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
from tqdm import tqdm
import argparse
import glob
from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from scripts.ms_utils import recover_from_local_rotation, smpl85_2_smpl322
import smplx
import joblib
import torch
import mujoco
import time
from scripts.preprocess.padding import pad_skeleton_state

import os
# 若没有图形界面，则切换到 EGL 离屏
if os.environ.get("DISPLAY", "") == "":
    os.environ["PYOPENGL_PLATFORM"] = "egl"   # 关键：改后端
    # 选 GPU（多卡时可按需指定）
    os.environ.setdefault("EGL_DEVICE_ID", "0")
    # 部分环境还需要这句确保找得到 EGL
    os.environ.setdefault("LD_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH",""))


def rot6d_to_rotmat(x):  # x: (..., 6)
    # Zhou et al. CVPR'19 的常见实现
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)
    return R

def rot6d_to_quat(x):  # x: (..., 6) -> (..., 4) in xyzw
    R = rot6d_to_rotmat(x)
    return sRot.from_matrix(R.reshape(-1, 3, 3)).as_quat().reshape(*x.shape[:-1], 4)

def quat_mul_xyzw(q, r):
    """
    Hamilton product q * r，均为(...,4)[x,y,z,w]，返回同形状。
    支持批量和广播，例如 (T,1,4) 与 (T,J,4)。
    """
    x1, y1, z1, w1 = np.split(q, 4, axis=-1)
    x2, y2, z2, w2 = np.split(r, 4, axis=-1)
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.concatenate([x, y, z, w], axis=-1)

def pose_272_to_smpl(data_272):
    smpl_85_data = recover_from_local_rotation(data_272, 22)  # get the 85-dim smpl data
    if len(smpl_85_data.shape) == 3:
        smpl_85_data = np.squeeze(smpl_85_data, axis=0)

    pose = smpl85_2_smpl322(smpl_85_data)

    assert pose.shape[1] == 322
    use_flame = (pose.shape[1] == 322)

    root_and_body = pose[:, :66].reshape(-1, 22, 3)

    if use_flame:
        trans = pose[:, 309:309 + 3]
    else:
        trans = pose[:, 159:159 + 3]

    trans = trans.reshape(-1, 3)
    return trans, root_and_body

import os, glob, cv2, time, joblib, numpy as np
import pyrender, trimesh
from scipy.spatial.transform import Rotation as sRot

def smpl_forward_vertices(smpl_model, root_trans, pose_aa, betas=None, device='cuda'):
    T = root_trans.shape[0]
    if betas is None:
        betas = torch.zeros(1, 10, device=device, dtype=torch.float32)
    if betas.ndim == 1:
        betas = betas[None]

    body_pose = torch.tensor(pose_aa[:, 1:24, :].reshape(T, -1), dtype=torch.float32, device=device)
    global_orient = torch.tensor(pose_aa[:, 0:1, :].reshape(T, 3), dtype=torch.float32, device=device)
    transl = torch.tensor(root_trans, dtype=torch.float32, device=device)

    out = smpl_model(betas=betas.expand(T, -1),
                     body_pose=body_pose,
                     global_orient=global_orient,
                     transl=transl,
                     pose2rot=True)
    verts = out.vertices.detach().cpu().numpy()
    return verts

# ---------------- 相机构建（K,R,t 或弱透视） ----------------
def add_camera_KRt(scene, K, R, t):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
    # OpenCV->OpenGL 坐标修正
    cv2_to_gl = np.eye(4); cv2_to_gl[1,1] *= -1; cv2_to_gl[2,2] *= -1
    ext = np.eye(4); ext[:3,:3] = R; ext[:3,3] = t.reshape(3)
    pose = ext @ cv2_to_gl
    cam_node = scene.add(cam, pose=pose)
    light_node = scene.add(pyrender.DirectionalLight(intensity=2.0), pose=pose)
    return cam_node, light_node

def add_camera_weak(scene, dist=2.5, fov_deg=30.0):
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov_deg))
    pose = np.eye(4); pose[:3,3] = np.array([0,0,dist])
    cam_node = scene.add(cam, pose=pose)
    light_node = scene.add(pyrender.DirectionalLight(intensity=2.0), pose=pose)
    return cam_node, light_node

def _look_at(eye, target, up=np.array([0, 1, 0], dtype=float)):
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    u = up / (np.linalg.norm(up) + 1e-8)
    s = np.cross(f, u);  s = s / (np.linalg.norm(s) + 1e-8)
    u = np.cross(s, f)
    # OpenGL 视图矩阵（相机看向 -Z）：把相机位姿塞进 4x4
    M = np.eye(4)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = eye
    return M

def add_camera_autofit(scene, verts0, fov_deg=35.0):
    # 基于首帧顶点：估计中心与尺度
    vmin = verts0.min(axis=0)
    vmax = verts0.max(axis=0)
    center = (vmin + vmax) * 0.5
    diag = np.linalg.norm(vmax - vmin) + 1e-8
    radius = diag * 0.5

    # 距离 = 半径 / tan(fov/2)，再乘系数放松一点
    dist = radius / np.tan(np.deg2rad(fov_deg) * 0.5) * 1.5
    # 稍微从上往下看一点，避免只见腿
    eye = center + np.array([0.0, 0.2 * radius, dist], dtype=float)
    cam_pose = _look_at(eye, center)

    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov_deg))
    cam_node = scene.add(cam, pose=cam_pose)
    light_node = scene.add(pyrender.DirectionalLight(intensity=2.0), pose=cam_pose)
    return cam_node, light_node

def add_ground_plane(scene, y=0.0, size=10.0, thickness=0.02):
    plane = trimesh.creation.box(extents=(size, thickness, size))
    plane.apply_translation([0.0, y - thickness * 0.5, 0.0])
    ground = pyrender.Mesh.from_trimesh(plane, smooth=False)
    return scene.add(ground)

# ---------------- 渲染单段序列（可拼接视频） ----------------
def render_sequence(verts_seq, faces, out_writer, width, height,
                    concat=False, video_reader=None, K=None, Rt_seq=None,
                    fps_data=45.0, fps_video=None, video_time_offset=0.0):
    scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[0.3,0.3,0.3])
    if K is not None and Rt_seq is not None:
        if Rt_seq.shape[-1] == 4:
            R0, t0 = Rt_seq[0][:,:3], Rt_seq[0][:,3]
        else:
            R0, t0 = Rt_seq[0][0], Rt_seq[0][1]
        cam_node, light_node = add_camera_KRt(scene, K, R0, t0)
        use_krt = True
    else:
        cam_node, light_node = add_camera_autofit(scene, verts_seq[0])
        use_krt = False

    add_ground_plane(scene, y=0.0, size=10.0, thickness=0.02)

    template = trimesh.Trimesh(vertices=np.zeros((verts_seq.shape[1],3)),
                               faces=faces, process=False, maintain_order=True)
    mesh_node = scene.add(pyrender.Mesh.from_trimesh(template, smooth=False))

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    # 贴地可视化
    verts_seq[..., 1] -= verts_seq[..., 1].min()

    if concat and video_reader is not None:
        # —— 以视频为主时钟：一帧视频 ↔ 一次渲染 ——
        j_video = 0
        while True:
            ok, frame_bgr = video_reader.read()
            if not ok:
                break
            # 当前视频时间（秒）+ 可微调偏移
            t = j_video / float(fps_video) + float(video_time_offset)
            i_motion = int(round(t * float(fps_data)))
            if i_motion < 0:
                j_video += 1
                continue
            if i_motion >= len(verts_seq):
                break

            v = verts_seq[i_motion].copy()
            tri = trimesh.Trimesh(vertices=v, faces=faces, process=False, maintain_order=True)
            new_mesh = pyrender.Mesh.from_trimesh(tri, smooth=False)
            scene.remove_node(mesh_node)
            mesh_node = scene.add(new_mesh)

            color, _ = renderer.render(scene)
            render_bgr = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)

            if frame_bgr.shape[:2] != (height, width):
                frame_bgr = cv2.resize(frame_bgr, (width, height))
            side = np.hstack([frame_bgr, render_bgr])
            out_writer.write(side)

            j_video += 1
    else:
        # —— 不拼接视频：把 45Hz 的运动按 30Hz 下采样输出（或直接 45fps 输出，见下方）——
        # 这里示例：按 30Hz 的目标时间取样运动（k/30）
        out_fps = 30.0  # 如果你想直接输出 45fps，就把这段改成遍历 i=0..len(verts_seq)-1
        n_out = int(np.floor(len(verts_seq) * (out_fps / float(fps_data))))
        for k in range(n_out):
            t = k / out_fps
            i_motion = int(round(t * float(fps_data)))
            i_motion = min(i_motion, len(verts_seq) - 1)
            v = verts_seq[i_motion].copy()

            tri = trimesh.Trimesh(vertices=v, faces=faces, process=False, maintain_order=True)
            new_mesh = pyrender.Mesh.from_trimesh(tri, smooth=False)
            scene.remove_node(mesh_node)
            mesh_node = scene.add(new_mesh)

            color, _ = renderer.render(scene)
            render_bgr = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)
            out_writer.write(render_bgr)

    renderer.delete()



def get_data_len_and_duration(data_272, fps_data=30.0):
    """从 (T, 272) 或 (T, …) 的 numpy 数组里读出帧数与时长"""
    T_data = int(data_272.shape[0])
    dur_data = T_data / float(fps_data)
    return T_data, dur_data


if __name__ == "__main__":
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--npy_glob", required=True, help="e.g. '/path/motion/*.npy'")
    # ap.add_argument("--video", type=str, default=None, help="原视频路径，可选")
    # ap.add_argument("--concat", action="store_true", help="是否左右拼接（左=视频，右=渲染）")
    # ap.add_argument("--out", required=True, help="输出 mp4")
    # ap.add_argument("--fps", type=float, default=30.0)
    # ap.add_argument("--smpl_dir", required=True, help="SMPL 模型目录（包含 SMPL_NEUTRAL 等）")
    # ap.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    # ap.add_argument("--K_npy", type=str, default=None, help="相机内参 K.npy，可选")
    # ap.add_argument("--Rt_npy", type=str, default=None, help="每帧外参 Rt.npy，可选，形状 (T,3,4) 或 list[(R,t)]")
    # args = ap.parse_args()

    npy_glob = "/mnt/lustre/work/ponsmoll/pba936/VideoSkills/MotionMillion/motion_272rpr/MotionUnion/kungfu"
    out = "/mnt/lustre/work/ponsmoll/pba936/VideoSkills/kungfu_272_video"
    seq_files = sorted(glob.glob(os.path.join(npy_glob, '*.npy')))
    source_video_path = '/mnt/lustre/work/ponsmoll/pba936/Video/kungfu_video'
    smpl_dir = "data/SMPL"

    # npy_glob = "/home/miku/Documents/Dataset/kungfu"
    # out = "/home/miku/Documents/VideoSkills/rollout"
    # seq_files = sorted(glob.glob(os.path.join(npy_glob, '*.npy')))
    # source_video_path = '/home/miku/Documents/Video/kungfu_video'
    # smpl_dir = "data/SMPL"


    # 打开视频（若需要拼接）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smpl_model = smplx.create(
        model_path=smpl_dir,
        model_type='smpl',
        gender='neutral',
        use_pca=False
    ).to(device)
    faces = smpl_model.faces

    # 相机参数（可选）
    K_npy = None
    Rt_npy = None
    K = np.load(K_npy) if K_npy else None
    Rt_seq = np.load(Rt_npy, allow_pickle=True) if Rt_npy else None

    # ==== 每个 npy 单独输出 ====
    for f in seq_files:
        name = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(out, f"{name}.mp4")
        print(f"[Render] {f} -> {out_path}")

        cap = None
        width = 720;
        height = 720
        fps_data = 45.0  # 你的 SMPL/数据帧率
        fps_video = None

        if source_video_path is not None:
            full_source_video_path = os.path.join(source_video_path, f"{name}.mp4")
            cap = cv2.VideoCapture(full_source_video_path)
            assert cap.isOpened(), f"Cannot open video: {full_source_video_path}"
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_video = cap.get(cv2.CAP_PROP_FPS)

        # —— 关键：输出 fps 用“数据帧率”，这样拼接后严格对齐 ——
        out_fps = fps_video
        out_size = (width * 2, height) if (source_video_path is not None) else (width, height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, out_size)

        # 转换并渲染
        data_272 = np.load(f, allow_pickle=True)
        root_trans, pose_aa = pose_272_to_smpl(data_272)
        T, J, C = pose_aa.shape
        pad = np.zeros((T, 2, 3), dtype=pose_aa.dtype)
        pose_aa = np.concatenate([pose_aa, pad], axis=1)
        verts_seq = smpl_forward_vertices(smpl_model, root_trans, pose_aa, device=device)

        render_sequence(
            verts_seq=verts_seq,
            faces=faces,
            out_writer=writer,
            width=width,
            height=height,
            concat=(source_video_path is not None),
            video_reader=cap,
            K=K, Rt_seq=Rt_seq,
            fps_data=fps_data,
            fps_video=fps_video,  # 仅用于估时长/打印，可留着
            video_time_offset=0.0  # 需要时改成正负几百毫秒
        )

        writer.release()
        if cap is not None:
            cap.release()
        print(f"[OK] saved to {out_path}")



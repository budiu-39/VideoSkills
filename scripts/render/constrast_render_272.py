import trimesh

from videoskills.utils.poselib.skeleton.skeleton3d import SkeletonTree
from scripts.render.utils.torch import copy2cpu as c2c
from scripts.render.viz.mesh_viewer import MeshViewer
from scripts.render.viz.utils import create_video
from scripts.render.viz.utils import smpl_connections

from smplx import SMPL
import joblib
import cv2
import pyrender
from datetime import datetime
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

import argparse
from scipy.spatial.transform import Rotation as sRot
import torch
from scripts.retarget.smpl_humanoid_tool import humanoid2smpl

import tempfile, shutil
import subprocess
from pathlib import Path
import glob
from videoskills.utils.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from videoskills.utils import torch_utils

from videoskills.utils.torch_utils import exp_map_to_quat, quat_to_exp_map  # 可直接用你项目里那套

def _time_remap_indices(T_in, fps_in, fps_out):
    dur = (T_in - 1) / fps_in  # 用时长对齐（首末帧覆盖同一时间段）
    T_out = int(round(dur * fps_out)) + 1
    t_out = torch.linspace(0, dur, T_out)  # 秒
    src_idx = t_out * fps_in  # 浮点帧号
    i0 = torch.clamp(src_idx.floor().long(), max=T_in - 1)
    i1 = torch.clamp(i0 + 1, max=T_in - 1)
    w  = (src_idx - i0.float()).unsqueeze(-1)  # 插值权重
    return i0, i1, w, T_out

@torch.no_grad()
def resample_transl(transl, fps_in, fps_out):
    # transl: (T, 3)
    T_in = transl.shape[0]
    i0, i1, w, T_out = _time_remap_indices(T_in, fps_in, fps_out)
    out = (1 - w) * transl[i0] + w * transl[i1]
    return out

@torch.no_grad()
def resample_pose_axis_angle(pose_aa, fps_in, fps_out):
    # pose_aa: (T, 24, 3)  轴角/exp-map
    T_in, J, _ = pose_aa.shape
    i0, i1, w, T_out = _time_remap_indices(T_in, fps_in, fps_out)
    # -> quat: (T, J, 4)
    q0 = exp_map_to_quat(pose_aa[i0])   # 或者把轴角转 quat
    q1 = exp_map_to_quat(pose_aa[i1])
    # SLERP 逐关节
    wJ = w  # (T_out, 1)
    q = torch_utils.slerp(q0.view(-1, 4), q1.view(-1, 4), wJ.repeat(1, J).view(-1, 1)).view(-1, J, 4)
    # -> 轴角/exp-map
    aa = quat_to_exp_map(q)
    return aa


class Body:
    def __init__(self, faces, vertices, jtr):
        self.f = torch.tensor(faces)
        self.v = torch.tensor(vertices)
        self.jtr = torch.tensor(jtr)

def viz_contrast_smpl_seq(sim_body, ref_body, imw=1080, imh=1080, fps=30, contacts=None,
                render_body=True, render_joints=False, render_skeleton=False, render_ground=True, ground_plane=None,
                use_offscreen=False, out_path=None, wireframe=False, RGBA=False,
                joints_seq=None, joints_vel=None, follow_camera=False, vtx_list=None, points_seq=None, points_vel=None,
                static_meshes=None, camera_intrinsics=None, img_seq=None, point_rad=0.015,
                skel_connections=smpl_connections, img_extn='png', ground_alpha=1.0, body_alpha=None, mask_seq=None,
                cam_offset=[0.0, 4.0, 1.25], ground_color0=[0.8, 0.9, 0.9], ground_color1=[0.6, 0.7, 0.7],
                skel_color=[0.0, 0.0, 1.0],
                joint_rad=0.015,
                point_color=[0.0, 1.0, 0.0],
                joint_color=[0.0, 1.0, 0.0],
                contact_color=[1.0, 0.0, 0.0],
                render_bodies_static=None,
                render_points_static=None,
                cam_rot=None, sim_color = [0.2, 0.8, 0.2], ref_color = [0.8, 0.2, 0.2]):
    '''
    Visualizes the body model output of a smpl sequence.
    - body : body model output from SMPL forward pass (where the sequence is the batch)
    - joints_seq : list of torch/numy tensors/arrays
    - points_seq : list of torch/numpy tensors
    - camera_intrinsics : (fx, fy, cx, cy)
    - ground_plane : [a, b, c, d]
    - render_bodies_static is an integer, if given renders all bodies at once but only every x steps
    '''


    # if contacts is not None and torch.is_tensor(contacts):
    #     contacts = c2c(contacts)
    #
    if render_body or vtx_list is not None:
        if sim_body is not None:
            nv = sim_body.v.size(1)
            vertex_colors = np.tile(sim_color, (nv, 1))
            if body_alpha is not None:
                vtx_alpha = np.ones((vertex_colors.shape[0], 1)) * body_alpha
                vertex_colors = np.concatenate([vertex_colors, vtx_alpha], axis=1)
            faces = c2c(sim_body.f)
            sim_body_mesh_seq = [
                trimesh.Trimesh(vertices=c2c(sim_body.v[i]), faces= sim_body.f, vertex_colors = sim_color, process=False)
                for i in range(sim_body.v.size(0))]

        if ref_body is not None:
            nv = ref_body.v.size(1)
            vertex_colors = np.tile(ref_color, (nv, 1))
            if body_alpha is not None:
                vtx_alpha = np.ones((vertex_colors.shape[0], 1)) * body_alpha
                vertex_colors = np.concatenate([vertex_colors, vtx_alpha], axis=1)
            faces = c2c(ref_body.f)
            body_mesh_seq = [
                trimesh.Trimesh(vertices=c2c(ref_body.v[i]), faces=ref_body.f, vertex_colors = ref_color,
                                  process=False)
                for i in range(ref_body.v.size(0))]

    mv = MeshViewer(width=imw, height=imh,
                    use_offscreen=use_offscreen,
                    follow_camera=follow_camera,
                    camera_intrinsics=camera_intrinsics,
                    img_extn=img_extn,
                    default_cam_offset=cam_offset,
                    default_cam_rot=cam_rot)

    # mv.add_axes()

    if render_body and render_bodies_static is None:
        if sim_body is not None:
            mv.add_mesh_seq(sim_body_mesh_seq)
        mv.add_mesh_seq(body_mesh_seq)
    elif render_body and render_bodies_static is not None:
        mv.add_static_meshes([sim_body_mesh_seq[i] for i in range(len(sim_body_mesh_seq)) if i % render_bodies_static == 0])
        mv.add_static_meshes([body_mesh_seq[i] for i in range(len(body_mesh_seq)) if i % render_bodies_static == 0])
    if render_joints and render_skeleton:
        mv.add_point_seq(joints_seq, color=joint_color, radius=joint_rad, contact_seq=contacts,
                         connections=skel_connections, connect_color=skel_color, vel=joints_vel,
                         contact_color=contact_color, render_static=render_points_static)
    elif render_joints:
        mv.add_point_seq(joints_seq, color=joint_color, radius=joint_rad, contact_seq=contacts, vel=joints_vel, contact_color=contact_color,
                            render_static=render_points_static)

    if vtx_list is not None:
        mv.add_smpl_vtx_list_seq(sim_body_mesh_seq, vtx_list, color=[0.0, 0.0, 1.0], radius=0.015)
        mv.add_smpl_vtx_list_seq(body_mesh_seq, vtx_list, color=[0.0, 0.0, 1.0], radius=0.015)

    if points_seq is not None:
        if torch.is_tensor(points_seq[0]):
            points_seq = [c2c(point_frame) for point_frame in points_seq]
        mv.add_point_seq(points_seq, color=point_color, radius=point_rad, vel=points_vel, render_static=render_points_static)

    if static_meshes is not None:
        mv.set_static_meshes(static_meshes)

    if img_seq is not None:
        mv.set_img_seq(img_seq)

    if mask_seq is not None:
        mv.set_mask_seq(mask_seq)

    if render_ground:
        xyz_orig = None
        if ground_plane is not None:
            if render_body:
                xyz_orig = sim_body_mesh_seq[0].vertices[0, :]
                xyz_orig = body_mesh_seq[0].vertices[0, :]
            elif render_joints:
                xyz_orig = joints_seq[0][0, :]
            elif points_seq is not None:
                xyz_orig = points_seq[0][0, :]

        mv.add_ground(ground_plane=ground_plane, xyz_orig=xyz_orig, color0=ground_color0, color1=ground_color1, alpha=ground_alpha)

    mv.set_render_settings(out_path=out_path, wireframe=wireframe, RGBA=RGBA,
                            single_frame=(render_points_static is not None or render_bodies_static is not None)) # only does anything for offscreen rendering
    try:
        mv.animate(fps=fps)
    except RuntimeError as err:
        print('Could not render properly with the error: %s' % (str(err)))

    del mv

import torch

def aa_to_R(aa):  # (N,3) -> (N,3,3)
    # Rodrigues
    theta = torch.linalg.norm(aa + 1e-8, dim=-1, keepdim=True)
    k = torch.nan_to_num(aa / theta)
    K = torch.zeros(aa.shape[0], 3, 3, device=aa.device, dtype=aa.dtype)
    K[:,0,1], K[:,0,2] = -k[:,2],  k[:,1]
    K[:,1,0], K[:,1,2] =  k[:,2], -k[:,0]
    K[:,2,0], K[:,2,1] = -k[:,1],  k[:,0]
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).unsqueeze(0)
    R = I + torch.sin(theta).unsqueeze(-1)*K + (1 - torch.cos(theta)).unsqueeze(-1)*(K@K)
    return R

def R_to_aa(R):  # (N,3,3) -> (N,3)
    cos = ((R[:,0,0]+R[:,1,1]+R[:,2,2]) - 1)/2
    cos = torch.clamp(cos, -1, 1)
    theta = torch.acos(cos)
    eps = 1e-8
    s = torch.sqrt(torch.clamp(1 - cos**2, 0, 1))
    kx = (R[:,2,1] - R[:,1,2]) / (2*torch.where(s.abs()<eps, torch.ones_like(s), s))
    ky = (R[:,0,2] - R[:,2,0]) / (2*torch.where(s.abs()<eps, torch.ones_like(s), s))
    kz = (R[:,1,0] - R[:,0,1]) / (2*torch.where(s.abs()<eps, torch.ones_like(s), s))
    k = torch.stack([kx,ky,kz], dim=-1)
    aa = torch.nan_to_num(k * theta.unsqueeze(-1))
    return aa

def basis_change_pose_aa(pose_aa, B):
    """
    pose_aa: (T, J, 3) axis-angle, pelvis at index 0
    B: (3,3) basis matrix (e.g., swap Y/Z)
    """
    T, J, _ = pose_aa.shape
    aa_flat = pose_aa.reshape(-1, 3)
    R = aa_to_R(aa_flat)                    # (T*J,3,3)
    BRB = (B @ R @ B.T)                     # basis change
    aa_new = R_to_aa(BRB).reshape(T, J, 3)
    return aa_new

def basis_change_transl(transl, swap='yz'):
    # transl: (T,3)
    if swap == 'yz':
        return transl[..., [0,2,1]]
    elif swap == 'xy':
        return transl[..., [1,0,2]]
    elif swap == 'xz':
        return transl[..., [2,1,0]]
    else:
        raise ValueError

import os.path as osp
from scripts.preprocess.mm2smpl import pose_272_to_smpl

def render(motionx_path, mm_path, output_path, use_offscreen, raw_video_path: str = None):
    MODEL_PATH = 'data/SMPL/smpl'
    smpl = SMPL(MODEL_PATH, gender='MALE', batch_size=1)
    # 判断输入是文件还是目录
    motionx_files = sorted(glob.glob(os.path.join(motionx_path,'*.npy')))
    motion_keys = []
    for file in motionx_files:
        curr_motion = SkeletonMotion.from_file(file)
        parts = file.split(os.sep)
        key_name = os.path.splitext(parts[-1])[0]

        ref_data = {}
        sim_data = {}
        ref_data['body_rot'] = curr_motion.global_rotation
        ref_data['transl'] =  curr_motion.global_translation[:, 0]

        fpath = osp.join(mm_path, f"{key_name}.npy")
        motion = np.load(fpath, allow_pickle=True)
        N = motion.shape[0]
        root_trans, pose_aa = pose_272_to_smpl(motion)
        # 加入降采样代码
        # 经典字段划分
        pose_aa_smpl = np.zeros((N, 24, 3), dtype=pose_aa.dtype)
        pose_aa_smpl[:, :22, :] = pose_aa
        sim_data['body_rot'] = torch.from_numpy(pose_aa_smpl).float()
        sim_data['transl'] = torch.from_numpy(root_trans).float()

        # fpath = osp.join(mm_path, "hmr4d_results.pt")
        # with open(fpath, 'rb') as f:
        #     motion = torch.load(f, map_location='cpu')
        # N = motion['smpl_params_global']['body_pose'].shape[0]
        # # 加入降采样代码
        # # 经典字段划分 # 绿
        # pose_aa = motion['smpl_params_global']['body_pose']
        # root_trans =  motion['smpl_params_global']['transl']
        # pose_aa_smpl = np.zeros((N, 24, 3), dtype=np.float32)
        # pose_aa_smpl[:, 1:22, :] = pose_aa.reshape(-1, 21, 3)
        # pose_aa_smpl[:, 0] = motion['smpl_params_global']['global_orient']
        # sim_data['body_rot'] = torch.from_numpy(pose_aa_smpl).float()
        # sim_data['transl'] = root_trans

        skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/smpl_humanoid.xml")

        ref_fps = 30
        sim_fps = 30
        target_fps = 30  # 统一到 30fps（也可统一到 45）

        ref_pose_aa, ref_transl = humanoid2smpl(ref_data['body_rot'], ref_data['transl'], skeleton_tree)

        ref_pose_aa = resample_pose_axis_angle(ref_pose_aa.view(-1,24,3), ref_fps, target_fps)
        sim_pose_aa = resample_pose_axis_angle(sim_data['body_rot'], sim_fps, target_fps)
        ref_transl = resample_transl(ref_transl, ref_fps, target_fps)
        sim_transl = resample_transl(sim_data['transl'], sim_fps, target_fps)

        T = min(ref_pose_aa.shape[0], sim_pose_aa.shape[0])
        ref_pose = ref_pose_aa[:T]
        sim_pose = sim_pose_aa[:T]
        ref_transl = ref_transl[:T]
        sim_transl = sim_transl[:T]

        # B = torch.tensor([[1., 0., 0.],
        #                   [0., 0., 1.],
        #                   [0., 1., 0.]], device=sim_pose_aa.device, dtype=sim_pose_aa.dtype)

        # sim_pose[:, 0, 1]
        # sim_transl = basis_change_transl(sim_transl, swap='yz')
        # sim_pose[:, 0, [1, 2]] = sim_pose[:, 0, [2, 1]]
        # sim_transl[:,0] =  -sim_transl[:,0]

        ref_pose = ref_pose.view(T, -1)
        sim_pose = sim_pose.view(T, -1)

# Yaw 取反
        fmt = "%b%d_%H:%M"
        timestamp = datetime.now().strftime(fmt)
        # pic_path = os.path.join(output_path, f'pic/{key_name}_{timestamp}')
        video_path = os.path.join(output_path)
        # video_path = os.path.join(output_path, f'video/{key_name}_{timestamp}')
        # os.makedirs(pic_path, exist_ok=1)
        os.makedirs(video_path, exist_ok=1)

        # sim_transl = ref_transl + torch.tensor([2.0, 0.0, 0.0])
        betas = torch.zeros(1, 10)

        # === Ref body ===
        ref_pose = ref_pose.float()
        output = smpl(betas=betas, body_pose=ref_pose[:,1:], global_orient=ref_pose[:, :1], pose2rot=True, transl=ref_transl)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        jtr = output.joints.detach().cpu().numpy().squeeze()
        ref_body = Body(smpl.faces.astype(np.int64), vertices, jtr)
        # ref_body = None

        # === Sim body ===
        sim_pose = sim_pose.float()
        output = smpl(betas=betas, body_pose=sim_pose[:, 1:], global_orient=sim_pose[:, :1], pose2rot=True,
                      transl=sim_transl)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        jtr = output.joints.detach().cpu().numpy().squeeze()
        sim_body = Body(smpl.faces.astype(np.int64), vertices, jtr)
        # sim_body = None

        tmp_dir = tempfile.mkdtemp(prefix="render_frames_")  # 临时帧目录
        pic_file = os.path.join(tmp_dir, key_name)
        video_file = os.path.join(video_path, key_name + '.mp4')
        combo_video_file  = os.path.join(video_path, key_name + '_with_raw.mp4')  # 左右拼接后的最终视频

        mat = np.load(os.path.join(os.getcwd(), "scripts", "render", "camera_pos.npy"))
        viz_contrast_smpl_seq(
            sim_body, ref_body, imw=1080, imh=1080, fps=30, contacts=None,
            render_body=True, render_joints=False, render_skeleton=False, render_ground=True,
            ground_plane=None,
            use_offscreen=use_offscreen, out_path= pic_file, wireframe=False, RGBA=False,
            joints_seq=None, joints_vel=None, follow_camera=False, vtx_list=None, points_seq=None,
            points_vel=None,
            static_meshes=None, camera_intrinsics=None, img_seq=None, point_rad=0.015,
            skel_connections=smpl_connections, img_extn='png', ground_alpha=1.0, body_alpha=1.0,
            mask_seq=None,
            cam_offset=mat[:3,3], ground_color0=[0.8, 0.9, 0.9], ground_color1=[0.6, 0.7, 0.7],
            skel_color=[1.0, 0.0, 1.0],
            joint_rad=0.015,
            point_color=[1.0, 1.0, 1.0],
            joint_color=[1.0, 1.0, 0.0],
            contact_color=[1.0, 0.0, 0.0],
            render_bodies_static=None,
            render_points_static=None,
            cam_rot=mat[:3,:3])

        create_video(pic_file + '/frame_%08d.png', video_file, 30)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if raw_video_path:
            raw_dir = Path(raw_video_path) / key_name
            raw_input = raw_dir / "0_input_video.mp4"
            if raw_input.is_file():
                # 竖向拼接：两路都缩放到同宽 960（高度自适应为偶数），然后 vstack
                combo_video_file = os.path.join(video_path, key_name + '_stack.mp4')
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(raw_input),
                    "-i", str(video_file),
                    "-filter_complex",
                    # 先把左侧缩放到同高 1080（宽自适应，偶数），两路都设 sar=1，最后 hstack
                    "[0:v]scale=-2:1080,setsar=1[v0];[1:v]setsar=1[v1];[v0][v1]hstack=inputs=2[v]",
                    "-map", "[v]", "-map", "0:a?",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-r", "30", "-shortest",
                    str(combo_video_file)
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"Finished rendering {file} → {combo_video_file}")
                except subprocess.CalledProcessError as e:
                    print(f"[WARN] ffmpeg vstack failed for {key_name}: {e}. Keep {video_file} only.")
                    print(f"Finished rendering {file} → {video_file}")
            else:
                print(f"[INFO] No raw video found at {raw_input}. Keep {video_file} only.")
                print(f"Finished rendering {file} → {video_file}")
        else:
            print(f"Finished rendering {file} → {video_file}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Render SMPL sequence')
    parser.add_argument("--use_offscreen", action='store_true', help="flag to mark if input is test or not ")
    parser.add_argument("--raw_video_path", type=str, required=False, help="raw video path for combining")
    opt = parser.parse_args()  #logs/smpl_ppo/refinement_folder_136_resume_Sep09_05-05-30/rollouts/failed

    if opt.use_offscreen:
        if os.environ.get("DISPLAY", "") == "":
            os.environ["PYOPENGL_PLATFORM"] = "egl"  # 若无 NVIDIA EGL，可改用 'osmesa'
            os.environ["PYGLET_HEADLESS"] = "True"

    # gvhmr_path = 'dataset/smpl_motion/motionx++_kungfu'
    output_path = 'demo'
    # gvhmr_path =  '/home/miku/Documents/VideoSkills/dataset/smpl_motion/test'  # GVHMR
    # motionx_path = '/home/miku/Documents/VideoSkills/demo/demo'
    # motionx_path = '/home/miku/Documents/VideoSkills/demo/demo'
    # gvhmr_pt = '/home/miku/Documents/VideoSkills/demo/Aerial_Kick_Kungfu_wushu_10_clip1'  # GVHMR
    mm2smpl_path = 'dataset/smpl_motion/kungfu'
    rep272_path = 'dataset/MotionMillion/kungfu'
    render(mm2smpl_path, rep272_path, output_path, opt.use_offscreen, opt.raw_video_path)
import os
import numpy as np
import torch
import trimesh
import sys
from smplx import SMPL
from scipy.spatial.transform import Rotation as sRot

# ---------------- 配置参数 ---------------- #
model_path = "/home/miku/Documents/VideoSkills/data/smpl"  # 修改为你的SMPL模型路径
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# SMPL 初始化
smpl_model = SMPL(
    model_path=model_path,
    gender='neutral',
    batch_size=1,
    use_pca=False,
    flat_hand_mean=True
)

# ---------------- ZeroPose 和 Toe Pose ---------------- #
pose_aa_zero = np.zeros((1, 72), dtype=np.float32)
pose_aa_toe = pose_aa_zero.copy()

# SMPL 索引：脚尖关节编号（默认最后一项）
R_TOE_IDX = 10
L_TOE_IDX = 11
R_ANKLE_IDX = 8
L_ANKLE_IDX = 7

# 对 Toe 进行旋转：绕 X 轴旋转 30°
toe_angle_rad = np.deg2rad(-45)
pose_aa_toe[0, R_TOE_IDX * 3 + 0] = toe_angle_rad
pose_aa_toe[0, L_TOE_IDX * 3 + 0] = toe_angle_rad
toe_verts_r = np.where(np.argmax(smpl_model.lbs_weights.numpy(), axis=1) == R_TOE_IDX)[0]
toe_verts_l = np.where(np.argmax(smpl_model.lbs_weights.numpy(), axis=1) == L_TOE_IDX)[0]
toe_verts = np.concatenate([toe_verts_r, toe_verts_l])

ankle_verts_r = np.where(np.argmax(smpl_model.lbs_weights.numpy(), axis=1) == R_ANKLE_IDX)[0]
ankle_verts_l = np.where(np.argmax(smpl_model.lbs_weights.numpy(), axis=1) == L_ANKLE_IDX)[0]
ankle_verts = np.concatenate([ankle_verts_r, ankle_verts_l])
# ---------------- beta 设置 ---------------- #
betas = np.zeros((1, 10), dtype=np.float32)



# ---------------- 导出函数 ---------------- #
def export_smpl_obj(smpl_model, pose_aa, betas, output_path):
    body_output = smpl_model(
        betas=torch.tensor(betas, dtype=torch.float32),
        body_pose=torch.tensor(pose_aa[:, 3:], dtype=torch.float32),
        global_orient=torch.tensor(pose_aa[:, :3], dtype=torch.float32),
        transl=torch.zeros((1, 3), dtype=torch.float32),
        return_verts=True
    )
    vertices = body_output.vertices[0].detach().cpu().numpy()
    faces = smpl_model.faces
    toe_center_z = vertices[toe_verts][:, 2].mean()
    cut_plane_z = toe_center_z - 0.005  # 设置切割平面

    colors = np.ones((vertices.shape[0], 4)) * 0.7  # 灰色
    colors[:, 3] = 1.0  # Alpha=1
    colors[toe_verts] = [1.0, 0.0, 0.0, 1.0]  # 脚尖：红色
    colors[ankle_verts] = [1.0, 1.0, 0.0, 1.0]  # 脚尖：黄

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    mesh.export(output_path)


# ---------------- 执行导出 ---------------- #
export_smpl_obj(smpl_model, pose_aa_zero, betas, os.path.join(output_dir, "zeropose.obj"))
# export_smpl_obj(smpl_model, pose_aa_toe, betas, os.path.join(output_dir, "zeropose_toe.obj"))

body_output = smpl_model(
    betas=torch.tensor(betas, dtype=torch.float32),
    body_pose=torch.tensor(pose_aa_zero[:, 3:], dtype=torch.float32),
    global_orient=torch.tensor(pose_aa_zero[:, :3], dtype=torch.float32),
    transl=torch.zeros((1, 3), dtype=torch.float32),
    return_verts=True
)


print("导出完成：zeropose.obj 与 zeropose_toe.obj")

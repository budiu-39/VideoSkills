

''' 1 skeleton for all motions, no need to load skeletons for each motion'''
''' preprocess motions with SkeletonMotion, SkeletonState, only output Information that MotionLib needs '''
import os
import pickle

from utils.motionlib.motion_lib import MotionLib
import torch

# 修改这些路径和参数以适应你的环境
amass_root = "AMASS_split_small"  # 例如包含 .npz 文件的文件夹路径
output_pkl_path = "./preprocessed_test.pkl"

# Dummy parameters，你需要根据你的环境替换为实际值
dof_body_ids = torch.arange(23).tolist()
dof_offsets = torch.arange(0, 69, 3).tolist()  # 替换为实际的 DOF offsets
key_body_ids = torch.arange(24).tolist()  # 替换为实际的 DOF offsets
device = torch.device("cuda:0")      # 或者 "cpu"

# # 搜集所有 motion 文件路径
motion_files = []
for root, _, files in os.walk(amass_root):
    for f in files:
        if f.endswith(".npy"):  # 你也可以改为 .pkl 或其他格式
            motion_files.append(os.path.join(root, f))

# 创建 MotionLib 实例（注意：这里我们不调用标准 __init__ 中的 _load_motions）
lib = MotionLib(motion_files, dof_body_ids, dof_offsets, key_body_ids, device)

# 调用新的预处理函数
data_dict = lib.preprocess_amass_motion(motion_files)

# 保存为 .pkl 文件
with open(output_pkl_path, "wb") as f:
    pickle.dump(data_dict, f)

print(f"✅ Exported motion data to: {output_pkl_path}")
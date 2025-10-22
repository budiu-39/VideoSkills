import os
import shutil
import numpy as np

# ====== 基础路径配置 ======
base_dir = "/mnt/lustre/work/ponsmoll/pba936/VideoSkills/ablation_exp"
motion_root = "/mnt/lustre/work/ponsmoll/pba936/VideoSkills/dataset/smpl_motion/humanml3d"

# 新的输出根目录（将放在 smpl_motion 下）
subset_root = "/mnt/lustre/work/ponsmoll/pba936/VideoSkills/dataset/smpl_motion/subset"
os.makedirs(subset_root, exist_ok=True)

# ====== 定义 5 个子集 ======
splits = [
    ("active_subset_train", "picked_names.txt",  "train_active"),
    ("active_subset_test",  "test_names.txt",    "test"),
    ("active_subset_train", "control1_names.txt","control1"),
    ("active_subset_train", "control2_names.txt","control2"),
    ("active_subset_train", "control3_names.txt","control3"),
]

# ====== 主逻辑 ======
for split_dir, txt_name, out_subdir in splits:
    txt_path = os.path.join(base_dir, split_dir, txt_name)
    if not os.path.isfile(txt_path):
        print(f" 跳过：{txt_path} (未找到)")
        continue

    # 创建目标子文件夹
    out_dir = os.path.join(subset_root, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # 读取文件名列表
    names = np.loadtxt(txt_path, dtype=str, encoding="utf-8").tolist()
    print(f"\n [{out_subdir}] 从 {txt_path} 读取 {len(names)} 个文件名")

    missing, copied = [], 0

    for name in names:
        src = os.path.join(motion_root, name)
        dst = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append(name)

    print(f" 已复制 {copied} 个文件到 {out_dir}")
    if missing:
        miss_file = os.path.join(out_dir, "_missing.txt")
        np.savetxt(miss_file, np.asarray(missing, dtype=str), fmt="%s", encoding="utf-8")
        print(f" 缺失 {len(missing)} 个文件，详情见：{miss_file}")

print("\n 所有子集处理完成！")
print(f"输出目录: {subset_root}")

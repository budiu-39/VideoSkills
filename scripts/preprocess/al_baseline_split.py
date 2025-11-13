import os
import shutil
from tqdm import tqdm

# === 配置路径 ===
TRAIN_DIR = "dataset/smpl_motion/AMASS_train_fixed_height"
TEST_DIR = "dataset/smpl_motion/test_amass_al"
OUT_DIR = "dataset/smpl_motion/amass_al_baseline"

os.makedirs(OUT_DIR, exist_ok=True)

# === 收集 test 集的相对路径 ===
test_files = set()
for root, _, files in os.walk(TEST_DIR):
    for f in files:
        if f.endswith(".npy"):
            rel_path = os.path.relpath(os.path.join(root, f), TEST_DIR)
            test_files.add(rel_path)

print(f"[Info] Collected {len(test_files)} files in test_amass_al")

# === 遍历 train 集，找出补集 ===
copied = 0
for root, _, files in os.walk(TRAIN_DIR):
    for f in files:
        if not f.endswith(".npy"):
            continue
        rel_path = os.path.relpath(os.path.join(root, f), TRAIN_DIR)
        if rel_path not in test_files:  # 不在 test 集中
            src_path = os.path.join(root, f)
            dest_path = os.path.join(OUT_DIR, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)
            copied += 1

print(f"[Done] Copied {copied} files to {OUT_DIR}")

import os
import joblib
import numpy as np
from tqdm import tqdm


def convert_occlusion_db():
    # ================= 配置路径 =================
    # 你的 AMASS 数据集根目录
    AMASS_ROOT = "/mnt/lustre/work/ponsmoll/pba936/AMASS"

    # 原始的 occlusion 文件路径
    OLD_DB_PATH = "data/amass_copycat_occlusion_v3.pkl"

    # 输出的新文件路径
    NEW_DB_PATH = "data/amass_occlusion_v3_reindexed.pkl"
    # ===========================================

    print(f"[1/4] Loading original DB from {OLD_DB_PATH}...")
    if not os.path.exists(OLD_DB_PATH):
        print(f"Error: File {OLD_DB_PATH} not found.")
        return

    # 加载原始字典
    old_db = joblib.load(OLD_DB_PATH)
    print(f"      Original DB has {len(old_db)} entries.")

    print(f"[2/4] Scanning AMASS directory: {AMASS_ROOT}...")
    npz_files = []
    # 遍历目录查找所有 .npz 文件
    for root, dirs, files in os.walk(AMASS_ROOT):
        for file in files:
            if file.endswith(".npz"):
                # 获取绝对路径
                full_path = os.path.join(root, file)
                npz_files.append(full_path)

    print(f"      Found {len(npz_files)} .npz files.")

    print(f"[3/4] Re-indexing keys...")
    new_db = {}
    mapped_count = 0
    missed_count = 0

    # 用于 Debug: 打印前几个没找到的看看规律
    missed_examples = []

    for full_path in tqdm(npz_files):
        # 1. 获取相对路径并拆分
        # 例如: rel_path = "ACCAD/Female1General_c3d/A10 - lie to crouch_poses.npz"
        rel_path = os.path.relpath(full_path, AMASS_ROOT)

        # 拆分为列表: ['ACCAD', 'Female1General_c3d', 'A10 - lie to crouch_poses.npz']
        # 注意：Linux系统下路径分隔符为 '/'
        key_parts = rel_path.split('/')

        # 2. 构建【旧 Key】 (用于查询)
        # 逻辑: "0-" + "_".join(key_name).replace(".npz", "")
        # 例如: "0-ACCAD_Female1General_c3d_A10 - lie to crouch_poses"
        old_key = "0-" + "_".join(key_parts).replace(".npz", "")

        # 3. 构建【新 Key】 (你的目标格式)
        # 逻辑: "-".join(key_name).replace(".npz", "")
        # 例如: "ACCAD-Female1General_c3d-A10 - lie to crouch_poses"
        new_key = "-".join(key_parts).replace(".npz", "")

        # 4. 查找并迁移
        if old_key in old_db:
            new_db[new_key] = old_db[old_key]
            mapped_count += 1
        else:
            missed_count += 1
            if len(missed_examples) < 5:
                missed_examples.append(old_key)

    print(f"[4/4] Saving new DB to {NEW_DB_PATH}...")
    joblib.dump(new_db, NEW_DB_PATH)

    print("=" * 40)
    print(f"Summary:")
    print(f"  - Total files scanned: {len(npz_files)}")
    print(f"  - Successfully mapped: {mapped_count}")
    print(f"  - Missed (not in old DB): {missed_count}")

    if missed_count > 0:
        print(f"  [Info] Example missed keys (generated from local files but not found in pkl):")
        for m in missed_examples:
            print(f"    {m}")
        print("  (This is normal if your local AMASS has more files than the occlusion DB covers)")

    print(f"Done! You can now load '{NEW_DB_PATH}' and query using keys like 'Dataset-Subject-Motion'.")


if __name__ == "__main__":
    convert_occlusion_db()
import os
import glob
import joblib

# === 配置部分 ===
processed_dataset_dir = "dataset/smplx_motion/AMASS_train"  # 修改为你实际的数据集路径
occlusion_pkl_path = "data/amass_copycat_occlusion_v3.pkl"

# === 加载 occlusion 字典 ===
amass_occlusion = joblib.load(occlusion_pkl_path)
occlusion_keys = set(amass_occlusion.keys())

print(f"Loaded {len(occlusion_keys)} occlusion entries from {occlusion_pkl_path}")

# === 搜索所有已处理的 motion 文件 ===
processed_files = glob.glob(f"{processed_dataset_dir}/**/*.npy", recursive=True)
print(f"Total processed motion files: {len(processed_files)}")
found, missing = [], []

# 检查匹配关系
for file_path in processed_files:
    # 例如路径: dataset/smplx_motion/AMASS_train/KIT/442/PizzaDelivery02.npy
    rel_path = os.path.relpath(file_path, processed_dataset_dir)
    key_name = "0-" + rel_path.replace(os.sep, "_").replace(".npy", "")
    if key_name in occlusion_keys:
        found.append(key_name)
    else:
        missing.append(key_name)

print(f"/n Found {len(found)} occlusion samples in processed dataset")
print(f" Missing {len(occlusion_keys - set(found))} occlusion samples not found in processed dataset\n")

# === 输出结果到文件 ===
with open("occlusion_check_result.txt", "w") as f:
    f.write(f"Found ({len(found)}):\n")
    f.writelines([f"{k}\n" for k in sorted(found)])
    f.write("\nMissing in processed dataset:\n")
    missing_occlusion = sorted(list(occlusion_keys - set(found)))
    f.writelines([f"{k}\n" for k in missing_occlusion])

print("Results saved to occlusion_check_result.txt")

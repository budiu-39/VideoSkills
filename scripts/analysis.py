# import joblib
# import os
# import re
# import glob
# log_dir = '/home/miku/Documents/VideoSkills/logs/smpl_ppo/kungfu_local'
# eval_dir = os.path.join(log_dir, "eval_outputs")
#
# # 查找所有以 failed_keys_iter 开头的文件
# all_failed_files = [f for f in os.listdir(eval_dir) if f.startswith("failed_keys_iter") and f.endswith(".pkl")]
#
# # 使用正则表达式提取 iteration 数字
# def extract_iter(f):
#     match = re.search(r"iter(\d+)", f)
#     return int(match.group(1)) if match else -1
#
# # 找到最大的 iteration 文件
# latest_failed_file = max(all_failed_files, key=extract_iter)
#
# # 加载最新的 failed_key 文件
# failed_key = joblib.load(os.path.join(eval_dir, latest_failed_file))
#
# # 加载 motion sampling 状态
# motion_sampling_status = joblib.load(os.path.join(log_dir, "motion_sampling_state.pkl"))
#
# amass_root = "AMASS_split"
#
# negative_samples = []
# for key, value in motion_sampling_status.items():
#     if value['termination_count'] > 13:
#         negative_samples.append(key)
#         # parts = key.split("-")
#         # if len(parts) >= 2:
#         #     dataset = parts[0]
#         #     subset = parts[1]
#         #     filename ="-".join(parts[2:])   # 保留中间所有 - 的名字
#         #     rel_path = os.path.join(amass_root, dataset, subset , filename + ".npy")
#         #     npy_paths.append(rel_path)
#             # motion = SkeletonMotion.from_file(rel_path)
#
# output_path =  os.path.join(log_dir, "negative_samples.txt")
# with open(output_path, "w", encoding="utf-8") as f:
#     for item in negative_samples:
#         f.write(str(item) + "\n")
#
# # fix_height_dict_load = joblib.load(os.path.join("AMASS_fixed_height", "fixed_height_keys.pkl"))
#
# dataset_path = "/home/miku/Documents/VideoSkills/dataset/smpl_motion/MotionX++/kungfu"
# dataset_dir = "/home/miku/Documents/VideoSkills/dataset/smpl_motion/MotionX++"
# motion_files = glob.glob(os.path.join(dataset_path, "*.npy"))
#
# output_dir = os.path.join(dataset_dir, "kungfu_clean")
# motion_files_clean = []
# for motion_file in motion_files:
#     base_name = os.path.basename(motion_file)
#     key_name = os.path.splitext(base_name)[0]  # 去掉扩展名
#     if key_name in negative_samples:
#         continue
#     motion_files_clean.append(motion_file)
#
# os.makedirs(output_dir, exist_ok=True)
# for motion_file in motion_files_clean:
#     base_name = os.path.basename(motion_file)
#     dest_file = os.path.join(output_dir, base_name)
#     os.system(f"cp '{motion_file}' '{dest_file}'")
#
# print("Done")


# import os
# import glob
# import math
# import random
# import shutil
#
# # ===== 路径设定（与你上文一致/相对）=====
# dataset_path = "/home/miku/Documents/VideoSkills/dataset/smpl_motion/MotionX++/kungfu"  # 全集（含噪）
# dataset_dir  = "/home/miku/Documents/VideoSkills/dataset/smpl_motion/MotionX++"
# clean_dir    = os.path.join(dataset_dir, "kungfu_clean")       # 你上面生成的干净集
# test_dir     = os.path.join(dataset_dir, "kungfu_clean_test")  # 10% 的测试集（从 clean 中抽）
# noisy_train_dir = os.path.join(dataset_dir, "kungfu_noisy_train")  # 全集 - test
#
# os.makedirs(test_dir, exist_ok=True)
# os.makedirs(noisy_train_dir, exist_ok=True)
#
# # ===== 1) 从 clean 中抽取 10% 作为 test，并从 clean 中移除 =====
# clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.npy")))
# n_total = len(clean_files)
# assert n_total > 0, f"[Error] clean 集为空：{clean_dir}"
#
# # 可复现实验
# random.seed(42)
#
# n_test = max(1, math.floor(n_total * 0.10))  # 抽 10%，至少 1 个
# test_samples = set(random.sample(clean_files, n_test))
#
# print(f"[Split] clean 总数: {n_total}, 抽取 test 数: {n_test}")
#
# # 移动到 test_dir（从 clean 中“删掉”）
# for src in test_samples:
#     dst = os.path.join(test_dir, os.path.basename(src))
#     # 若 test_dir 已存在同名文件，可选择覆盖或跳过；这里选择覆盖以保持一致性
#     if os.path.exists(dst):
#         os.remove(dst)
#     shutil.move(src, dst)
# print(f"[Move] 已将 {n_test} 个样本从 {clean_dir} 移动到 {test_dir}")
#
# # ===== 2) 从全集 dataset_path 中减去 test，得到 noisy_train =====
# #   定义：noisy_train = 全集（含噪） - test（干净测试集）
# all_files_full = sorted(glob.glob(os.path.join(dataset_path, "*.npy")))
# test_basenames = {os.path.basename(p) for p in glob.glob(os.path.join(test_dir, "*.npy"))}
#
# # 需要拷贝到 noisy_train 的文件（全集中不在 test 的）
# to_copy = [p for p in all_files_full if os.path.basename(p) not in test_basenames]
#
# print(f"[Build noisy_train] 全集: {len(all_files_full)}, test: {len(test_basenames)}, "
#       f"noisy_train 目标数: {len(to_copy)}")
#
# for src in to_copy:
#     dst = os.path.join(noisy_train_dir, os.path.basename(src))
#     # 覆盖以保持同步
#     shutil.copy2(src, dst)
#
# print("[Done] 生成完成：")
# print(f" - Test set:        {test_dir}")
# print(f" - Noisy train set: {noisy_train_dir}")

# import os
# import glob
#
# base_dir = "/home/miku/Documents/VideoSkills/dataset/smpl_motion/MotionX++"
# folders = ["kungfu_clean", "kungfu_clean_test", "kungfu_noisy_train"]
#
# for folder in folders:
#     folder_path = os.path.join(base_dir, folder)
#     output_txt = os.path.join(base_dir, f"{folder}_list.txt")
#
#     # 获取所有 .npy 文件
#     files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
#     filenames = [os.path.basename(f) for f in files]
#
#     # 写入 txt
#     with open(output_txt, "w", encoding="utf-8") as f:
#         for name in filenames:
#             f.write(name + "\n")
#
#     print(f"[Saved] {len(filenames)} files -> {output_txt}")

import os
import shutil

base_dir = "/home/miku/Documents/VideoSkills/dataset/smpl_motion/MotionMillion"

# 定义输入输出
src_dir = os.path.join(base_dir, "kungfu")  # 原始全集
lists = {
    "kungfu_clean_list.txt": "kungfu_clean_train",
    "kungfu_noisy_train_list.txt": "kungfu_noisy_train",
    "kungfu_clean_test_list.txt": "kungfu_clean_test"
}

# 创建输出文件夹并执行复制
for list_file, out_folder in lists.items():
    out_dir = os.path.join(base_dir, out_folder)
    os.makedirs(out_dir, exist_ok=True)

    list_path = os.path.join(base_dir, list_file)
    if not os.path.exists(list_path):
        print(f"[Skip] 找不到 {list_path}")
        continue

    # 读取文件名列表
    with open(list_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    print(f"[Copying] {len(names)} files from {src_dir} → {out_dir}")

    for name in names:
        src_file = os.path.join(src_dir, name)
        dst_file = os.path.join(out_dir, name)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
        else:
            print(f"[Warning] {src_file} 不存在，跳过")

    print(f"[Done] {out_folder} ✅")

print("\n✅ 所有文件已复制完成。")
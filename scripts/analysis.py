import joblib
import os
import re
# from retarget.poselib.skeleton.skeleton3d import SkeletonMotion
#
# log_dir = '/home/miku/Documents/VideoSkills/logs/smpl_ppo'
# run_name = 'universal_00001_torque_100_imi'
# eval_dir = os.path.join(log_dir, run_name, "eval_outputs")
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
# motion_sampling_status = joblib.load(os.path.join(log_dir, run_name, "motion_sampling_state.pkl"))
#
# amass_root = "AMASS_split"
#
# npy_paths = []
# for key, value in motion_sampling_status.items():
#     if value['termination_count'] > 3:
#         parts = key.split("-")
#         if len(parts) >= 2:
#             dataset = parts[0]
#             subset = parts[1]
#             filename ="-".join(parts[2:])   # 保留中间所有 - 的名字
#             rel_path = os.path.join(amass_root, dataset, subset , filename + ".npy")
#             npy_paths.append(rel_path)
#             # motion = SkeletonMotion.from_file(rel_path)
#
# joblib.dump(npy_paths, "mostly_failed_motion.pkl")

fix_height_dict_load = joblib.load(os.path.join("AMASS_fixed_height", "fixed_height_keys.pkl"))

print("Done")

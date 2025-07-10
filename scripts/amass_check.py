import os
from collections import defaultdict
from retarget.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import glob
import os.path as osp
import joblib

def find_duplicate_filenames(folder_path):
    # 用来存储文件名映射到路径的列表
    file_map = defaultdict(list)

    # 遍历文件夹
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_map[file].append(file_path)

    # 筛选出有重复名字的文件
    duplicates = {name: paths for name, paths in file_map.items() if len(paths) > 1}

    print(f"Found {len(duplicates)} duplicate file names in '{folder_path}':")

    # 输出重复文件名的路径和大小

def duplicate_check(motion_files):
    num_motion_files = len(motion_files)
    for f in range(num_motion_files):
        curr_file = motion_files[f]
        print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
        curr_motion = SkeletonMotion.from_file(curr_file)

    file_map = defaultdict(list)
    num = 0

    # 检查是否有同名文件且大小差距在允许范围内
    for filename, file_list in motion_files.items():
        if len(file_list) <= 1:
            continue  # 只有一个文件，跳过

        # 比较每一对文件
        for i in range(len(file_list)):
            path1, size1 = file_list[i]
            for j in range(i + 1, len(file_list)):
                path2, size2 = file_list[j]
                # 计算相对误差
                diff = abs(size1 - size2)
                max_size = max(size1, size2)
                if diff == 0:
                    print(f"\nDuplicate name with similar size: {filename}")
                    print(f"  Path 1: {path1}, Size: {size1} bytes")
                    print(f"  Path 2: {path2}, Size: {size2} bytes")
                    num += 1
    print(num)

def get_all_motion_files(amass_processed_dir: str, ext=".npy") -> list:
    """
    Recursively collect all motion file paths under AMASS_processed.

    Args:
        amass_processed_dir (str): e.g. "AMASS_processed"
        ext (str): extension of the motion files, default ".pkl"

    Returns:
        List[str]: list of full file paths
    """
    motion_paths = glob.glob(os.path.join(amass_processed_dir, f"**/*{ext}"), recursive=True)
    motion_paths.sort()  # optional: ensure deterministic order
    return motion_paths

def get_motion_frame_count_by_loader(path: str):
    try:
        motion = SkeletonMotion.from_file(path)
        return motion.local_rotation.shape[0]  # 或 motion.num_frames 如果你定义了这个属性
    except Exception as e:
        print(f"[Warning] Failed to load motion file: {path}: {e}")
        return None

def deduplicate_motion_files_with_loader(motion_files: list):
    seen = dict()  # key: (filename, size, frame_count)
    duplicates = []

    for path in motion_files:
        filename = os.path.basename(path)
        size = os.path.getsize(path)
        frame_count = get_motion_frame_count_by_loader(path)
        if frame_count is None:
            continue

        key = (filename, size, frame_count)
        if key in seen:
            duplicates.append(path)
        else:
            seen[key] = path

    print(f"\n Total duplicate files skipped: {len(duplicates)}")
    print(f" Total unique files kept: {len(seen)}")
    for path in duplicates:
        print(f" Skipped: {path}")

def collect_keys(base_dir, ext, allowed_subsets):
    all_files = glob.glob(osp.join(base_dir, '**', f'*{ext}'), recursive=True)
    key_map = {}
    for path in all_files:
        rel_path = osp.relpath(path, base_dir)
        key = "0-" + rel_path.replace(ext, '').replace(os.sep, '_')
        subset = rel_path.split(os.sep)[0]
        if subset in allowed_subsets:
            key_map[key] = subset
    return key_map

def compare_keys(orig_keys, proc_keys, occluded_keys, name):
    missing_total = sorted(set(orig_keys) - set(proc_keys))
    missing_excluding_occlusion = sorted(set(missing_total) - occluded_keys)
    missing_due_to_occlusion = sorted(set(missing_total) & occluded_keys)

    print(f"\n=== {name} - Total Missing: {len(missing_total)} ===")
    print(f"→ Missing but NOT in occlusion list: {len(missing_excluding_occlusion)}")
    print(f"→ Missing due to occlusion: {len(missing_due_to_occlusion)}")

    print("\n--- Missing not due to occlusion ---")
    for key in missing_excluding_occlusion:
        print(f"{key} (subset: {orig_keys[key]})")

    # print("\n--- Missing due to occlusion ---")
    # for key in missing_due_to_occlusion:
    #     issue = occlusion_dict[key]["issue"]
    #     print(f"{key} (subset: {orig_keys[key]}, issue: {issue})")


if __name__ == "__main__":
    # 设置路径
    consider_splits = ["train", "valid"]

    # AMASS split 映射
    amass_splits = {
        'valid': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'KIT', 'EKUT', 'TCD_handMocap', "BMLhandball", "DanceDB",
                  "ACCAD", "BMLmovi", "BioMotionLab_NTroje", "Eyes_Japan_Dataset", "DFaust_67"]
    }

    allowed_subsets = set(sum([amass_splits[split] for split in consider_splits], []))

    original_dir = "../AMASS"
    processed1_dir = "AMASS_fixed_height"
    processed2_dir = "AMASS_processed"
    occlusion_pkl = "output/SMPL_Robot_motion/amass_copycat_occlusion_v3.pkl"

    # 扫描所有文件
    orig_keys = collect_keys(original_dir, ".npz", allowed_subsets)
    proc1_keys = collect_keys(processed1_dir, ".npy", allowed_subsets)
    proc2_keys = collect_keys(processed2_dir, ".npy", allowed_subsets)

    print(f"\nTotal motions in original set (filtered by split): {len(orig_keys)}")
    print(f"Total motions in Processed Version 1: {len(proc1_keys)}")
    print(f"Total motions in Processed Version 2: {len(proc2_keys)}")

    occlusion_dict = joblib.load(occlusion_pkl)
    occluded_keys = set([k for k in occlusion_dict if k in orig_keys])

    # 对比
    compare_keys(orig_keys, proc1_keys, occluded_keys, "Fixed_height Version")
    compare_keys(orig_keys, proc2_keys, occluded_keys, "Processed Version")
    compare_keys(proc2_keys, proc1_keys, occluded_keys,"Comparison between Fixed_height & Processed")
# # 用法示例
# if __name__ == "__main__":
#     folder_to_check = "AMASS_processed"  # ← 改为你要检查的文件夹路径
#     get_all_motion_files = get_all_motion_files(folder_to_check, ext=".npy")
#     deduplicate_motion_files_with_loader(get_all_motion_files)
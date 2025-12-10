import os
import shutil
import math

def split_folder(src_folder, split_size=150):
    """
    将 src_folder 中的文件按 split_size 分组，
    分到子文件夹 src_folder_1, src_folder_2, ...
    """
    # 获取绝对路径
    src_folder = os.path.abspath(src_folder)

    # 获取所有文件（不包含子文件夹）
    files = [f for f in os.listdir(src_folder)
             if os.path.isfile(os.path.join(src_folder, f))]

    files.sort()  # 可选：按名字排序，保证顺序一致
    total = len(files)
    num_splits = math.ceil(total / split_size)

    print(f"共有 {total} 个文件，将分成 {num_splits} 个文件夹，每个最多 {split_size} 个。")

    for i in range(num_splits):
        start = i * split_size
        end = min((i + 1) * split_size, total)
        split_files = files[start:end]

        # 新文件夹名，如 x_1, x_2, ...
        new_folder = f"{src_folder}_{i+1}"
        os.makedirs(new_folder, exist_ok=True)

        for f in split_files:
            src_path = os.path.join(src_folder, f)
            dst_path = os.path.join(new_folder, f)
            shutil.move(src_path, dst_path)

        print(f"✅ 已移动 {len(split_files)} 个文件到 {new_folder}")

    print("✅ 所有文件已分组完成。")

# ===== 使用示例 =====
# 把 "x" 改成你的文件夹路径
split_folder("/home/miku/Documents/VideoSkills/dataset/smpl_motion/Kungfu", split_size=150)

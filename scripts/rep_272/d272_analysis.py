import os
import numpy as np

# 两个目录
new_dir = "/home/miku/Documents/00008"
old_dir = "/home/miku/Documents/00008_old"

# 列出各自目录下的 npy 文件（只看一层目录，如果有子目录可以换成 os.walk 或 glob）
new_files = [f for f in os.listdir(new_dir) if f.endswith(".npy")]
old_files = [f for f in os.listdir(old_dir) if f.endswith(".npy")]

# 找出同名的 npy 文件
common_names = sorted(set(new_files) & set(old_files))

print(f"共同的 npy 文件数: {len(common_names)}")

# 取前 10 个共同文件名
sample_names = common_names[:10]
print("取出的前 10 个文件名:")
for name in sample_names:
    print("  ", name)

print("\n===== 逐个比较同名 npy 文件的数据 =====")
for name in sample_names:
    path_new = os.path.join(new_dir, name)
    path_old = os.path.join(old_dir, name)

    arr_new = np.load(path_new, allow_pickle=True)
    arr_old = np.load(path_old, allow_pickle=True)

    # 先打印 shape
    print(f"File = {name}")
    print(f"  new shape: {arr_new.shape}, dtype: {arr_new.dtype}")
    print(f"  old shape: {arr_old.shape}, dtype: {arr_old.dtype}")

    # 如果形状不同，后面就不用比了
    if arr_new.shape != arr_old.shape:
        print("  → ✘ 形状不同，数据肯定不一样\n")
        continue

    # 计算差异（L2 范数）
    diff = np.linalg.norm(arr_new.astype(np.float64) - arr_old.astype(np.float64))
    print(f"  差异 L2 norm = {diff:.6f}")

    # 判断是否近似相等
    if np.allclose(arr_new, arr_old, atol=1e-6):
        print("  → ✔ 完全一致 (within tolerance)")
    else:
        print("  → ✘ 数值不同")
    print()

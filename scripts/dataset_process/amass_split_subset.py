import os, shutil, glob

def _collect_amass_keys(amass_root: str):
    """
    从 AMASS 根目录递归收集所有 .npy 文件。
    返回:
      key2path: dict[key] = full_path
      path2rel: dict[key] = 相对路径 (相对于 amass_root)
    """
    all_npys = glob.glob(f"{amass_root}/**/*.npy", recursive=True)
    split_len = len(amass_root.rstrip("/").split("/"))

    key2path = {}
    path2rel = {}
    for p in all_npys:
        rel_parts = p.split("/")[split_len:]
        key = "-".join(rel_parts).replace(".npy", "")
        key2path[key] = p
        path2rel[key] = os.path.join(*rel_parts)
    return key2path, path2rel

# amass_root = "dataset/smpl_motion/AMASS_train_fixed_height"
# key2path, path2rel = _collect_amass_keys(amass_root)
# key = key2path.keys()
# with open("embed/AMASS/amass_train_keys.txt", "w", encoding="utf-8") as f:
#     for k in key:
#         f.write(str(k) + "\n")


# ======== 配置路径 ========
amass_root = "dataset/smpl_motion/AMASS_train_fixed_height"
subset_root = "/mnt/lustre/work/ponsmoll/pba936/VideoSkills/dataset/splits/causal_longer"
out_dir = "dataset/smpl_motion/"
# 1. 建立 key → path, rel_path 映射
key2path, path2rel = _collect_amass_keys(amass_root)

# 2. 遍历各个 *_names.txt
txt_files = [f for f in os.listdir(subset_root) if f.endswith("_names.txt")]
txt_files = ["/home/miku/Documents/VideoSkills/logs/smpl_ppo/traj_augmented_output/success_keys.txt"]
for txt in txt_files:
    # set_name = txt.replace("_names.txt", "")
    dest_root = os.path.join(out_dir, "traj_aug_success")
    # print(f"\n[{set_name}] 输出目录: {dest_root}")
    os.makedirs(dest_root, exist_ok=True)

    # 读取该列表
    # keys = [ln.strip() for ln in open(os.path.join(subset_root, txt), encoding="utf-8") if ln.strip()]
    keys = [ln.strip() for ln in open(txt, encoding="utf-8") if ln.strip()]

    missing = 0
    for k in keys:
        if k not in key2path:
            missing += 1
            continue
        src_path = key2path[k]
        rel_path = path2rel[k]              # 例如 CMU/S01/seq01.npy
        dest_path = os.path.join(dest_root, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # ★ 保留层级
        shutil.copy2(src_path, dest_path)                       # 复制文件

    print(f"✅ 已复制 {len(keys)-missing} / {len(keys)} 个 motion 文件 ({missing} 缺失)")

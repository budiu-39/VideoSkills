import os
import joblib
import numpy as np


def compare_two_directories():
    # ================= 1. 配置路径 (请修改为你实际的路径) =================

    # [Input 1] 原始 AMASS 数据的根目录 (层级结构: Dataset/Subject/Motion.npz)
    RAW_AMASS_ROOT = "/mnt/lustre/work/ponsmoll/pba936/AMASS"

    # [Input 2] 处理后的 AMASS 272 数据的根目录 (扁平结构: Dataset-Subject-Motion.npy)
    # 请确保这是你生成的那个包含 .npy 文件的文件夹路径
    PROCESSED_AMASS_ROOT = "/mnt/lustre/work/ponsmoll/pba936/VideoSkills/dataset/272_rep/AMASS_272"

    # 两个数据库
    OLD_DB_PATH = "data/amass_copycat_occlusion_v3.pkl"
    NEW_DB_PATH = "data/amass_occlusion_v3_reindexed.pkl"

    # Train Split 定义
    train_subsets = set(['CMU', 'MPI_Limits', 'TotalCapture', 'KIT', 'EKUT',
                         'TCD_handMocap', "BMLhandball", "DanceDB", "ACCAD",
                         "BMLmovi", "BioMotionLab_NTroje", "Eyes_Japan_Dataset",
                         "DFaust_67"])
    # ====================================================================

    print("Loading databases...")
    old_db = joblib.load(OLD_DB_PATH)
    new_db = joblib.load(NEW_DB_PATH)

    # ---------------------------------------------------------
    # 步骤 1: 扫描 Raw AMASS (旧逻辑)
    # ---------------------------------------------------------
    print(f"[Step 1] Scanning Raw AMASS (Old Logic) at: {RAW_AMASS_ROOT}")

    # 字典: unique_id -> old_key
    # unique_id 统一格式为: "Dataset-Subject-Motion" (不带后缀)
    old_valid_files = {}

    for root, dirs, files in os.walk(RAW_AMASS_ROOT):
        for file in files:
            if not file.endswith(".npz"):
                continue

            # 路径解析: .../AMASS/CMU/10/10_01_poses.npz
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, RAW_AMASS_ROOT)
            parts = rel_path.split("/")  # ['CMU', '10', '10_01_poses.npz']

            dataset_name = parts[0]
            if dataset_name not in train_subsets:
                continue

            # --- 构建 Old Key ---
            # 旧逻辑通常是: "0-" + "_".join(parts)
            old_key = "0-" + "_".join(parts).replace(".npz", "")

            # --- 模拟旧筛选逻辑 ---
            keep = True

            npz_data = dict(np.load(open(full_path, "rb"), allow_pickle=True))

            if 'trans' not in npz_data.keys():
                continue
            framerate = npz_data['mocap_framerate']

            skip = int(framerate / 30)

            root_trans = npz_data['trans'][::skip, :]

            num_frames = root_trans.shape[0]

            if num_frames < 10:
                keep = False
            if num_frames > 5000:
                keep = False
            # 1. 硬编码特例
            if "0-KIT_442_PizzaDelivery02_poses" == old_key:
                keep = True
            # 2. 查表
            elif old_key in old_db:
                issue_data = old_db[old_key]
                issue = issue_data["issue"]
                if (issue == "sitting" or issue == "airborne") and "idxes" in issue_data:
                    if issue_data["idxes"][0] < 10:
                        keep = False
                else:
                    keep = False  # irrecoverable
            else:
                # 3. 没找到 Key -> 默认保留
                keep = True

            if keep:
                # 生成一个唯一ID用于后续对比
                unique_id = "-".join(parts).replace(".npz", "")
                old_valid_files[unique_id] = old_key

    print(f" -> Found {len(old_valid_files)} valid files in Raw AMASS (Old Logic).")

    # ---------------------------------------------------------
    # 步骤 2: 扫描 Processed AMASS 272 (新逻辑结果)
    # ---------------------------------------------------------
    print(f"[Step 2] Scanning Processed AMASS 272 at: {PROCESSED_AMASS_ROOT}")

    # 集合: 存放所有存在的 unique_id
    new_valid_files = set()

    # 注意：这里我们遍历的是磁盘上实际存在的 272 文件
    # 如果文件在磁盘上不存在，说明它已经被之前的步骤过滤掉了
    # 另外我们也会再次验证一遍 New DB 的逻辑，以防磁盘上有脏数据

    processed_files = []
    if os.path.exists(PROCESSED_AMASS_ROOT):
        # 假设是扁平结构，或者带子文件夹结构，用 walk 通吃
        for root, dirs, files in os.walk(PROCESSED_AMASS_ROOT):
            for file in files:
                if file.endswith(".npy"):
                    processed_files.append(file)
    else:
        print(f"Error: Processed root {PROCESSED_AMASS_ROOT} does not exist!")
        return

    for filename in processed_files:
        # filename 格式: "CMU-10-10_01_poses.npy"
        # 提取 Dataset Name
        try:
            dataset_name = filename.split("-")[0]
        except:
            continue

        if dataset_name not in train_subsets:
            continue

        unique_id = filename.replace(".npy", "")
        new_key = unique_id  # 272 的 key 就是 unique_id

        # --- 再次验证新逻辑 (可选，或者直接信任磁盘文件) ---
        # 这里我们加上验证，看看"新逻辑"是否真的认可它
        keep = True
        if new_key in new_db:
            issue_data = new_db[new_key]
            issue = issue_data["issue"]
            if (issue == "sitting" or issue == "airborne") and "idxes" in issue_data:
                if issue_data["idxes"][0] < 10:
                    keep = False
            else:
                keep = False

        if keep:
            new_valid_files.add(unique_id)

    print(f" -> Found {len(new_valid_files)} valid files in Processed AMASS 272.")

    # ---------------------------------------------------------
    # 步骤 3: 对比差异
    # ---------------------------------------------------------
    old_ids = set(old_valid_files.keys())
    missing_ids = old_ids - new_valid_files

    print("\n" + "=" * 80)
    print(f"Old Logic Count : {len(old_ids)}")
    print(f"New Logic Count : {len(new_valid_files)}")
    print(f"Missing Files   : {len(missing_ids)}")
    print("=" * 80)

    print(f"\n{'[Old Key (amass_copycat_occlusion_v3)]':<50} | {'[Reason (New DB)]'}")
    print("-" * 100)

    for uid in sorted(list(missing_ids)):
        old_key_str = old_valid_files[uid]

        # 查新表看看为什么没了
        reason = "File missing from disk"
        if uid in new_db:
            info = new_db[uid]
            reason = f"Issue: {info['issue']}"
            if 'idxes' in info:
                reason += f", Bound: {info['idxes'][0]}"

        print(f"{old_key_str:<50} | {reason}")

    print("-" * 100)


if __name__ == "__main__":
    compare_two_directories()
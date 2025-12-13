import os
import glob
import joblib

def get_amass_splits(data_root):
    """
    根据文件名开头划分 AMASS 数据集。
    """

    # 1. 定义划分字典
    amass_splits = {
        'valid': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'KIT', 'EKUT', 'TCD_handMocap',
                  "BMLhandball", "DanceDB", "ACCAD", "BMLmovi", "BioMotionLab_NTroje",
                  "Eyes_Japan_Dataset", "DFaust_67"]
    }

    # 为了快速查找，将列表转换为集合，或者建立反向映射
    # name_to_split: {'ACCAD': 'train', 'HumanEva': 'valid', ...}
    name_to_split = {}
    for split_name, dataset_names in amass_splits.items():
        for ds_name in dataset_names:
            name_to_split[ds_name] = split_name

    # 2. 获取所有文件
    all_files = sorted(
        glob.glob(os.path.join(data_root, "**", "*.npy"), recursive=True)
    )

    train_files = []
    val_files = []
    test_files = []
    skipped_files = []

    print(f"[Data Split] Found {len(all_files)} files in {data_root}")

    amass_occlusion = joblib.load("data/amass_occlusion_v3_reindexed.pkl")

    # 3. 遍历并分类
    for fpath in all_files:
        filename = os.path.basename(fpath)  # 例如: "ACCAD-Female1General_c3d-A10 - lie to crouch_poses.npy"

        # 提取第一个元素作为数据集名称
        # 注意：这里假设文件名严格符合 "DatasetName-..." 的格式
        key_name_new = filename.replace(".npy", "").replace(".npz", "")
        # 或者如果 filename 是路径，根据你之前的 split 逻辑生成 key

        if key_name_new in amass_occlusion:
            issue_data = amass_occlusion[key_name_new]
            issue = issue_data["issue"]

            # 你的筛选逻辑保持不变
            if (issue == "sitting" or issue == "airborne") and "idxes" in issue_data:
                bound = issue_data["idxes"][0]
                if bound < 10:
                    # print("bound too small", key_name_new, bound)
                    continue
            else:
                # print("issue irrecoverable", key_name_new, issue)
                continue

        try:
            dataset_name = filename.split('-')[0]
        except IndexError:
            skipped_files.append(filename)
            continue

        # 查找该数据集属于哪个 split
        split_type = name_to_split.get(dataset_name)

        if split_type == 'train':
            train_files.append(fpath)
        elif split_type == 'valid':
            val_files.append(fpath)
        elif split_type == 'test':
            test_files.append(fpath)
        else:
            # 如果数据集名字不在字典里（可能是拼写错误或新数据集）
            skipped_files.append(fpath)

    # 4. 打印统计信息
    print(f"  - Train: {len(train_files)}")
    print(f"  - Valid: {len(val_files)}")
    print(f"  - Test : {len(test_files)}")
    if skipped_files:
        print(f"  - [Warning] Skipped {len(skipped_files)} files (unknown dataset name):")
        # 打印前5个看看是什么情况
        for s in skipped_files[:5]:
            print(f"      {os.path.basename(s)}")

    return train_files, val_files, test_files

def get_amass_splits_hierarchical(data_root):
    """
    针对层级结构的 AMASS 数据进行划分。
    假设结构: data_root / DatasetName / Subject / Sequence.npy
    """

    # 1. 定义划分字典 (保持不变)
    amass_splits = {
        'valid': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'KIT', 'EKUT', 'TCD_handMocap',
                  "BMLhandball", "DanceDB", "ACCAD", "BMLmovi", "BioMotionLab_NTroje",
                  "Eyes_Japan_Dataset", "DFaust_67"]
    }

    name_to_split = {}
    for split_name, dataset_names in amass_splits.items():
        for ds_name in dataset_names:
            name_to_split[ds_name] = split_name

    # 2. 获取所有 .npy 文件 (递归)
    # 也可以支持 .npz，视具体情况而定
    all_files = sorted(
        glob.glob(os.path.join(data_root, "**", "*.npy"), recursive=True)
    )

    # 尝试加载遮挡数据，如果文件不存在则跳过过滤
    try:
        amass_occlusion = joblib.load("data/amass_occlusion_v3_reindexed.pkl")
        print("[Info] Loaded occlusion filter.")
    except:
        amass_occlusion = {}
        print("[Warning] Occlusion file not found, skipping filter.")

    train_files = []
    val_files = []
    test_files = []
    skipped_files = []

    print(f"[Data Split] Found {len(all_files)} files in {data_root}")

    for fpath in all_files:
        # 获取相对于 data_root 的路径
        # 例如: "ACCAD/Female1General_c3d/A10 - lie to crouch_poses.npy"
        rel_path = os.path.relpath(fpath, data_root)

        # 分解路径组件
        path_parts = rel_path.split(os.sep)

        # 即使文件在根目录下，parts[0] 也是文件名，这里我们要防止这种情况
        if len(path_parts) < 2:
            skipped_files.append(fpath)
            continue

        dataset_name = path_parts[0]  # 第一级目录即为 DatasetName

        # --- 构建 Occlusion Key ---
        # 原有的扁平文件名通常是: Dataset-Subject-Sequence.npy
        # 我们需要从层级路径重构这个 Key 来查询字典
        # 例如: ACCAD (part0) - Female1 (part1) - Sequence (part2)
        # 注意: sequence 文件名可能包含 .npy，需要去掉

        if len(path_parts) >= 2:
            # 这种重构方式适用于标准的 AMASS 处理流程 (如 HumanML3D)
            # 假设路径是 Dataset/Subject/Motion.npy
            subject_name = path_parts[1]
            motion_name = path_parts[-1].replace('.npy', '').replace('.npz', '')

            # 尝试构建 key，通常是用 '-' 连接
            # 注意：有些数据集 Subject 名字里自带空格或特殊字符，需要视之前的 key 生成逻辑而定
            # 这里采用最通用的 Dataset-Subject-Motion 格式
            key_name_new = f"{dataset_name}-{subject_name}-{motion_name}"

            # --- 遮挡过滤逻辑 ---
            if key_name_new in amass_occlusion:
                issue_data = amass_occlusion[key_name_new]
                issue = issue_data["issue"]
                if (issue == "sitting" or issue == "airborne") and "idxes" in issue_data:
                    bound = issue_data["idxes"][0]
                    if bound < 10:
                        continue
                else:
                    continue
        # ------------------------

        # 查找该数据集属于哪个 split
        split_type = name_to_split.get(dataset_name)

        if split_type == 'train':
            train_files.append(fpath)
        elif split_type == 'valid':
            val_files.append(fpath)
        elif split_type == 'test':
            test_files.append(fpath)
        else:
            skipped_files.append(fpath)

    print(f"  - Train: {len(train_files)}")
    print(f"  - Valid: {len(val_files)}")
    print(f"  - Test : {len(test_files)}")

    return train_files, val_files, test_files
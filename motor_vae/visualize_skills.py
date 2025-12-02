import os
import glob
import torch
import umap
from torch.utils.data import DataLoader, Subset

import numpy as np
import matplotlib.pyplot as plt

# 引入修改后的 Dataset
from motion_window_dataset import MotionWindowDataset, TargetedMotionDataset
# 引入模型和 Scaler
from motor_vae import PolicySkillTreeVAE, StandardScaler


def load_failed_info(txt_path, wild_root):
    """
    读取失败列表，并找到对应的文件路径。
    返回:
        failed_targets: [(full_path, frame_idx), ...]
        failed_paths_set: set(full_path) 用于排除成功样本
    """
    if not os.path.exists(txt_path):
        print(f"[Warning] Failed keys file not found: {txt_path}")
        return [], set()

    # 1. 建立 Wild 目录下所有文件的索引: {clean_key: full_path}
    print("Indexing wild files...")
    wild_files = sorted(glob.glob(os.path.join(wild_root, "*.npy")))
    key_to_path = {}

    for fpath in wild_files:
        fname = os.path.basename(fpath)
        key = os.path.splitext(fname)[0]

        key_to_path[key] = fpath

    # 2. 解析 txt
    failed_targets = []
    failed_file_paths = set()

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            key, frame_str = line.split(",")
            frame = int(frame_str)

            if key in key_to_path:
                fpath = key_to_path[key]
                failed_targets.append((fpath, frame))
                failed_file_paths.add(fpath)

    return failed_targets, failed_file_paths


def load_success_key(txt_path, wild_root):
    """
    读取失败列表，并找到对应的文件路径。
    返回:
        failed_targets: [(full_path, frame_idx), ...]
        failed_paths_set: set(full_path) 用于排除成功样本
    """
    if not os.path.exists(txt_path):
        print(f"[Warning] Failed keys file not found: {txt_path}")
        return [], set()

    # 1. 建立 Wild 目录下所有文件的索引: {clean_key: full_path}
    print("Indexing wild files...")
    wild_files = sorted(glob.glob(os.path.join(wild_root, "*.npy")))
    key_to_path = {}
    key_list = []

    for fpath in wild_files:
        fname = os.path.basename(fpath)
        key = os.path.splitext(fname)[0]

        key_to_path[key] = fpath

    with open(txt_path, 'r') as f:
        for line in f:
            key = line.strip()

            if key in key_to_path:
                key_list.append(key)

    success_key_paths = [key_to_path[k] for k in key_list]


    return key_list, success_key_paths


def get_embeddings(model, loader, scaler, device):
    """
    遍历数据，提取 Latent Code (Mu)
    使用 get_skill_embedding 加速推理
    """
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in loader:
            # 1. 获取数据
            if isinstance(batch, dict):
                state = batch["state"]
            else:
                state = batch

            state = state.to(device)

            # 2. ★★★ 关键步骤：必须先归一化！★★★
            state_norm = scaler.transform_state(state)

            # 3. 调用加速接口 (只跑 Encoder)
            mu = model.get_skill_embedding(state_norm)

            embeddings.append(mu.cpu().numpy())

    return np.vstack(embeddings)


def create_sampled_dataset(dataset, num_samples):
    """
    从 Dataset 中随机抽取 num_samples 个样本
    """
    total_len = len(dataset)
    if total_len <= num_samples:
        return dataset  # 样本不够，全取

    # 随机生成索引
    indices = torch.randperm(total_len)[:num_samples]
    return Subset(dataset, indices)

def visualize_latent_space(
        train_data_root,
        wild_data_root,
        failed_keys_path,  # 新增：失败列表路径
        checkpoint_path,
        scaler_path,
        state_dim=272,
        action_dim=69,
        window_size=32,
        latent_dim=64,
        device="cuda",
        emb_save_path = None
):
    print(f"Loading checkpoint: {checkpoint_path}")

    # 1. 加载模型
    model = PolicySkillTreeVAE(
        state_dim=state_dim,
        action_dim=action_dim,
        window_size=window_size,
        hidden_dim=512,
        latent_dim=latent_dim
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    # 2. 加载 Scaler
    print(f"Loading scaler stats: {scaler_path}")
    scaler_state = torch.load(scaler_path, map_location=device)
    scaler = StandardScaler(device=device)
    scaler.mean_state = scaler_state["mean_state"]
    scaler.std_state = scaler_state["std_state"]

    # 3. 准备文件列表
    train_files = sorted(glob.glob(os.path.join(train_data_root, "*.npy")))
    train_paths_set =  [os.path.basename(fpath) for fpath in train_files]

    # B. 解析失败数据信息

    failed_targets, failed_paths_set = load_failed_info(failed_keys_path, wild_data_root)

    # C. 准备成功数据文件 (Wild All - Wild Failed)
    all_wild_files = sorted(glob.glob(os.path.join(wild_data_root, "*.npy")))
    success_files_txt = 'logs/smpl_ppo/amass_valid_Dec02_21-34-39/success_keys.txt'
    success_keys = load_success_key(success_files_txt, wild_data_root)
    success_files = [f for f in all_wild_files if f in success_keys]
    # split_idx = int(len(all_wild_files) * 0.9)
    # success_files = all_wild_files[split_idx:]
    # success_files = [f for f in success_files if f not in failed_paths_set and os.path.basename(f) in train_paths_set]

    print(
        f"Stats: TrainFiles={len(train_files)} | SuccessFiles={len(success_files)} | FailedTargets={len(failed_targets)}")

    # 4. 创建 Datasets
    # Train (普通滑动窗口)
    train_pool_ds = MotionWindowDataset(window_size, stride=32, action_dim=action_dim, file_paths=train_files)
    train_ds = create_sampled_dataset(train_pool_ds, num_samples=109545)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=False, num_workers=6)

    # Success (普通滑动窗口)
    wild_pool_ds = MotionWindowDataset(window_size, stride=32, action_dim=action_dim, file_paths=success_files)
    success_ds = create_sampled_dataset(wild_pool_ds, num_samples=156176)
    success_loader = DataLoader(success_ds, batch_size=2048, shuffle=False, num_workers=6)

    # ★★★ Failed (使用 TargetedMotionDataset) ★★★
    # 这里直接传入我们解析好的 [(path, frame), ...] 列表
    failed_ds = TargetedMotionDataset(
        target_frames=failed_targets,
        window_size=window_size,
        action_dim=action_dim
    )
    failed_loader = DataLoader(failed_ds, batch_size=2048, shuffle=False, num_workers=6)

    # 5. 提取特征
    print("Extracting embeddings...")
    emb_train = get_embeddings(model, train_loader, scaler, device)
    emb_success = get_embeddings(model, success_loader, scaler, device)
    emb_failed = get_embeddings(model, failed_loader, scaler, device)

    if emb_save_path is not None:
        np.savez_compressed(emb_save_path,
                            train=emb_train,
                            success=emb_success,
                            failed=emb_failed)
        print(f"Embeddings saved to {emb_save_path}")

    print(f"Embeddings -> Train: {emb_train.shape}, Success: {emb_success.shape}, Failed: {emb_failed.shape}")

    # 6. UMAP 降维
    print("Running UMAP (fitting on Train data)...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)

    # Fit on Train
    proj_train = reducer.fit_transform(emb_train)

    # Transform Wild data (mapping them into the Train manifold)
    proj_success = reducer.transform(emb_success) if len(emb_success) > 0 else np.empty((0, 2))
    proj_failed = reducer.transform(emb_failed) if len(emb_failed) > 0 else np.empty((0, 2))

    # 7. 绘图
    plt.figure(figsize=(14, 12))

    # Layer 1: Train Data (Blue, low alpha, small size) - The Skill Tree Landscape
    plt.scatter(proj_train[:, 0], proj_train[:, 1],
                c='royalblue', alpha=0.4, s=5, label='Skill Tree (Train)')

    # Layer 2: Wild Success (Green, medium alpha)
    if len(proj_success) > 0:
        plt.scatter(proj_success[:, 0], proj_success[:, 1],
                    c='limegreen', alpha=0.4, s=5, label='In-the-Wild (Success)')

    # Layer 3: Wild Failed (Red, high alpha, large size) - Highlights
    if len(proj_failed) > 0:
        plt.scatter(proj_failed[:, 0], proj_failed[:, 1],
                    c='crimson', alpha=0.8, s=10, label='In-the-Wild (Failed)')

    plt.title(f"Skill Tree Projection (UMAP)\nRed points are failed cases from '{os.path.basename(failed_keys_path)}'")
    plt.legend(markerscale=5)  # 让图例的点大一点
    plt.grid(True, alpha=0.2)
    plt.axis('off')  # 去掉坐标轴更美观

    save_file = "logs/motor_vae/skill_tree_analysis.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Done! Plot saved to {save_file}")
    plt.show()


if __name__ == "__main__":
    # 配置
    TRAIN_ROOT = "dataset/272_rep_w_action/272_w_action"
    # TRAIN_ROOT = "dataset/272_rep_w_action/amass_train_success"
    WILD_ROOT = "dataset/272_rep/AMASS_272"  # 你的 Wild 数据文件夹
    FAILED_LIST = "logs/smpl_ppo/amass_valid_Dec02_21-34-39/failed_keys.txt"  # 你的 txt 文件

    CKPT_PATH = "vae_checkpoint/mse_optimal_sigma_strict/best_model.pt"
    SCALER_PATH = "vae_checkpoint/mse_optimal_sigma_strict/scaler_stats.pt"

    visualize_latent_space(
        TRAIN_ROOT, WILD_ROOT, FAILED_LIST, CKPT_PATH, SCALER_PATH,
        state_dim=272, action_dim=69, window_size=32, latent_dim=64, emb_save_path="logs/motor_vae/train_test_embeddings_new_new.npz"
    )
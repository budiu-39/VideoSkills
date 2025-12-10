import os
import glob
import torch
import umap
from torch.utils.data import DataLoader, Subset
from motor_vae.embedding_utils import get_embeddings
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml

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

    success_file_paths = set()

    with open(txt_path, 'r') as f:
        for line in f:
            key = line.strip()

            if key in key_to_path:
                key_list.append(key)
                success_file_paths.add(key_to_path[key])


    return key_list, success_file_paths



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


def visualize_latent_space(config, args):
    """
    config: 来自模型文件夹的训练配置 (包含 hidden_dim, window_size 等)
    args:   来自命令行的参数 (包含 wild_root, failed_list 等路径)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 从 config 获取模型参数
    state_dim = config.get('state_dim', 272)
    action_dim = config.get('action_dim', 69)
    window_size = config['window_size']
    hidden_dim = config['hidden_dim']
    latent_dim = config['latent_dim']
    down_t = config.get('down_t', 3)
    norm_type = config.get('norm_type', 'LN')

    # 2. 确定路径
    # 模型路径：优先用 final_model.pt
    model_dir = args.model_dir
    if os.path.exists(os.path.join(model_dir, 'final_model.pt')):
        ckpt_path = os.path.join(model_dir, 'final_model.pt')
    elif os.path.exists(os.path.join(model_dir, 'best_model.pt')):
        ckpt_path = os.path.join(model_dir, 'best_model.pt')
    else:
        raise FileNotFoundError(f"No model found in {model_dir}")

    scaler_path = os.path.join(model_dir, 'scaler_stats.pt')

    # 数据路径：训练数据路径通常在 config 里，wild 数据路径在 args 里
    train_data_root = config['data_root']
    wild_data_root = args.wild_root
    failed_keys_path = args.failed_list
    success_keys_path = args.success_list

    exp_name = os.path.basename(os.path.normpath(model_dir))
    output_dir = os.path.join(model_dir, "embeddings_result")  # 结果直接保存在模型文件夹下
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Init] Loading Model: {ckpt_path}")
    print(f"[Init] Config: Win={window_size}, Hidden={hidden_dim}, Down={down_t}, Norm={norm_type}")

    # 3. 初始化模型
    model = PolicySkillTreeVAE(
        state_dim=state_dim,
        action_dim=action_dim,
        window_size=window_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        down_t=down_t,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 4. 加载 Scaler
    scaler_state = torch.load(scaler_path, map_location=device)
    scaler = StandardScaler(device=device)
    scaler.mean_state = scaler_state["mean_state"]
    scaler.std_state = scaler_state["std_state"]

    # 5. 准备数据
    train_files = sorted(glob.glob(os.path.join(train_data_root, "*.npy")))
    failed_targets, _ = load_failed_info(failed_keys_path, wild_data_root)

    if success_keys_path:
        _, success_key_paths = load_success_key(success_keys_path, wild_data_root)
        success_files = sorted(list(success_key_paths))
    else:
        # 如果没指定成功列表，默认读取全部 Wild 数据
        success_files = None

    print(f"Stats: Train={len(train_files)} | Success={len(success_files)} | Failed={len(failed_targets)}")

    # 6. Dataloader
    train_ds = MotionWindowDataset(window_size, 32, action_dim, train_files)
    train_ds = create_sampled_dataset(train_ds, 50000)  # 采样
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=False, num_workers=4)

    wild_ds = MotionWindowDataset(window_size, 32, action_dim, success_files)
    success_ds = create_sampled_dataset(wild_ds, 50000)
    success_loader = DataLoader(success_ds, batch_size=2048, shuffle=False, num_workers=4)

    if len(failed_targets) > 0:
        failed_ds = TargetedMotionDataset(failed_targets, window_size, action_dim)
        failed_loader = DataLoader(failed_ds, batch_size=2048, shuffle=False, num_workers=4)
    else:
        failed_loader = []

    # 7. 提取特征
    print("Extracting embeddings...")
    emb_train = get_embeddings(model, train_loader, scaler, device)
    emb_success = get_embeddings(model, success_loader, scaler, device)
    emb_failed = get_embeddings(model, failed_loader, scaler, device) if failed_loader else np.empty((0, latent_dim))

    emb_save_path = os.path.join(output_dir, "embeddings.npz")
    np.savez_compressed(
        emb_save_path,
        train=emb_train,
        success=emb_success,
        failed=emb_failed
    )
    print(f"[Info] Embeddings saved to {emb_save_path}")

    # 8. UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)

    # 用部分训练数据 Fit
    fit_data = emb_train[:20000] if len(emb_train) > 20000 else emb_train
    reducer.fit(fit_data)

    proj_train = reducer.transform(emb_train[:5000])
    proj_success = reducer.transform(emb_success[:5000]) if len(emb_success) > 0 else np.empty((0, 2))
    proj_failed = reducer.transform(emb_failed) if len(emb_failed) > 0 else np.empty((0, 2))

    # 9. 绘图
    plt.figure(figsize=(14, 12))
    plt.scatter(proj_train[:, 0], proj_train[:, 1], c='royalblue', alpha=0.3, s=5, label='Skill Tree (Train)')
    if len(proj_success) > 0:
        plt.scatter(proj_success[:, 0], proj_success[:, 1], c='limegreen', alpha=0.3, s=5, label='Wild (Success)')
    if len(proj_failed) > 0:
        plt.scatter(proj_failed[:, 0], proj_failed[:, 1], c='crimson', alpha=0.9, s=15, label='Wild (Failed)')

    plt.title(f"Skill Tree: {exp_name}")
    plt.legend(markerscale=4)
    plt.axis('off')

    out_path = os.path.join(output_dir, "umap_vis.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Done! Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 核心参数：指向模型文件夹
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing config.yaml/json and model weights')

    # 动态参数：分析用的数据路径 (因为每次分析的失败案例可能不同)
    parser.add_argument('--wild_root', type=str, required=True, help='Path to AMASS wild .npy dataset')
    parser.add_argument('--failed_list', type=str, required=True, help='Path to failed_keys.txt')
    parser.add_argument('--success_list', type=str, default=None, help='Path to success_keys.txt (optional)')

    args = parser.parse_args()

    # 1. 自动寻找配置文件
    config_path = os.path.join(args.model_dir, 'config.yaml')

    if os.path.exists(config_path):
        print(f"Loading config from YAML: {config_path}")
        with open(config_path, 'r') as f:
            train_config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"No config.yaml or config.json found in {args.model_dir}")

    # 2. 运行可视化
    visualize_latent_space(train_config, args)
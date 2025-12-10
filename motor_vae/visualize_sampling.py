import os
import glob
import torch
import umap
import json
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# 复用之前的类
from motion_window_dataset import MotionWindowDataset, TargetedMotionDataset
from motor_vae import PolicySkillTreeVAE, StandardScaler
from motor_vae.embedding_utils import get_embeddings


def load_selected_spans_as_targets(json_path, data_root, window_size):
    """
    解析 select_initial_dataset 生成的 JSON，将其转换为 TargetedMotionDataset 需要的 (path, frame) 格式
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 建立 Key -> Full Path 的映射
    print("Indexing files...")
    all_files = sorted(glob.glob(os.path.join(data_root, "**", "*.npy"), recursive=True))
    key_to_path = {}

    # 复用生成时的 Key 逻辑 (这里假设是 0-RelPath_With_Underscores)
    # 你可能需要根据实际情况微调这里的逻辑，确保能匹配上
    for fpath in all_files:
        rel = os.path.relpath(fpath, data_root)
        k = "0-" + rel.replace(os.sep, "_")
        if k.endswith(".npy"): k = k[:-4]
        key_to_path[k] = fpath

    targets = []
    matched_count = 0

    for key, spans in data.items():
        if key not in key_to_path:
            continue

        fpath = key_to_path[key]
        for span in spans:
            start, end = span
            # 我们取片段的中点作为 Latent 的提取点
            center_frame = (start + end) // 2
            targets.append((fpath, center_frame))
        matched_count += 1

    print(f"Loaded {len(targets)} clips from {matched_count} files in JSON.")
    return targets


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载 Config
    config_path = os.path.join(args.model_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 加载模型
    state_dim = config.get('state_dim', 272)
    action_dim = config.get('action_dim', 69)
    window_size = config['window_size']
    hidden_dim = config['hidden_dim']
    latent_dim = config['latent_dim']
    down_t = config.get('down_t', 4)
    norm_type = config.get('norm_type', 'LN')

    ckpt_path = os.path.join(args.model_dir, 'final_model.pt')
    if not os.path.exists(ckpt_path): ckpt_path = os.path.join(args.model_dir, 'best_model.pt')

    print(f"Loading Model: {ckpt_path}")
    model = PolicySkillTreeVAE(state_dim, action_dim, window_size, hidden_dim, latent_dim, down_t=down_t).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 3. 加载 Scaler
    scaler_path = os.path.join(args.model_dir, 'scaler_stats.pt')
    scaler_data = torch.load(scaler_path, map_location=device)
    scaler = StandardScaler(device=device)
    scaler.mean_state = scaler_data['mean_state']
    scaler.std_state = scaler_data['std_state']

    # 4. 准备数据
    # A. 背景数据 (全量/随机采样)
    data_root = args.data_root
    all_files = sorted(glob.glob(os.path.join(data_root, "**", "*.npy"), recursive=True))

    # 随机采 20000 个窗口作为背景分布
    bg_ds = MotionWindowDataset(window_size, stride=60, action_dim=action_dim, file_paths=all_files,
                                load_to_memory=False)
    bg_ds = Subset(bg_ds, torch.randperm(len(bg_ds))[:20000])
    bg_loader = DataLoader(bg_ds, batch_size=1024, shuffle=False, num_workers=4)

    # B. 选中的数据 (从 JSON 加载)
    selected_targets = load_selected_spans_as_targets(args.json_path, data_root, window_size)
    sel_ds = TargetedMotionDataset(selected_targets, window_size, action_dim, load_to_memory=False)
    sel_loader = DataLoader(sel_ds, batch_size=1024, shuffle=False, num_workers=4)

    # 5. 提取特征
    print("Extracting Background Embeddings...")
    z_bg = get_embeddings(model, bg_loader, scaler, device)

    print("Extracting Selected Embeddings...")
    z_sel = get_embeddings(model, sel_loader, scaler, device)

    # 6. 计算距离指标 (量化分析)
    # 计算全量数据的均值中心
    global_mean = np.mean(z_bg, axis=0)

    # 计算每个点到全局中心的距离 (Mahalanobis 更好，这里用欧氏简化)
    dist_bg = np.linalg.norm(z_bg - global_mean, axis=1)
    dist_sel = np.linalg.norm(z_sel - global_mean, axis=1)

    print("-" * 30)
    print(f"Avg Distance to Global Mean (Background): {np.mean(dist_bg):.4f}")
    print(f"Avg Distance to Global Mean (Selected):   {np.mean(dist_sel):.4f}")
    print("如果 Selected 距离更小，说明它们更趋向于平均动作（简单样本）。")
    print("-" * 30)

    # 7. UMAP 可视化
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.2, n_components=2, random_state=42)
    # 用背景数据训练 UMAP
    proj_bg = reducer.fit_transform(z_bg)
    # 投影选中的数据
    proj_sel = reducer.transform(z_sel)

    # 8. 绘图
    plt.figure(figsize=(12, 12))
    # 画背景 (灰色/蓝色)
    plt.scatter(proj_bg[:, 0], proj_bg[:, 1], c='lightgray', s=10, alpha=0.5, label='Full Distribution')
    # 画密度图 (可选)
    # plt.hexbin(proj_bg[:,0], proj_bg[:,1], gridsize=50, cmap='Blues', alpha=0.5)

    # 画选中点 (红色)
    plt.scatter(proj_sel[:, 0], proj_sel[:, 1], c='crimson', s=20, alpha=0.9, edgecolors='white', linewidth=0.5,
                label='Selected Clips')

    plt.title(f"Distribution of Selected Clips\nAvg Dist: Bg={np.mean(dist_bg):.2f}, Sel={np.mean(dist_sel):.2f}")
    plt.legend()
    plt.axis('off')

    save_path = args.json_path.replace('.json', '_vis.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--json_path', type=str, required=True, help='The diversity_spans.json to check')
    args = parser.parse_args()
    main(args)
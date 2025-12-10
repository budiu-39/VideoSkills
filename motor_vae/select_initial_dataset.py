import os
import glob
import torch
import numpy as np
import json
import yaml
import argparse
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm

# 引入你的项目依赖
from motion_window_dataset import MotionWindowDataset
from motor_vae import PolicySkillTreeVAE, StandardScaler


# ================= 辅助函数 =================

def make_key_from_path(fpath, root):
    """
    生成与 amass_split_clip.py 兼容的 Key。
    假设特征数据的目录结构与原始数据一致：
    Feature Path: .../272_w_action/CMU/01/01_01_poses.npy
    Raw Path:     .../AMASS_train/CMU/01/01_01_poses.npy
    Key:          0-CMU_01_01_01_poses
    """
    rel = os.path.relpath(fpath, root)
    k = "0-" + rel.replace(os.sep, "_")
    if k.endswith(".npy"):
        k = k[:-4]
    return k


def merge_intervals(intervals, gap_threshold=30):
    """合并重叠或邻近的时间段"""
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])

    merged = []
    current_start, current_end = intervals[0]

    for next_start, next_end in intervals[1:]:
        if next_start <= current_end + gap_threshold:
            current_end = max(current_end, next_end)
        else:
            merged.append([current_start, current_end])
            current_start, current_end = next_start, next_end

    merged.append([current_start, current_end])
    return merged


# ================= 主流程 =================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载配置
    config_path = os.path.join(args.model_dir, 'config.yaml')
    json_path = os.path.join(args.model_dir, 'config.json')  # 兼容 JSON

    if os.path.exists(config_path):
        print(f"[Init] Loading config from YAML: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif os.path.exists(json_path):
        print(f"[Init] Loading config from JSON: {json_path}")
        with open(json_path, 'r') as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"No config.yaml or config.json found in {args.model_dir}")

    # 2. 确定模型权重路径
    if os.path.exists(os.path.join(args.model_dir, 'best_model.pt')):
        ckpt_path = os.path.join(args.model_dir, 'best_model.pt')
    elif os.path.exists(os.path.join(args.model_dir, 'final_model.pt')):
        ckpt_path = os.path.join(args.model_dir, 'final_model.pt')
    else:
        raise FileNotFoundError(f"No model weights found in {args.model_dir}")

    scaler_path = os.path.join(args.model_dir, 'scaler_stats.pt')

    # 3. 解析参数
    state_dim = config.get('state_dim', 272)
    action_dim = config.get('action_dim', 69)
    window_size = config['window_size']
    hidden_dim = config['hidden_dim']
    latent_dim = config['latent_dim']
    down_t = config.get('down_t', 4)
    norm_type = config.get('norm_type', 'LN')

    # 数据路径 (优先使用命令行参数，其次使用 config 里的，但 config 里的通常是训练路径)
    data_root = args.data_root if args.data_root else config['data_root']

    # 采样参数
    target_num_clips = args.target_clips
    clip_duration = args.clip_duration
    sampling_stride = args.stride
    batch_size = args.batch_size
    output_json = args.output

    print(f"[Init] Loading Model from {ckpt_path}")
    print(f"[Init] Model Config: Win={window_size}, Hidden={hidden_dim}, Down={down_t}, Norm={norm_type}")

    # 4. 初始化模型
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

    # 5. 加载 Scaler
    print(f"[Init] Loading Scaler from {scaler_path}")
    scaler_state = torch.load(scaler_path, map_location=device)
    scaler = StandardScaler(device=device)
    scaler.mean_state = scaler_state["mean_state"]
    scaler.std_state = scaler_state["std_state"]

    # 6. 准备数据
    print(f"[Init] Indexing files in {data_root}...")
    all_files = sorted(glob.glob(os.path.join(data_root, "**", "*.npy"), recursive=True))
    print(f"Found {len(all_files)} feature files.")

    if len(all_files) == 0:
        raise ValueError(f"No .npy files found in {data_root}. Please check the path.")

    dataset = MotionWindowDataset(
        window_size=window_size,
        stride=sampling_stride,
        action_dim=action_dim,
        file_paths=all_files,
        load_to_memory=False
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # 7. 提取特征
    print(f"[Process] Extracting Latent Codes (Stride={sampling_stride})...")

    all_embeddings = []
    valid_indices_map = dataset.index

    with torch.no_grad():
        for batch in tqdm(loader):
            state = batch["state"].to(device)
            state = scaler.transform_state(state)

            if hasattr(model, 'encode'):
                encoded_out = model.encode(state)
                if isinstance(encoded_out, tuple):
                    z = encoded_out[0]
                elif hasattr(encoded_out, 'loc'):
                    z = encoded_out.loc
                else:
                    z = encoded_out
            else:
                recon, pred_action, mu, logvar = model(state)
                z = mu

            all_embeddings.append(z.cpu().numpy())

    if len(all_embeddings) == 0:
        print("No embeddings extracted. Check dataset.")
        return

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Total extracted windows: {all_embeddings.shape[0]}")

    # 8. 聚类采样
    print(f"[Process] Clustering into {target_num_clips} centers...")

    kmeans = MiniBatchKMeans(
        n_clusters=target_num_clips,
        random_state=42,
        batch_size=4096,
        n_init='auto'
    )
    kmeans.fit(all_embeddings)

    print("[Process] Selecting DIVERSE representatives (Not just centers)...")

    selected_indices = []

    # 获取每个样本所属的 Cluster Label
    labels = kmeans.labels_  # [N_samples]

    # 遍历每个 Cluster
    for i in range(target_num_clips):
        # 找到属于 Cluster i 的所有样本索引
        cluster_indices = np.where(labels == i)[0]

        if len(cluster_indices) == 0:
            continue

        # 获取这些样本的 Embeddings
        cluster_embs = all_embeddings[cluster_indices]  # [M, D]
        center = kmeans.cluster_centers_[i]  # [D]

        # --- 策略 A: 选离 Cluster 中心最远的点 (Outlier within Cluster) ---
        # dists = np.linalg.norm(cluster_embs - center, axis=1)
        # target_idx_in_cluster = np.argmax(dists)

        # --- 策略 B: 随机采样 (Random) ---
        # 避免总是选中位数，增加随机性
        # target_idx_in_cluster = np.random.randint(len(cluster_indices))

        # --- 策略 C (推荐): 概率采样 (按距离加权) ---
        # 离中心越远，被选中的概率越大 (倾向于边缘样本，但保留一定的中心性)
        dists = np.linalg.norm(cluster_embs - center, axis=1)
        # 归一化距离作为概率 (加个小 epsilon 防止全0)
        probs = dists / (np.sum(dists) + 1e-6)
        # 按概率抽取 1 个
        target_idx_in_cluster = np.random.choice(len(cluster_indices), p=probs)

        # 映射回全局索引
        global_idx = cluster_indices[target_idx_in_cluster]
        selected_indices.append(global_idx)

    selected_indices = sorted(list(set(selected_indices)))
    print(f"Selected {len(selected_indices)} unique seed windows using Weighted Sampling.")

    # 9. 构建结果
    file_idx_to_spans = {}

    for global_idx in selected_indices:
        file_idx, start_frame = valid_indices_map[global_idx]

        center = start_frame + window_size // 2
        clip_half = clip_duration // 2

        clip_start = center - clip_half
        clip_end = center + clip_half

        if file_idx not in file_idx_to_spans:
            file_idx_to_spans[file_idx] = []
        file_idx_to_spans[file_idx].append([clip_start, clip_end])

    # 10. 格式化并保存
    print("[Process] Merging overlaps and formatting keys...")
    final_dict = {}

    for file_idx, intervals in file_idx_to_spans.items():
        fpath = dataset.files[file_idx]
        key = make_key_from_path(fpath, data_root)  # 使用当前的 data_root 生成 key

        merged = merge_intervals(intervals, gap_threshold=30)
        merged = [[int(s), int(e)] for s, e in merged]

        final_dict[key] = merged

    # 11. 保存为 JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    print(f"[Output] Saving spans to {output_json}...")

    with open(output_json, 'w') as f:
        json.dump(final_dict, f, indent=4)

    print("Done! Next steps:")
    print(f"1. Check the JSON file at {output_json}")
    print(f"2. Run: python amass_split_clip.py --spans_json {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diversity Sampling using MotorVAE")

    # 必需参数
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing config.yaml/json and model weights')

    # 可选参数 (覆盖 Config 或 默认值)
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to feature dataset (if different from config)')

    parser.add_argument('--output', type=str, default="logs/motor_vae/diversity_spans.json",
                        help='Output JSON path')

    # 采样超参数
    parser.add_argument('--target_clips', type=int, default=400, help='Number of clusters/clips to select')
    parser.add_argument('--clip_duration', type=int, default=96, help='Duration of exported clips (frames)')
    parser.add_argument('--stride', type=int, default=15, help='Stride for window extraction')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for inference')

    args = parser.parse_args()

    main(args)
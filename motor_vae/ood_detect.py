import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # 需要 pip install seaborn
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

emb_path = "logs/motor_vae/train_test_embeddings_new.npz"

print(f"Loading {emb_path}...")
data = np.load(emb_path)
emb_train   = data["train"]
emb_success = data["success"]
emb_failed  = data["failed"]

print(f"Original Shapes -> Train: {emb_train.shape}, Success: {emb_success.shape}, Failed: {emb_failed.shape}")

# ==========================================
# 改进 1: 构建索引时对 Train 进行下采样
# ==========================================
# 目的：打破轨迹的时间连续性，让最近邻寻找真正的“相似动作”而不是“下一帧”
# 建议采样到 20000 - 50000 个点即可
max_ref_points = 50000
if len(emb_train) > max_ref_points:
    # 随机采样构建索引
    np.random.seed(42)
    idx = np.random.choice(len(emb_train), max_ref_points, replace=False)
    ref_data = emb_train[idx]
    print(f"Subsampled Train for Index: {ref_data.shape}")
else:
    ref_data = emb_train

# 增大 K 值，进一步跳出局部邻域
k = 50
print(f"Building KNN Index (k={k})...")
nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(ref_data)

# ==========================================
# 2. 计算距离 (OOD Score)
# ==========================================
print("Computing distances...")
# 对于 Train 自身，我们也只对一部分点计算距离用于画图（节省时间且分布一致）
# 或者计算全量
dist_train, _ = nbrs.kneighbors(emb_train)
dist_succ, _  = nbrs.kneighbors(emb_success) if len(emb_success) > 0 else (np.empty((0, k)), None)
dist_fail, _  = nbrs.kneighbors(emb_failed)  if len(emb_failed) > 0 else (np.empty((0, k)), None)

# 取第 k 个邻居的距离作为 Score
score_train = dist_train[:, -1]
score_succ  = dist_succ[:, -1] if len(emb_success) > 0 else np.array([])
score_fail  = dist_fail[:, -1] if len(emb_failed) > 0 else np.array([])

# ==========================================
# 3. 统计指标 & AUC
# ==========================================
# 阈值：Train 的 95% 分位数
thr = np.percentile(score_train, 95)

def get_stats(scores, name, threshold):
    if len(scores) == 0: return
    ratio = (scores > threshold).mean()
    mean_dist = scores.mean()
    print(f"[{name:7s}] Mean Dist: {mean_dist:.4f} | Outlier Ratio (>Train95%): {ratio:.2%}")

print("-" * 60)
print(f"Threshold (Train 95%): {thr:.4f}")
get_stats(score_train, "Train", thr)
get_stats(score_succ,  "Success", thr)
get_stats(score_fail,  "Failed", thr)

# 计算 AUC (分辨 Train 和 Failed 的能力)
if len(score_fail) > 0:
    # 构造标签：Train=0, Failed=1
    y_true = np.concatenate([np.zeros(len(score_train)), np.ones(len(score_fail))])
    y_scores = np.concatenate([score_train, score_fail])
    auc = roc_auc_score(y_true, y_scores)
    print(f"\n[Metric] OOD Detection AUC (Train vs Failed): {auc:.4f}")
    # AUC > 0.8 说明距离指标非常有效
print("-" * 60)

# ==========================================
# 改进 2: 使用 KDE 绘图
# ==========================================
plt.figure(figsize=(10, 6))

# 使用 seaborn kdeplot，自动处理样本量不平衡问题，且曲线平滑
sns.kdeplot(score_train, fill=True, color='blue', label='Train (Skill Tree)', alpha=0.3, clip=(0, None))
if len(score_succ) > 0:
    sns.kdeplot(score_succ, fill=True, color='green', label='Wild Success', alpha=0.3, clip=(0, None))
if len(score_fail) > 0:
    sns.kdeplot(score_fail, fill=True, color='red', label='Wild Failed', alpha=0.3, clip=(0, None))

plt.axvline(thr, color='k', linestyle='--', label='95% Threshold')
plt.title(f"OOD Analysis: Distance to {k}-th Neighbor in Skill Tree")
plt.xlabel("Latent Distance")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)

out_file = "logs/motor_vae/knn_density_analysis.png"
plt.savefig(out_file, dpi=300)
print(f"Saved plot to {out_file}")
plt.show()
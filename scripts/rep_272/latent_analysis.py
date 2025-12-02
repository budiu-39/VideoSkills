import json
import numpy as np

# 文件路径
json_path = "/mnt/lustre/work/ponsmoll/pba936/MotionStreamer/data/AMASS/embeddings.json"
npz_path = "/mnt/lustre/work/ponsmoll/pba936/MotionStreamer/data/AMASS_new/embeddings.npz"

# 加载 JSON
with open(json_path, "r", encoding="utf-8") as f:
    json_emb = json.load(f)   # dict: key -> list[256]

# 加载 NPZ
npz_data = np.load(npz_path, allow_pickle=True)
npz_emb = npz_data["embeds"]     # shape (N, 256)
npz_keys = npz_data["keys"]      # shape (N,)

# 形成 key → embedding 的 dict（方便对比）
npz_dict = {k: npz_emb[i] for i, k in enumerate(npz_keys)}

# 找出两个文件的交集 key
common_keys = list(set(json_emb.keys()) & set(npz_dict.keys()))

print(f"共同 key 数量: {len(common_keys)}")

# 取前 10 个交集 key
sample_keys = common_keys[:10]
print("取出的前 10 个 key:")
for k in sample_keys:
    print("  ", k)

print("\n===== 逐个比较 embedding 是否完全一致 =====")
for k in sample_keys:
    emb_json = np.array(json_emb[k])
    emb_npz = npz_dict[k]

    # 误差（L2 范数）
    diff = np.linalg.norm(emb_json - emb_npz)

    print(f"Key = {k}")
    print(f"  JSON embedding shape: {emb_json.shape}")
    print(f"  NPZ  embedding shape: {emb_npz.shape}")
    print(f"  差异 L2 norm = {diff:.6f}")

    if np.allclose(emb_json, emb_npz, atol=1e-6):
        print("  → ✔ 完全一致 (within tolerance)")
    else:
        print("  → ✘ 不一致")
    print()

# ========== 新增部分：在同一个 embedding 文件内随机比较 ==========
print("===== 在同一个 NPZ embedding 文件内随机比较若干对样本的距离 =====")

num_pairs = 10  # 想看多少对就改这个
N = npz_emb.shape[0]

for i in range(num_pairs):
    # 随机抽两个不同的索引
    idx1, idx2 = np.random.randint(0, N, size=2)
    if idx1 == idx2:
        continue

    v1 = npz_emb[idx1]
    v2 = npz_emb[idx2]
    d = np.linalg.norm(v1 - v2)

    print(f"Pair {i+1}: {npz_keys[idx1]}  vs  {npz_keys[idx2]}")
    print(f"  L2 距离 = {d:.6f}")


print("===== 在同一个 json embedding 文件内随机比较若干对样本的距离 =====")

num_pairs = 10  # 想看多少对就改这个
N = emb_json.shape[0]

for i in range(num_pairs):
    # 随机抽两个不同的索引
    idx1, idx2 = np.random.randint(0, N, size=2)
    if idx1 == idx2:
        continue

    v1 = emb_json[idx1]
    v2 = emb_json[idx2]
    d = np.linalg.norm(v1 - v2)

    print(f"Pair {i+1}: {npz_keys[idx1]}  vs  {npz_keys[idx2]}")
    print(f"  L2 距离 = {d:.6f}")

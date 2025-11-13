#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv
import numpy as np
from collections import defaultdict

# ====== 配置 ======
NPZ_PATH         = "../MotionStreamer/data/AMASS/causal_16/causal_latents_per_motion.npz"  # 含有每个 key 的 {'mu','spans','T'}
OUT_SPANS_NPZ    = "picked_800s_len48.npz"
ALLOWED_KEYS_TXT = "/mnt/lustre/work/ponsmoll/pba936/MotionStreamer/data/AMASS/amass_train_keys.txt"
SEED             = 42

FPS          = 30
TOTAL_SEC    = 800
TOTAL_FRAMES = TOTAL_SEC * FPS          # 24000
WIN          = 48
STRIDE       = 48                        # 不重叠；想允许重叠就改小
NEEDED_SEGS  = 500

def load_allowed_keys(txt_path):
    if not txt_path: return None
    with open(txt_path, "r", encoding="utf-8") as f:
        return set(ln.strip() for ln in f if ln.strip())

def main():
    rng = np.random.default_rng(SEED)
    npz = np.load(NPZ_PATH, allow_pickle=True)
    allowed = load_allowed_keys(ALLOWED_KEYS_TXT)

    # 收集候选 (key, start, end)
    candidates = []
    for mkey in npz.files:
        if allowed is not None and mkey not in allowed:
            continue
        obj = npz[mkey].item()
        T = int(obj.get('T', 0))
        if T < WIN:
            continue
        # 生成窗口（无重叠；若想重叠，把 STRIDE 改小）
        for s in range(0, T - WIN + 1, STRIDE):
            candidates.append((mkey, s, s + WIN))

    if len(candidates) == 0:
        raise RuntimeError("没有可用候选片段（检查 NPZ_PATH / ALLOWED_KEYS_TXT / WIN / STRIDE）")

    # 随机不放回采样
    pick_idx = rng.choice(len(candidates), size=NEEDED_SEGS, replace=False)
    picked = [candidates[i] for i in pick_idx]

    # 聚合为 key -> spans
    key2spans = defaultdict(list)
    for k, s, e in picked:
        key2spans[k].append([s, e])

    # 保存 npz（每个 key 一个 Nx2 的 int32 数组）
    packed = {k: np.asarray(v, dtype=np.int32) for k, v in key2spans.items()}
    np.savez_compressed(OUT_SPANS_NPZ, **packed)
    print(f"[Saved] NPZ spans → {OUT_SPANS_NPZ}  （共 {len(picked)} 段，覆盖约 {len(picked)*WIN/FPS:.1f} 秒）")

    # 简要统计
    n_keys = len(packed)
    n_segs = sum(len(v) for v in packed.values())
    print(f"[Info] 选中 {n_segs} 段（每段 {WIN} 帧），来自 {n_keys} 个 motions；"
          f"总帧数 = {n_segs*WIN}，约 {n_segs*WIN/FPS:.1f} 秒。")

if __name__ == "__main__":
    main()

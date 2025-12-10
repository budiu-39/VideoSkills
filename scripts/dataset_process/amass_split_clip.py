#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal exporter without LATENTS_NPZ.

- Read frame-level spans from SPANS_NPZ: each item is a closed interval (s, e) in frames.
- Map key -> source AMASS .npy, load SkeletonState to get T_file.
- Normalize span order, extend by ±EXTEND, clamp to [0, T_file-1].
- Slice and save as SkeletonMotion .npy under OUT_DIR.
- Print a concise summary (counts, total frames, total duration).
"""

import os
import glob
import json
import numpy as np
from scripts.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

# ========= PATHS =========
AMASS_ROOT = "dataset/smpl_motion/AMASS_train_fixed_height"  # AMASS 源数据根目录
# AMASS_ORIGIN = "/mnt/lustre/work/ponsmoll/pba936/AMASS"
# SPANS_NPZ  = "/mnt/lustre/work/ponsmoll/pba936/MotionStreamer/data/AMASS/causal_16_text/picked_segments_with_span.npz"
SPANS_JSON = "logs/motor_vae/diversity_spans_96w_edge.json"
# LATENT_NPZ = "/mnt/lustre/work/ponsmoll/pba936/MotionStreamer/data/AMASS/causal_16/causal_latents_per_motion.npz"

# 修改为 JSON 文件路径

OUT_DIR    = "dataset/smpl_motion/initial_96w_300clips_edge"

FPS        = 30
EXTEND     = 0

# ========= HELPERS =========
def make_key_from_path(fpath: str) -> str:
    """key = '0-' + relpath(AMASS_ROOT) with slashes -> '_' and no .npy"""
    rel = os.path.relpath(fpath, AMASS_ROOT)
    k = '0-' + rel.replace(os.sep, "-")
    if k.endswith(".npy"):
        k = k[:-4]
    return k


def build_key_to_path(amass_root: str):
    exts = ("npy", "npz")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(amass_root, "**", f"*.{ext}"), recursive=True))
    return {make_key_from_path(p): p for p in paths}

def slice_to_motion(sk_state: SkeletonState, s: int, e: int) -> SkeletonMotion:
    sub_root   = sk_state.root_translation[s:e+1]
    sub_global = sk_state.global_rotation[s:e+1]
    sub_state  = SkeletonState.from_rotation_and_root_translation(
        sk_state.skeleton_tree, sub_global, sub_root, is_local=False
    )
    return SkeletonMotion.from_skeleton_state(sub_state, FPS)

# ========= MAIN =========
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[Info] Using spans json : {SPANS_JSON}")
    print(f"[Info] AMASS root      : {AMASS_ROOT}")

    # ★★★ 修改为 JSON load ★★★
    with open(SPANS_JSON, 'r') as f:
        spans_data = json.load(f)

    missing_keys = []
    exported_rows = []
    total_segments = 0
    saved_segments = 0
    total_frames = 0

    key2path  = build_key_to_path(AMASS_ROOT)

    # ★★★ 遍历 JSON 字典的 keys ★★★
    for key in sorted(spans_data.keys()):
        # 将 list of lists 转为 numpy 数组，以便复用后续逻辑
        spans = np.array(spans_data[key])

        # 简单校验
        if spans.ndim != 2 or spans.shape[1] != 2:
            print(f"[Warn] bad spans shape for {key}: {spans.shape}, skip.")
            continue

        src = key2path.get(key)
        if not src:
            missing_keys.append(key)
            continue

        motion_dict = np.load(src, allow_pickle=True).item()
        sk = SkeletonState.from_dict(motion_dict)
        T_file = int(sk.root_translation.shape[0])

        for i, (s_raw, e_raw) in enumerate(spans.tolist()):
            total_segments += 1
            s0, e0 = int(s_raw), int(e_raw)
            if e0 < s0:
                s0, e0 = e0, s0

            # extend & clamp with file T
            s_ext = max(0, s0 - EXTEND)
            e_ext = min(T_file - 1, e0 + EXTEND)

            if e_ext <= s_ext:
                print(f"[Warn] invalid extended span for {key} seg{i}: {s_ext}-{e_ext} (raw={s_raw}-{e_raw}, T={T_file})")
                continue

            try:
                sub_motion = slice_to_motion(sk, s_ext, e_ext)
            except Exception as e:
                print(f"[Err] slice failed {key} seg{i}: {s_ext}-{e_ext} ({e})")
                continue

            seg_len = int(e_ext - s_ext + 1)
            total_frames += seg_len

            out_name = f"{key}_seg{i:03d}_{s_ext}-{e_ext}.npy"
            out_path = os.path.join(OUT_DIR, out_name)
            sub_motion.to_file(out_path)
            saved_segments += 1
            exported_rows.append([key, i, s_raw, e_raw, s_ext, e_ext, seg_len, out_path])

    # reports
    if missing_keys:
        with open(os.path.join(OUT_DIR, "missing_keys.json"), "w", encoding="utf-8") as f:
            json.dump(missing_keys, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "exported_segments.tsv"), "w", encoding="utf-8") as f:
        f.write("key\tseg_idx\ts\te\ts_ext\te_ext\tlen\tpath\n")
        for row in exported_rows:
            f.write("\t".join(map(str, row)) + "\n")

    secs = total_frames / float(FPS)
    print("=" * 60)
    print(f"[Done] total span segments = {total_segments}, saved = {saved_segments}")
    print(f"[Info] Total frames exported = {total_frames}")
    print(f"[Info] Total duration        = {secs:.2f} s (~{secs/60.0:.2f} min)")
    if missing_keys:
        print(f"[Info] Missing keys logged   -> {os.path.join(OUT_DIR, 'missing_keys.json')}  (count={len(missing_keys)})")
    print(f"[Info] Exported list         -> {os.path.join(OUT_DIR, 'exported_segments.tsv')}")

if __name__ == "__main__":
    main()

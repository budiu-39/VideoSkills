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
SPANS_NPZ  = "/mnt/lustre/work/ponsmoll/pba936/MotionStreamer/data/AMASS/causal_16_text/picked_segments_with_span.npz"
OUT_DIR    = "dataset/smpl_motion/causal_16_text_export_min"  # 导出目录

FPS        = 30
EXTEND     = 8  # 前后各扩展帧数（总32 = 前后各8）

# ========= HELPERS =========
def make_key_from_path(fpath: str) -> str:
    """key = '0-' + relpath(AMASS_ROOT) with slashes -> '_' and no .npy"""
    rel = os.path.relpath(fpath, AMASS_ROOT)
    k = "0-" + rel.replace(os.sep, "_")
    if k.endswith(".npy"):
        k = k[:-4]
    return k

def build_key_to_path(amass_root: str):
    paths = glob.glob(os.path.join(amass_root, "**", "*.npy"), recursive=True)
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
    print(f"[Info] Using spans npz : {SPANS_NPZ}")
    print(f"[Info] AMASS root      : {AMASS_ROOT}")

    spans_npz = np.load(SPANS_NPZ, allow_pickle=True)
    key2path  = build_key_to_path(AMASS_ROOT)

    missing_keys = []
    exported_rows = []
    total_segments = 0
    saved_segments = 0
    total_frames = 0

    for key in sorted(spans_npz.files):
        spans = np.asarray(spans_npz[key])
        if spans.ndim != 2 or spans.shape[1] != 2:
            print(f"[Warn] bad spans shape for {key}: {spans.shape}, skip.")
            continue

        src = key2path.get(key)
        if not src:
            missing_keys.append(key)
            continue

        # load source motion once
        try:
            motion_dict = np.load(src, allow_pickle=True).item()
            sk = SkeletonState.from_dict(motion_dict)
            T_file = int(sk.root_translation.shape[0])
        except Exception as e:
            print(f"[Err] load failed: {src} ({e})")
            continue

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
            try:
                sub_motion.to_file(out_path)
                saved_segments += 1
                exported_rows.append([key, i, s_raw, e_raw, s_ext, e_ext, seg_len, out_path])
            except Exception as e:
                print(f"[Err] save failed: {out_path} ({e})")

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

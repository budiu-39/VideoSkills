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
import os.path as osp
import glob
import json
import numpy as np
from videoskills.utils.torch_utils import calc_heading, quat_multiply, my_quat_rotate
from isaacgym.torch_utils import quat_from_angle_axis
from scripts.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import shutil
from scripts.preprocess.amass2smpl import vis_mujoco

import torch

# ========= PATHS =========
AMASS_ROOT = "dataset/smpl_motion/AMASS_train_fixed_height"  # AMASS 源数据根目录
SELECTED_MOTION_TXT = "../MotionStreamer/data/AMASS/causal_16/selected_motions_100.txt"
SPANS_NPZ  = "../MotionStreamer/data/AMASS/causal_16/augment_pairs_OR.npz"
OUT_DIR    = "dataset/smpl_motion/traj_augmented"  # 导出目录

FPS = 30
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

def concat_motion(motion_a_key, motion_a_end, motion_b_key, motion_b_start, key2path) -> SkeletonMotion:
    motion_a_src = key2path.get(motion_a_key)
    motion_b_src = key2path.get(motion_b_key)

    motion_dict_a = np.load(motion_a_src, allow_pickle=True).item()
    motion_dict_b = np.load(motion_b_src, allow_pickle=True).item()
    sk_a = SkeletonState.from_dict(motion_dict_a)
    sk_b = SkeletonState.from_dict(motion_dict_b)

    A_root   = sk_a.root_translation[:motion_a_end+1]
    A_local  = sk_a.local_rotation[:motion_a_end+1]
    B_root   = sk_b.root_translation[motion_b_start:]
    B_local  = sk_b.local_rotation[motion_b_start:]

    device = A_root.device
    dtype  = A_root.dtype  # 一般是 torch.float32

    A_root  = A_root.to(device=device, dtype=dtype)
    A_local = A_local.to(device=device, dtype=dtype)
    B_root  = B_root.to(device=device, dtype=dtype)
    B_local = B_local.to(device=device, dtype=dtype)

    # 在 cat 之前要确保 A 的最后一帧和 B 的第一帧对齐，只旋转 yaw
    A_end_q = A_local[-1, 0].unsqueeze(0)  # [1,4]
    B_start_q = B_local[0, 0].unsqueeze(0)
    yaw_A = calc_heading(A_end_q)  # [1]
    yaw_B = calc_heading(B_start_q)
    yaw_diff = yaw_A - yaw_B


    z_axis   = torch.tensor([[0., 0., 1.]], device=device, dtype=dtype)
    yaw_quat = quat_from_angle_axis(yaw_diff, z_axis)  # [1,4]
    yaw_quat = yaw_quat / torch.norm(yaw_quat, dim=-1, keepdim=True)

    B_local_aligned = B_local.clone()
    B_local_aligned[:, 0] = quat_multiply(yaw_quat.repeat(B_local.shape[0], 1), B_local[:, 0])

    # --- 3) 旋转 B 的根位置 ---
    B_root_rel = B_root - B_root[0]  # 以首帧为原点
    B_root_rot = my_quat_rotate(yaw_quat.repeat(B_root_rel.shape[0], 1), B_root_rel)
    # 平移到 A 的末帧
    B_root_aligned = B_root_rot + A_root[-1]

    cat_root   = torch.cat([A_root, B_root_aligned], dim=0)
    cat_local = torch.cat([A_local, B_local_aligned], dim=0)

    cat_state = SkeletonState.from_rotation_and_root_translation(
        sk_a.skeleton_tree, cat_local, cat_root, is_local=True
    )

    motion_traj = {}
    motion_traj['root_trans_offset'] = cat_state.root_translation.numpy()
    motion_traj['root_rotation'] = cat_state.global_root_rotation.numpy()
    motion_traj['dof'] = sRot.from_quat(cat_state.local_rotation[:, 1:].reshape(-1, 4)).as_rotvec().reshape(-1, 23, 3)
    vis_mujoco(motion_traj, f"data/robots/smpl/smpl_humanoid.xml")
    return SkeletonMotion.from_skeleton_state(cat_state, FPS)
# ========= MAIN =========
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[Info] Using spans npz : {SPANS_NPZ}")
    print(f"[Info] AMASS root      : {AMASS_ROOT}")

    spans_npz = np.load(SPANS_NPZ, allow_pickle=True)
    key2path  = build_key_to_path(AMASS_ROOT)

    # ========================
    # NEW: 两类导出各自的统计器
    # ========================
    # concat 部分
    concat_rows = []         # 每条拼接的记录（写入TSV）
    concat_frames_total = 0  # 拼接后总帧数
    concat_count = 0         # 拼接条目计数

    # 原版拷贝部分
    orig_rows = []           # 每个原始文件的记录（写入TSV）
    orig_frames_total = 0    # 原始段总帧数（可选：读取以统计）
    orig_count = 0           # 原始文件计数

    # ========== 1) CONCAT 部分 ==========
    # 如需开启拼接导出，就把下面这段的注释去掉
    for key in sorted(spans_npz.files):
        spans = spans_npz[key].item()
        motion_a_key  = spans['motion_a_key']
        motion_b_key  = spans['motion_b_key']
        motion_a_end  = int(spans['motion_a_end'])
        motion_b_start= int(spans['motion_b_start'])

        # 生成拼接结果
        motion_concated = concat_motion(motion_a_key, motion_a_end, motion_b_key, motion_b_start, key2path)

        # 保存
        save_path = osp.join(OUT_DIR, "concat", key + ".npy")  # 放到 OUT_DIR/concat/ 下，便于区分
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        motion_concated.to_file(save_path)

        # 统计信息
        T_out = int(motion_concated.root_translation.shape[0])
        concat_frames_total += T_out
        concat_count += 1
        concat_rows.append([
            key,                 # aug 名称（如 aug0001）
            motion_a_key, motion_a_end,
            motion_b_key, motion_b_start,
            T_out,               # 输出帧数
            save_path
        ])

    # ========== 2) 原版拷贝部分 ==========
    motion_txt = os.path.join(SELECTED_MOTION_TXT)
    with open(motion_txt, "r", encoding="utf-8") as f:
        selected_keys = [line.strip() for line in f if line.strip()]

    for key in selected_keys:
        src_path = key2path.get(key)
        if src_path is None:
            print(f"[Warn] Missing path for {key}")
            continue

        rel_path  = os.path.relpath(src_path, AMASS_ROOT)
        dest_path = os.path.join(OUT_DIR, "original", rel_path)  # 放到 OUT_DIR/original/ 下
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)

        # 可选：读取一遍来统计帧数（如果不需要，可把下面3行注释掉）
        motion_dict = np.load(src_path, allow_pickle=True).item()
        sk = SkeletonState.from_dict(motion_dict)
        T_file = int(sk.root_translation.shape[0])
        orig_frames_total += T_file

        orig_count += 1
        orig_rows.append([key, T_file, src_path, dest_path])

    # ========== 写出两份 TSV 清单 ==========
    # concat.tsv
    concat_tsv = os.path.join(OUT_DIR, "concat_export.tsv")
    os.makedirs(os.path.dirname(concat_tsv), exist_ok=True)
    with open(concat_tsv, "w", encoding="utf-8") as f:
        f.write("aug_key\tmotion_a_key\tmotion_a_end\tmotion_b_key\tmotion_b_start\tframes_out\tsave_path\n")
        for r in concat_rows:
            f.write("\t".join(map(str, r)) + "\n")

    # original.tsv
    orig_tsv = os.path.join(OUT_DIR, "original_export.tsv")
    os.makedirs(os.path.dirname(orig_tsv), exist_ok=True)
    with open(orig_tsv, "w", encoding="utf-8") as f:
        f.write("motion_key\tframes\tsrc_path\tdest_path\n")
        for r in orig_rows:
            f.write("\t".join(map(str, r)) + "\n")

    # ========== 分块总结输出 ==========
    print("=" * 60)
    print("# Concat 部分")
    concat_secs = concat_frames_total / float(FPS)
    print(f"[Concat] 条目数        : {concat_count}")
    print(f"[Concat] 总帧数         : {concat_frames_total}")
    print(f"[Concat] 总时长         : {concat_secs:.2f} s (~{concat_secs/60.0:.2f} min)")
    print(f"[Concat] 导出清单       : {concat_tsv}")

    print("-" * 60)
    print("# 原版拷贝部分")
    orig_secs = orig_frames_total / float(FPS) if orig_frames_total > 0 else 0.0
    print(f"[Original] 文件数量     : {orig_count}")
    print(f"[Original] 总帧数（可选）: {orig_frames_total}")
    print(f"[Original] 总时长（可选）: {orig_secs:.2f} s (~{orig_secs/60.0:.2f} min)")
    print(f"[Original] 导出清单     : {orig_tsv}")

if __name__ == "__main__":
    main()

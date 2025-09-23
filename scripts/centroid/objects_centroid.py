#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, os.path as osp, json, csv, shutil
from typing import List, Dict, Tuple
import numpy as np, trimesh

def find_textures_in_mtl(mtl_path: str) -> List[str]:
    if not osp.exists(mtl_path): return []
    textures, exts = [], (".png",".jpg",".jpeg",".tga",".bmp",".exr",".hdr",".tiff",".tif")
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"): continue
            key = ln.split(maxsplit=1)[0].lower()
            if key in ("map_kd","map_ks","map_bump","bump","map_d","map_ns","disp","map_ka"):
                parts = ln.split()
                for tok in reversed(parts):
                    if tok.lower().endswith(exts):
                        textures.append(tok); break
    seen, uniq = set(), []
    for t in textures:
        if t not in seen: uniq.append(t); seen.add(t)
    return uniq

def safe_copy_with_dirs(src_root: str, dst_root: str, rel_path: str):
    src_path = osp.join(src_root, rel_path)
    dst_path = osp.join(dst_root, rel_path)
    os.makedirs(osp.dirname(dst_path), exist_ok=True)
    if osp.isfile(src_path):
        shutil.copy2(src_path, dst_path)

def copy_mtl_and_textures(src_obj_path: str, dst_obj_path: str):
    src_dir, dst_dir = osp.dirname(src_obj_path), osp.dirname(dst_obj_path)
    base = osp.splitext(osp.basename(src_obj_path))[0]
    src_mtl = osp.join(src_dir, base + ".mtl")
    if not osp.exists(src_mtl): return
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src_mtl, osp.join(dst_dir, osp.basename(src_mtl)))
    for rel_tex in find_textures_in_mtl(src_mtl):
        safe_copy_with_dirs(src_dir, dst_dir, rel_tex)

def compute_anchor_offset(mesh, mode="vertex_mean", up_axis="z"):
    if mode == "vertex_mean":
        return mesh.vertices.mean(axis=0).astype(np.float32)
    elif mode == "mesh_centroid":
        return mesh.centroid.astype(np.float32)
    elif mode == "bbox_center":
        return mesh.bounding_box.centroid.astype(np.float32)
    elif mode == "bottom_center":
        (mn, mx) = mesh.bounds
        cx, cy, cz = mesh.bounding_box.centroid
        if up_axis=="z": anchor = np.array([cx, cy, mn[2]], np.float32)
        elif up_axis=="y": anchor = np.array([cx, mn[1], cz], np.float32)
        elif up_axis=="x": anchor = np.array([mn[0], cy, cz], np.float32)
        else: raise ValueError("up_axis must be x/y/z")
        return anchor
    else:
        raise ValueError("mode must be vertex_mean/bbox_center/bottom_center")


def recenter_mesh(mesh: trimesh.Trimesh, anchor: np.ndarray) -> trimesh.Trimesh:
    m2 = mesh.copy(); m2.apply_translation(-anchor); return m2

def process_one_obj(src_obj_path: str, dst_obj_path: str, mode: str, up_axis: str) -> Tuple[np.ndarray, Dict]:
    mesh = trimesh.load_mesh(src_obj_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.dump().geometry.values()))
    anchor = compute_anchor_offset(mesh, mode=mode, up_axis=up_axis)  # origin->anchor (local)
    centered = recenter_mesh(mesh, anchor)
    os.makedirs(osp.dirname(dst_obj_path), exist_ok=True)
    centered.export(dst_obj_path)
    copy_mtl_and_textures(src_obj_path, dst_obj_path)
    meta = {
        "src": src_obj_path, "dst": dst_obj_path,
        "mode": mode, "up_axis": up_axis,
        "num_vertices": int(centered.vertices.shape[0]),
        "num_faces": int(centered.faces.shape[0]),
    }
    t_fix_local = anchor.astype(np.float32)
    return t_fix_local, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源根目录，例如 .../dataset/behave/objects")
    ap.add_argument("--dst", required=True, help="目标根目录，例如 .../dataset/behave/objects_centroid")
    ap.add_argument("--mode", default="mesh_centroid", choices=["bbox_center","bottom_center","vertex_mean"])
    ap.add_argument("--up", default="z", choices=["x","y","z"])
    ap.add_argument("--suffix", default="", help="输出文件名追加后缀（可留空）")
    args = ap.parse_args()

    src_root = osp.abspath(args.src)
    dst_root = osp.abspath(args.dst)
    os.makedirs(dst_root, exist_ok=True)

    mapping: Dict[str, Dict] = {}
    rows = []

    # 递归遍历
    obj_list = []
    for root, _, files in os.walk(src_root):
        for fn in files:
            if fn.lower().endswith(".obj"):
                obj_list.append(osp.join(root, fn))
    obj_list.sort()
    print(f"[INFO] Found {len(obj_list)} OBJ files under {src_root}")

    for src_obj in obj_list:
        rel_path = osp.relpath(src_obj, src_root)                   # e.g. objects/backpack/backpack.obj
        rel_dir  = osp.dirname(rel_path)
        base     = osp.splitext(osp.basename(src_obj))[0]           # backpack
        out_name = base + (args.suffix if args.suffix else "") + ".obj"
        dst_obj  = osp.join(dst_root, rel_dir, out_name)

        key = osp.splitext(rel_path)[0]  # 作为字典键，避免重名冲突
        try:
            t_fix_local, meta = process_one_obj(src_obj, dst_obj, args.mode, args.up)
            mapping[key] = {
                "t_fix_local": t_fix_local.tolist(),
                "mode": args.mode, "up_axis": args.up,
                "obj_out_rel": osp.join(rel_dir, out_name).replace("\\","/"),
                **meta
            }
            rows.append([key, float(t_fix_local[0]), float(t_fix_local[1]), float(t_fix_local[2]), args.mode, args.up])
            print(f"[OK] {key}: t_fix_local = {t_fix_local}")
        except Exception as e:
            print(f"[ERR] {key}: {e}")

    # 统一记录
    json_path = osp.join(dst_root, "objects_centroid_map.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {json_path}")

    csv_path = osp.join(dst_root, "objects_centroid_map.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["rel_key","t_fix_x","t_fix_y","t_fix_z","mode","up_axis"]); w.writerows(rows)
    print(f"[SAVE] {csv_path}")
    print("[DONE] All objects processed.")

if __name__ == "__main__":
    main()

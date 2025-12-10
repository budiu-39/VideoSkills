import argparse
import json
import os
import random
import shutil
import time
from glob import glob

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_dict_structure(obj):
    return isinstance(obj, dict)

def is_list_structure(obj):
    return isinstance(obj, list)

def extract_keys(data, keyfield="key_name"):
    """Return (keys, structure_type) where structure_type in {'dict','list'}"""
    if is_dict_structure(data):
        return list(data.keys()), "dict"
    if is_list_structure(data):
        keys = []
        for i, item in enumerate(data):
            if not isinstance(item, dict) or keyfield not in item:
                raise ValueError(
                    f"JSON 看起来是列表结构，但第 {i} 个元素缺少字段 '{keyfield}'。"
                )
            keys.append(item[keyfield])
        return keys, "list"
    raise ValueError("不支持的 JSON 结构：既不是 dict 也不是 list。")

def remove_selected_from_data(data, selected, structure_type, keyfield="key_name"):
    if structure_type == "dict":
        for k in selected:
            data.pop(k, None)
        return data
    else:  # list
        selected_set = set(selected)
        new_list = [item for item in data if item.get(keyfield) not in selected_set]
        return new_list

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def find_file_by_key(src_dir, key, default_ext):
    """优先用 {key}{ext}，找不到则用 glob 兜底匹配 key.* 或 key*"""
    cand1 = os.path.join(src_dir, f"{key}{default_ext}")
    if os.path.isfile(cand1):
        return cand1

    # 尝试严格按“完整文件名 = key”匹配任何扩展
    hits = glob(os.path.join(src_dir, key + ".*"))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # 如果存在多个，优先选常见的几种
        preferred_exts = [".npy", ".npz", ".pkl"]
        for ext in preferred_exts:
            cand = os.path.join(src_dir, key + ext)
            if os.path.isfile(cand):
                return cand
        # 否则返回第一个
        return hits[0]

    # 再宽松一些：有些数据文件名前缀是 key，后面带后缀
    hits = glob(os.path.join(src_dir, key + "*" + default_ext))
    if hits:
        return hits[0]

    return None

def write_keys_txt(keys, out_dir, prefix="selected_keys"):
    ensure_dir(out_dir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(out_dir, f"{prefix}_{ts}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for k in keys:
            f.write(str(k) + "\n")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default='dataset/motion_embeds',  help="embedding JSON 路径")
    ap.add_argument("--src", required=True, help="源目录，例如 dataset/smpl_motion/kungfu")
    ap.add_argument("--num", type=int, default=100, help="抽样数量（默认100）")
    ap.add_argument("--ext", default=".npy", help="数据文件扩展名（默认 .npy）")
    ap.add_argument("--keyfield", default="key_name", help="当 JSON 为列表结构时，key 字段名（默认 key_name）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    subset_name = args.json.split("/")[-1].split(".")[0]
    dst = os.path.join('dataset/smpl_motion', subset_name + "_test")
    txt_export_dir = f'dataset/splits/{subset_name}_test.txt'

    # 1) 读取 JSON 并抽样
    data = load_json(args.json)
    all_keys, structure = extract_keys(data, keyfield=args.keyfield)

    if len(all_keys) == 0:
        raise ValueError("在 JSON 中没有找到任何 key。")

    n_select = min(args.num, len(all_keys))
    selected = random.sample(all_keys, n_select)

    # 2) 导出选中 key 到 txt
    txt_path = write_keys_txt(selected, txt_export_dir, prefix="selected_keys")
    print(f"[INFO] 已导出 {n_select} 个 key 到: {txt_path}")

    # 3) 移动对应文件
    ensure_dir(dst)
    moved_ok, missing = 0, []

    for key in selected:
        src_file = find_file_by_key(args.src, key, args.ext)
        if src_file is None:
            missing.append(key)
            continue
        dst_file = os.path.join(dst, os.path.basename(src_file))
        # 避免覆盖同名文件：若存在则在文件名加时间戳
        if os.path.exists(dst_file):
            base, ext = os.path.splitext(os.path.basename(src_file))
            dst_file = os.path.join(dst, f"{base}_{int(time.time())}{ext}")
        shutil.move(src_file, dst_file)
        moved_ok += 1

    print(f"[INFO] 成功移动 {moved_ok}/{n_select} 个文件到 {dst}")
    if missing:
        print(f"[WARN] 下列 key 未找到对应文件（已从 JSON 中移除，若不希望移除，请回滚备份）：")
        for k in missing:
            print("  -", k)

    new_data = remove_selected_from_data(data, selected, structure, keyfield=args.keyfield)

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False)

    print(f"[INFO] 已从 JSON 删除 {n_select} 个条目并覆盖写回: {args.json}")

if __name__ == "__main__":
    main()

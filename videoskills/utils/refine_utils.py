import os, shutil, tempfile
from typing import List, Iterable

def chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def make_symlink_batch_dir(files: List[str], base_tmp: str) -> str:
    """
    在 base_tmp 下创建一个临时子目录，放入这些 files 的软链接。
    返回该目录路径；调用方用完记得删除（或用 mkdtemp 自动清理策略）。
    """
    os.makedirs(base_tmp, exist_ok=True)
    batch_dir = tempfile.mkdtemp(prefix="refine_batch_", dir=base_tmp)
    for f in files:
        dst = os.path.join(batch_dir, os.path.basename(f))
        try:
            os.symlink(os.path.abspath(f), dst)
        except FileExistsError:
            pass
    return batch_dir

def reset_motion_lib_dir(runner, dir_path):
    """
    假定 runner.reset_motion_lib 接受“目录路径”，并会加载目录中所有 .npy 文件。
    若你的实现是 `reset_motion_lib(file_or_dir)`, 直接传目录即可。
    """
    runner.reset_motion_lib(dir_path)

def build_key_from_path(path: str, root: str) -> str:
    """
    根据 amass_root 下某个文件的 path，生成 key_name_dump。
    兼容 .npz / .npy 等扩展名。
    """
    rel = os.path.relpath(path, root)          # 例如 'CMU/sub1/file_001.npz'
    parts = rel.split(os.sep)                  # ['CMU', 'sub1', 'file_001.npz']
    stem, _ = os.path.splitext(parts[-1])      # 'file_001'
    parts[-1] = stem                           # ['CMU', 'sub1', 'file_001']
    key_name_dump = '0-' + "_".join(parts)     # '0-CMU_sub1_file_001'
    return key_name_dump

def build_key_to_path_index(amass_root: str,
                            exts=(".npy", ".npz")) -> dict:
    """
    扫描 amass_root 下所有文件，建立:
        key_name_dump -> 文件绝对路径
    只保留指定扩展名的文件（默认 .npy & .npz）
    """
    key_to_path = {}

    for dirpath, dirnames, filenames in os.walk(amass_root):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            key = build_key_from_path(full_path, amass_root)
            key_to_path[key] = full_path

    print(f"[INFO] Indexed {len(key_to_path)} motions from {amass_root}")
    return key_to_path
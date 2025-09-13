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

def reset_motion_lib_dir(runner, dir_path: str):
    """
    假定 runner.reset_motion_lib 接受“目录路径”，并会加载目录中所有 .npy 文件。
    若你的实现是 `reset_motion_lib(file_or_dir)`, 直接传目录即可。
    """
    runner.reset_motion_lib(dir_path)
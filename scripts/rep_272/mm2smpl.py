import os
import sys
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
import glob
from joblib import Parallel, delayed
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion

# 环境变量
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制 CPU

sys.path.append(os.getcwd())

from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState

# 进程内缓存
_SKELETON_TREE = None

# 环境变量
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制 CPU

sys.path.append(os.getcwd())

from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion
from scripts.rep_272.d272_to_sim_utils import convert_272_and_pad_to_24

# 进程内缓存
_SKELETON_TREE = None
def get_skeleton_tree(robot_cfg):
    """
    只加载完整的 24 关节骨骼树
    """
    global _SKELETON_TREE
    if _SKELETON_TREE is None:
        _SKELETON_TREE = SkeletonTree.from_mjcf(f"data/robots/smpl/{robot_cfg['model']}_humanoid.xml")
    return _SKELETON_TREE


def process_vae_one(fpath, output_dir, robot_cfg, fps=30):
    try:
        filename = os.path.basename(fpath)
        save_path = osp.join(output_dir, filename)

        if osp.exists(save_path):
            return ("skip", fpath)

        # 1. 加载数据
        data_dict = np.load(fpath, allow_pickle=True).item()
        z_all = data_dict['z']
        recon_all = data_dict['recon']

        # 2. 获取完整骨骼树 (24关节)
        full_tree = get_skeleton_tree(robot_cfg)

        # 3. 转换并填充 (272 -> 24 Joint SkeletonMotion)
        motion_obj = convert_272_and_pad_to_24(
            recon_all,
            full_tree,
            fps=fps
        )

        # 4. 保存
        final_save_dict = {
            'z': z_all,
            'motion': motion_obj.to_dict()
        }

        os.makedirs(osp.dirname(save_path), exist_ok=True)
        np.save(save_path, final_save_dict)

        return ("ok", fpath)

    except Exception as e:
        return ("err", f"{fpath} :: {repr(e)}")

if __name__ == "__main__":
    robot_cfg = {"model": "smpl"}

    input_vae_dir = "dataset/272_rep/AMASS_sim_test_predicted"
    output_physics_dir = "dataset/smpl_motion/AMASS_sim_test_predicted"

    all_files = sorted(glob.glob(osp.join(input_vae_dir, "**", "*_sub*.npy"), recursive=True))
    print(f"Found {len(all_files)} VAE sub-files.")

    n_jobs = 16

    print(f"Starting conversion (Direct 24-Joint Padding) with n_jobs={n_jobs}...")

    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=16)(
        delayed(process_vae_one)(fpath, output_physics_dir, robot_cfg, fps=30)
        for fpath in tqdm(all_files, desc="Converting")
    )

    ok = sum(1 for s, _ in results if s == "ok")
    skip = sum(1 for s, _ in results if s == "skip")
    err_list = [msg for s, msg in results if s == "err"]
    print(f"Done. ok={ok}, skip={skip}, err={len(err_list)}")
    if err_list:
        print("Errors (first 5):")
        for m in err_list[:5]: print(" -", m)
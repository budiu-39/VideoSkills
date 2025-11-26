import os, os.path as osp, glob, json, time, warnings
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import joblib
import torch

from scipy.spatial.transform import Rotation as sRot
from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_parser import SMPL_Parser

# -----------------------------
# 全局只读参数（主进程传给子进程）
# -----------------------------
GLOBAL = {}

def init_worker(smpl_model_dir, robot_model_xml, process_set, amass_occlusion, out_root, src_root):
    """
    每个子进程启动时初始化一次重资源，存到 GLOBAL 里。
    """
    # 限制每个进程的内部并行，避免过度抢占 CPU
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    GLOBAL["skeleton_tree"] = SkeletonTree.from_mjcf(robot_model_xml)
    GLOBAL["smpl_parser"]   = SMPL_Parser(model_path=smpl_model_dir, gender="neutral")
    GLOBAL["process_set"]   = set(process_set)   # e.g. {'BioMotionLab_NTroje'}
    GLOBAL["amass_occ"]     = amass_occlusion    # dict
    GLOBAL["out_root"]      = out_root
    GLOBAL["src_root"]      = src_root

    # 预计算 SMPL->MuJoCo 索引
    GLOBAL["smpl_2_mj"] = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]


def process_one_file(npz_path):
    """
    子进程处理单个 npz；写出 .npy，返回 (fix_key, diff_fix) 或 None
    """
    try:
        skeleton_tree = GLOBAL["skeleton_tree"]
        smpl_parser   = GLOBAL["smpl_parser"]
        process_set   = GLOBAL["process_set"]
        amass_occ     = GLOBAL["amass_occ"]
        out_root      = GLOBAL["out_root"]
        src_root      = GLOBAL["src_root"]
        smpl_2_mj     = GLOBAL["smpl_2_mj"]

        # --------- 根据路径获取 split / 生成 key ---------
        path_parts = npz_path.split("/")
        # 找 AMASS 根所在位置
        amass_index = None
        for i, part in enumerate(path_parts):
            if part.lower().startswith("amass"):
                amass_index = i
                break
        if amass_index is None or amass_index + 1 >= len(path_parts):
            return None  # 跳过非 AMASS 结构

        split_name = path_parts[amass_index + 1]  # train/test/valid（或子库名，按你数据组织）
        key_name_dump = "0-" + "_".join(path_parts[amass_index + 1:]).replace(".npz", "")

        if split_name not in process_set:
            return None

        # --------- 根据 occlusion 信息做 bound ----------
        bound = 0
        if key_name_dump in amass_occ:
            entry = amass_occ[key_name_dump]
            issue = entry.get("issue", "")
            if (issue in ("sitting", "airborne")) and ("idxes" in entry):
                bound = entry["idxes"][0]
                if bound < 10:
                    return None
            else:
                return None

        # --------- 读 npz ---------
        entry_data = dict(np.load(npz_path, allow_pickle=True))
        if "mocap_framerate" not in entry_data:
            return None

        framerate = float(entry_data["mocap_framerate"])
        skip = max(1, int(round(framerate / 30.0)))  # 统一到 30Hz
        root_trans = entry_data["trans"][::skip, :]
        poses_full = entry_data["poses"][::skip, :]

        # 只取前 22 个 body joint（24-2 手，按你需要），这里与你原代码等价构造：
        pose_aa = np.concatenate([poses_full[:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)

        N = pose_aa.shape[0]
        if bound == 0: bound = N
        root_trans = root_trans[:bound]
        pose_aa    = pose_aa[:bound]
        N = pose_aa.shape[0]
        if N < 10:
            return None

        # rotvec→quat & 关节重排
        pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mj]
        pose_quat  = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, -1, 4)

        # 计算地面对齐（fix height）
        beta = np.zeros(16, dtype=np.float32)
        frame_check = min(100, N)
        pose_t = torch.from_numpy(pose_aa[:frame_check])
        beta_t = torch.from_numpy(beta[None, :])
        trans_t = torch.from_numpy(root_trans[:frame_check])

        with torch.no_grad():
            verts, joints = smpl_parser.get_joints_verts(pose_t, beta_t, trans_t)
            offset = joints[:, 0] - trans_t
            feet_z = (verts - offset[:, None])[..., -1]
            diff_fix = feet_z.min().item()

        root_trans_offset = torch.from_numpy(root_trans.copy())
        root_trans_offset[..., -1] -= diff_fix  # 下移

        # 构造 SkeletonState（用全局旋转直立化）
        # 注意：这里直接用 local→global 的 pipeline
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True
        )

        # 直立化（如需要）
        pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) *
                            sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_global),
            root_trans_offset,
            is_local=False
        )

        motion_obj = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30)

        # 保存路径：保持与源结构相同的相对层级
        rel_path = osp.relpath(npz_path, src_root).replace(".npz", ".npy")
        save_path = osp.join(out_root, rel_path)
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        motion_obj.to_file(save_path)

        # 返回 fix height 记录（可选）
        dataset = path_parts[-3] if len(path_parts) >= 3 else "unknown"
        subset  = path_parts[-2] if len(path_parts) >= 2 else "unknown"
        filename = path_parts[-1].replace(".npz", "")
        key_str = f"{dataset}-{subset}-{filename}"
        return (key_str, float(round(diff_fix, 5)))

    except Exception as e:
        # 这里不抛出，让主进程继续；你也可以返回错误信息
        return None

def main_parallel(
    amass_root: str,
    out_root: str = "dataset/smpl_motion/AMASS_new",
    smpl_model_dir: str = "data/SMPL/smpl",
    robot_model_xml: str = "data/robots/smpl/smpl_humanoid.xml",
    process_set = ('BioMotionLab_NTroje',),    # 你的 amass_splits['train'] 列表
    amass_occ_pkl: str = "data/amass_copycat_occlusion_v3.pkl",
    max_workers: int = None
):
    # 收集全部 npz
    all_npz = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    print(f"[Info] Found {len(all_npz)} npz files")

    amass_occlusion = joblib.load(amass_occ_pkl)

    # 选择进程数
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"[Info] Using {max_workers} workers")

    # Windows/macOS 下必须 spawn；Linux 默认 fork 就行（集群上也建议手动设置一次）
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    # 初始化子进程的全局资源
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(smpl_model_dir, robot_model_xml, process_set, amass_occlusion, out_root, amass_root)
    ) as ex:

        futures = [ex.submit(process_one_file, p) for p in all_npz]

        fix_pairs = []
        for fut in tqdm(as_completed(futures), total=len(futures), ncols=100, desc="Processing AMASS"):
            res = fut.result()
            if res is not None:
                fix_pairs.append(res)

    # 汇总保存
    fix_dict = {k: v for k, v in fix_pairs}
    os.makedirs(out_root, exist_ok=True)
    pkl_path = osp.join(out_root, "fixed_height_keys.pkl")
    joblib.dump(fix_dict, pkl_path)
    print(f"[saved] {pkl_path} ({len(fix_dict)} entries)")
    print("Done.")

if __name__ == "__main__":
    # 例子：只处理 'BioMotionLab_NTroje'（按你原来的选择）
    main_parallel(
        amass_root="/mnt/lustre/work/ponsmoll/pba936/AMASS",
        out_root="dataset/smpl_motion/AMASS_new",
        smpl_model_dir="data/SMPL/smpl",
        robot_model_xml="data/robots/smpl/smpl_humanoid.xml",
        process_set=('BioMotionLab_NTroje',),
        amass_occ_pkl="data/amass_copycat_occlusion_v3.pkl",
        max_workers=7                                       # 按机器核数调整
    )

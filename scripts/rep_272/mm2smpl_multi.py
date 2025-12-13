import os
import sys
import os.path as osp
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
from tqdm import tqdm
import glob
from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from scripts.utils.ms_utils import recover_from_local_rotation, smpl85_2_smpl322
import torch
# import mujoco  # 并行时不建议打开 viewer
from scripts.preprocess.padding import pad_skeleton_state

from joblib import Parallel, delayed

# --------- 你已有的函数保留（略），例如 rot6d_to_rotmat / pose_272_to_smpl 等 ---------

# 进程内缓存（懒加载）
_PARSER = None
_SKELETON_TREE = None

def get_parser_and_tree(robot_cfg):
    global _PARSER, _SKELETON_TREE
    if _PARSER is None:
        _PARSER = SMPL_Parser(model_path='data/SMPL/smpl', gender="neutral")
    if _SKELETON_TREE is None:
        _SKELETON_TREE = SkeletonTree.from_mjcf(f"data/robots/smpl/{robot_cfg['model']}_humanoid.xml")
    return _PARSER, _SKELETON_TREE

def apply_cam2world_rotvec_trans(rotvec, trans, R3x3):
    r_new = sRot.from_matrix(R3x3) * sRot.from_rotvec(rotvec)
    t_new = (R3x3 @ trans.T).T
    return r_new.as_rotvec().astype(np.float32), t_new.astype(np.float32)

def pose_272_to_smpl(data_272):
    smpl_85_data = recover_from_local_rotation(data_272, 22)  # get the 85-dim smpl data
    if len(smpl_85_data.shape) == 3:
        smpl_85_data = np.squeeze(smpl_85_data, axis=0)
    pose = smpl85_2_smpl322(smpl_85_data)
    assert pose.shape[1] == 322
    use_flame = (pose.shape[1] == 322)
    root_and_body = pose[:, :66].reshape(-1, 22, 3)
    trans = pose[:, 309:312] if use_flame else pose[:, 159:162]
    return trans.reshape(-1, 3), root_and_body

def process_one(fpath, folder_path, output_dir, robot_cfg, fps=45, padding=0):
    try:
        rel_path = os.path.relpath(fpath, folder_path)
        rel_noext = os.path.splitext(rel_path)[0]
        motion_key = rel_noext.replace(os.sep, "-")
        save_path = osp.join(output_dir, f"{motion_key}.npy")

        # 已存在就跳过（可选）
        if osp.exists(save_path):
            return ("skip", fpath)

        motion = np.load(fpath, allow_pickle=True)
        N = motion.shape[0]
        root_trans, pose_aa = pose_272_to_smpl(motion)

        pose_aa_smpl = np.zeros((N, 24, 3), dtype=pose_aa.dtype)
        pose_aa_smpl[:, :22, :] = pose_aa

        smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
        R_cam2world = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], dtype=np.float32)

        smpl_parser, skeleton_tree = get_parser_and_tree(robot_cfg)
        root_trans_offset = root_trans + skeleton_tree.local_translation[0].numpy()
        pose_aa_smpl[:, 0], root_trans_offset = apply_cam2world_rotvec_trans(pose_aa_smpl[:, 0], root_trans_offset, R_cam2world)
        root_trans_offset = torch.from_numpy(root_trans_offset)
        pose_aa_mj = pose_aa_smpl[:, smpl_2_mujoco]

        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

        beta = np.zeros(16, dtype=np.float32)

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True
        )

        # 落地高度修正
        with torch.no_grad():
            frame_check = min(100, N)
            pose_t = torch.from_numpy(pose_aa_smpl[:frame_check])
            beta_t = torch.from_numpy(beta[None, ...])
            trans_t = root_trans_offset[:frame_check]
            verts, joints = smpl_parser.get_joints_verts(pose_t, beta_t, trans_t)
            offset = joints[:, 0] - trans_t
            feet_z = (verts - offset[:, None])[..., -1]
            diff_fix = feet_z.min().item()
            root_trans_offset[..., -1] -= diff_fix

        # 直立修正
        pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) *
                            sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_global),
            root_trans_offset,
            is_local=False
        )

        # new_sk_state = pad_skeleton_state(new_sk_state, padding)
        motion_obj = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)

        os.makedirs(osp.dirname(save_path), exist_ok=True)
        motion_obj.to_file(save_path)

        # 如果需要额外保存轨迹，可解除下注释
        # motion_traj = {
        #     'root_trans_offset': new_sk_state.root_translation.numpy(),
        #     'root_rotation': new_sk_state.global_root_rotation.numpy(),
        #     'dof': sRot.from_quat(new_sk_state.local_rotation[:,1:].reshape(-1, 4)).as_rotvec().reshape(N + padding, -1, 3)
        # }

        return ("ok", fpath)
    except Exception as e:
        return ("err", f"{fpath} :: {repr(e)}")

if __name__ == "__main__":
    upright_start = True
    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": upright_start,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smpl",
    }

    subset_list = ["idea400", "music", "perform", "haa500", "game_motion"]
    subset_dir = "MotionMillion/motion_272rpr/MotionUnion"
    base_output_dir = "dataset/smpl_motion/MotionMillion"

    all_jobs = []
    for subset in subset_list:
        folder_path = os.path.join(subset_dir, subset)
        output_dir = os.path.join(base_output_dir, subset)
        npy_files = sorted(glob.glob(os.path.join(folder_path, '**', '*.npy'), recursive=True))
        for fpath in npy_files:
            all_jobs.append( (fpath, folder_path, output_dir) )

    # 并行执行
    n_jobs = max(1, (os.cpu_count() or 4) - 1)  # 留1核给系统
    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1, prefer="processes")(
        delayed(process_one)(fpath, folder_path, output_dir, robot_cfg, 45, 0)
        for (fpath, folder_path, output_dir) in tqdm(all_jobs, desc="Dispatch jobs")
    )

    # 简单汇总
    ok = sum(1 for s,_ in results if s=="ok")
    skip = sum(1 for s,_ in results if s=="skip")
    err_list = [msg for s,msg in results if s=="err"]
    print(f"Done. ok={ok}, skip={skip}, err={len(err_list)}")
    if err_list:
        print("Errors (first 10):")
        for m in err_list[:10]:
            print(" -", m)

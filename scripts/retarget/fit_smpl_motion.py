import os
import os.path as osp
import glob
import time
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from tqdm import tqdm
import joblib
from scipy.spatial.transform import Rotation as sRot

# --- project imports (use your existing modules) ---
sys.path.append(os.getcwd())
from videoskills.envs.g1.g1_config import G1RoughCfg
from videoskills.utils.torch_humanoid_batch import Humanoid_Batch
from videoskills.utils.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from smpl_sim.utils.smoothing_utils import gaussian_filter_1d_batch
from smpl_sim.utils import torch_utils

# helpers you already had in the original script
from scipy.spatial.transform import Rotation as sRot
import torch
from torch.autograd import Variable
import mujoco
import imageio

# If you already have these utilities elsewhere, feel free to import them instead.
# They are kept here for self-containment.
@dataclass
class ExtendCfgEntry:
    joint_name: str
    parent_name: str
    pos: List[float]
    rot: List[float]

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def rotate(pose, trans, rotate_matrix=[[1.0, 0.0, 0.0], [0.0, 0.0, 1], [0.0, -1.0, 0.0]]):
    pose[:, :3] = torch.tensor(
        (sRot.from_matrix(rotate_matrix) * sRot.from_rotvec(pose)).as_rotvec().reshape(-1, 3),
        dtype=torch.float32,
    )
    trans = (torch.tensor(rotate_matrix, dtype=torch.float32) @ trans.transpose(1, 0)).transpose(1, 0)
    return pose, trans


def load_amass_data(data_path):
    try:
        with np.load(data_path, allow_pickle=True) as data:
            entry_data = dict(data)
    except Exception as e:
        print(f"Failed to load npz file: {data_path} | Error: {e}")
        return None

    if "mocap_framerate" not in entry_data:
        return None

    framerate = entry_data["mocap_framerate"]
    root_trans = entry_data["trans"]
    pose_aa = np.concatenate([entry_data["poses"][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
    betas = entry_data["betas"]
    gender = entry_data["gender"]
    return {"pose_aa": pose_aa, "gender": gender, "trans": root_trans, "betas": betas, "fps": framerate}


def load_GVHMR_data(data_path):
    entry_data = torch.load(data_path, map_location="cpu")["smpl_params_global"]
    framerate = 30
    root_trans = entry_data["transl"]
    pose_aa = entry_data["body_pose"]
    zeros_tensor = torch.zeros((root_trans.shape[0], 6), device=pose_aa.device, dtype=pose_aa.dtype)
    global_orient = entry_data["global_orient"]
    pose_aa = torch.cat([global_orient, pose_aa, zeros_tensor], dim=-1)
    betas = entry_data["betas"]
    gender = "neutral"
    return {"pose_aa": pose_aa, "gender": gender, "trans": root_trans, "betas": betas, "fps": framerate}


# ---------------- AMASS split helpers -----------------
AMASS_SPLITS = {
    "valid": ["HumanEva", "MPI_HDM05", "SFU", "MPI_mosh"],
    "test": ["Transitions_mocap", "SSM_synced"],
    "train": [
        "CMU",
        "MPI_Limits",
        "TotalCapture",
        "KIT",
        "EKUT",
        "TCD_handMocap",
        "BMLhandball",
        "DanceDB",
        "ACCAD",
        "BMLmovi",
        "BioMotionLab_NTroje",
        "Eyes_Japan_Dataset",
        "DFaust_67",
    ],
}


def _match_amass_dataset(key_name: str, all_datasets: set) -> Optional[str]:
    suffix = key_name.split("-")[1]
    for dataset in all_datasets:
        if suffix.startswith(dataset + "_"):
            return dataset
    return None


def _collect_amass_keys(amass_root: str) -> Dict[str, str]:
    all_npzs = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.rstrip("/").split("/"))
    return {"0-" + "_".join(p.split("/")[split_len:]).replace(".npz", ""): p for p in all_npzs}


def _collect_gvhmr_keys(gvhmr_root: str) -> Dict[str, str]:
    pt_files = glob.glob(osp.join(gvhmr_root, "*", "*.pt"))
    return {osp.basename(osp.dirname(f)): f for f in pt_files}


# --------------- Core retarget (no saving inside) ---------------

def _process_motion(keys: List[str], key2file: Dict[str, str], cfg, skeleton_tree: SkeletonTree):
    device = torch.device("cpu")
    humanoid_fk = Humanoid_Batch(cfg)
    num_augment_joint = len(getattr(cfg, "extend_config", []) or [])

    # joint mapping comes from cfg.joint_matches (already defined in your config)
    robot_joint_names_augment = humanoid_fk.body_names_augment
    robot_joint_pick = [i[0] for i in cfg.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.joint_matches]
    robot_joint_pick_idx = [robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    shape_new, scale = joblib.load(f"data/retarget/{cfg.humanoid_type}/shape_optimized_v1.pkl")

    motions: Dict[str, SkeletonMotion] = {}
    export_payloads: Dict[str, dict] = {}

    pbar = tqdm(keys, position=0, leave=True)
    for data_key in pbar:
        # --- load source ---
        if cfg.input_motion_type == "AMASS":
            dat = load_amass_data(key2file[data_key])
            if dat is None:
                continue
            skip = int(max(1, dat["fps"] // 30))
            trans = torch.from_numpy(dat["trans"][::skip])
            pose_aa = torch.from_numpy(dat["pose_aa"][::skip]).float()
        else:  # GVHMR
            dat = load_GVHMR_data(key2file[data_key])
            if dat is None:
                continue
            pose_aa = dat["pose_aa"]
            trans = dat["trans"]
            # align conventions
            pose_aa[:, :3], trans = rotate(pose_aa[:, :3], trans.squeeze())
            pose_aa[:, :3], trans = rotate(pose_aa[:, :3], trans.squeeze(), [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

        N = trans.shape[0]
        if N < 10:
            print("too short")
            continue

        # smpl -> joints
        with torch.no_grad():
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa, shape_new, trans)
            root_pos = joints[:, 0:1]
            joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos
        joints[..., 2] -= verts[0, :, 2].min().item()

        offset = joints[:, 0] - trans
        root_trans_offset = (trans + offset).clone()

        # heading-only root rot
        gt_root_rot_quat = torch.from_numpy(
            (sRot.from_rotvec(pose_aa[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()
        ).float()
        gt_root_rot = torch.from_numpy(
            sRot.from_quat(torch_utils.calc_heading_quat(gt_root_rot_quat)).as_rotvec()
        ).float()

        dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))
        dof_pos_new = Variable(dof_pos.clone(), requires_grad=True)
        root_rot_new = Variable(gt_root_rot.clone(), requires_grad=True)
        root_pos_offset = Variable(torch.zeros(1, 3), requires_grad=True)
        optimizer = torch.optim.Adam([dof_pos_new, root_rot_new, root_pos_offset], lr=0.02)

        kernel_size, sigma = 5, 0.75

        for it in range(cfg.fitting_iterations):
            pose_aa_new = torch.cat(
                [
                    root_rot_new[None, :, None],
                    humanoid_fk.dof_axis * dof_pos_new,
                    torch.zeros((1, N, num_augment_joint, 3)).to(device),
                ],
                dim=2,
            )

            fk_ret = humanoid_fk.fk_batch(pose_aa_new, root_trans_offset[None, ] + root_pos_offset)

            if num_augment_joint > 0:
                diff = fk_ret.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            else:
                diff = fk_ret.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]

            loss = diff.norm(dim=-1).mean() + 0.01 * torch.mean(torch.square(dof_pos_new))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])
            dof_pos_new.data = (
                gaussian_filter_1d_batch(dof_pos_new.squeeze().transpose(1, 0)[None, ], kernel_size, sigma)
                .transpose(2, 1)
                .contiguous()[..., None]
            )

            pbar.set_description_str(f"{data_key}-Iter: {it}\t{loss.item() * 1000:.3f}")

        # compose final motion
        dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])
        pose_aa_new = torch.cat(
            [
                root_rot_new[None, :, None],
                humanoid_fk.dof_axis * dof_pos_new,
                torch.zeros((1, N, num_augment_joint, 3)).to(device),
            ],
            dim=2,
        )
        root_trans_offset_dump = (root_trans_offset + root_pos_offset).clone()

        # light height fix
        N_clip = min(10, N)
        pose_clip = pose_aa_new[:, :N_clip]
        trans_clip = root_trans_offset_dump[None, :N_clip]
        z_min_list = []
        for i in range(N_clip):
            mesh = humanoid_fk.mesh_fk(pose_clip[:, i : i + 1].detach(), trans_clip[:, i : i + 1].detach())
            z_min_list.append(np.asarray(mesh.vertices)[..., 2].min())
        height_diff = float(np.mean(z_min_list))
        root_trans_offset_dump[..., 2] -= height_diff

        # build SkeletonMotion
        actuated = pose_aa_new.squeeze()[:, humanoid_fk.actuated_joints_idx, :]
        local_quat = (
            sRot.from_rotvec(actuated.detach().reshape(-1, 3)).as_quat().reshape(actuated.shape[0], -1, 4)
        )
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(local_quat).float(),
            root_trans_offset_dump.float().squeeze().detach(),
            is_local=True,
        )
        motion_obj = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30)
        motions[data_key] = motion_obj

        data_dump = {
            "root_trans_offset": root_trans_offset_dump.squeeze().detach().numpy(),  # (N,3)
            "pose_aa": pose_aa_new.squeeze().detach().numpy(),  # (N, J, 3)；渲染器用 [:,0]
            "dof": dof_pos_new.squeeze().detach().numpy(),  # (N, DoF, 1)
            "fps": 30,
        }
        export_payloads[data_key] = data_dump

    return motions, export_payloads


# ---------------------------- Public APIs ----------------------------

def retarget_from_amass(input_dir: str, output_dir: str, *, render: bool = False, process_split: str = "train", num_jobs: Optional[int] = None) -> None:
    """
    AMASS (.npz) -> SkeletonMotion .npy files.

    Parameters
    ----------
    input_dir : str
        Root of AMASS tree.
    output_dir : str
        Where .npy motions (and optional videos) will be saved.
    render : bool
        If True, additionally render .mp4 using MuJoCo.
    process_split : str
        'train' | 'valid' | 'test'.
    num_jobs : Optional[int]
        Multiprocessing workers; defaults to cfg.num_jobs.
    """
    robot_cfg = G1RoughCfg()
    robot_cfg.asset.file = robot_cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=os.getcwd())
    cfg = robot_cfg.retarget
    cfg.file = robot_cfg.asset.file
    cfg.output_dir = output_dir
    cfg.extend_config = [ExtendCfgEntry(**d) for d in (cfg.extend_config or [])]

    cfg.input_motion_type = "AMASS"
    cfg.amass_root = input_dir
    cfg.process_split = process_split

    skeleton_tree = SkeletonTree.from_mjcf("data/robots/g1/g1_29dof.xml")

    # collect keys & filter splits and occlusions
    key2file = _collect_amass_keys(cfg.amass_root)
    all_datasets = set(sum(AMASS_SPLITS.values(), []))

    # optional occlusion filter
    occl_path = "data/amass_copycat_occlusion_v3.pkl"
    amass_occlusion = joblib.load(occl_path) if osp.exists(occl_path) else {}

    keys_to_keep = []
    for key in list(key2file.keys()):
        dataset = _match_amass_dataset(key, all_datasets)
        if dataset not in AMASS_SPLITS[cfg.process_split]:
            continue
        # occlusion filtering (same logic as before)
        if key in amass_occlusion:
            issue = amass_occlusion[key]["issue"]
            if (issue in {"sitting", "airborne"}) and ("idxes" in amass_occlusion[key]):
                bound = amass_occlusion[key]["idxes"][0]
                if bound < 10:
                    continue
            else:
                continue
        keys_to_keep.append(key)

    key_list = sorted(keys_to_keep)

    # multiprocessing
    n_workers = num_jobs or getattr(cfg, "num_jobs", 8)
    if len(key_list) == 0:
        print("No AMASS files found to process.")
        return

    # run (single or multi)
    if n_workers <= 1 or len(key_list) < 2:
        motions, export_payloads = _process_motion(key_list, key2file, cfg, skeleton_tree)
    else:
        from multiprocessing import Pool

        chunk = int(np.ceil(len(key_list) / n_workers))
        chunks = [key_list[i : i + chunk] for i in range(0, len(key_list), chunk)]
        args = [(c, key2file, cfg, skeleton_tree) for c in chunks]
        motions = {}
        with Pool(n_workers) as pool:
            for part in pool.starmap(_process_motion, args):
                motions.update(part)

    # saving
    for key, motion in motions.items():
        rel_path = osp.relpath(key2file[key], cfg.amass_root)
        save_path = osp.join(output_dir, f"AMASS_{cfg.process_split}", rel_path).replace(".npz", ".npy")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        motion.to_file(save_path)

    # optional rendering
    if render:
        try:
            from mujoco import mjtCamera
            from mujoco import MjModel, MjData, Renderer
            from scripts.render.mujoco_render import export_mujoco_video  # if you have a local util
        except Exception:
            export_mujoco_video = None
        if export_mujoco_video is None:
            print("Render=True but no export_mujoco_video available. Skipping.")
        else:
            xml_path = "data/robots/g1/g1_29dof.xml"
            vid_dir = osp.join(output_dir, "videos")
            os.makedirs(vid_dir, exist_ok=True)
            for key, motion in motions.items():
                rel = osp.relpath(key2file[key], cfg.amass_root).replace(".npz", ".mp4")
                out_mp4 = osp.join(vid_dir, rel)
                os.makedirs(osp.dirname(out_mp4), exist_ok=True)
                export_mujoco_video(export_payloads[key], out_mp4, 30, xml_path)



def retarget_from_gvhmr(input_dir: str, output_dir: str, *, render_dir: str = None, num_jobs: Optional[int] = None) -> None:
    """
    GVHMR (*.pt) -> SkeletonMotion .npy files.

    Parameters
    ----------
    input_dir : str
        Root dir with layout: <input_dir>/<video_name>/*.pt
    output_dir : str
        Where .npy motions (and optional videos) will be saved.
    render : bool
        If True, also render .mp4 using MuJoCo.
    num_jobs : Optional[int]
        Multiprocessing workers; defaults to cfg.num_jobs.
    """
    robot_cfg = G1RoughCfg()
    robot_cfg.asset.file = robot_cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=os.getcwd())
    cfg = robot_cfg.retarget
    cfg.file = robot_cfg.asset.file
    cfg.output_dir = output_dir
    cfg.extend_config = [ExtendCfgEntry(**d) for d in (cfg.extend_config or [])]

    cfg.input_motion_type = "GVHMR"
    cfg.gvhmr_path = input_dir

    skeleton_tree = SkeletonTree.from_mjcf("data/robots/g1/g1_29dof.xml")

    key2file = _collect_gvhmr_keys(cfg.gvhmr_path)
    key_list = sorted(list(key2file.keys()))
    if len(key_list) == 0:
        print("No GVHMR .pt files found.")
        return

    n_workers = num_jobs or getattr(cfg, "num_jobs", 8)

    if n_workers <= 1 or len(key_list) < 2:
        motions, export_payloads = _process_motion(key_list, key2file, cfg, skeleton_tree)

    else:
        from multiprocessing import Pool

        chunk = int(np.ceil(len(key_list) / n_workers))
        chunks = [key_list[i : i + chunk] for i in range(0, len(key_list), chunk)]
        args = [(c, key2file, cfg, skeleton_tree) for c in chunks]
        motions = {}
        with Pool(n_workers) as pool:
            for part in pool.starmap(_process_motion, args):
                motions.update(part)

    # saving
    out_root = osp.join(output_dir)
    os.makedirs(out_root, exist_ok=True)
    for key, motion in motions.items():
        save_path = osp.join(out_root, f"{key}.npy")
        motion.to_file(save_path)

    # optional rendering
    if render_dir is not None:
        try:
            from scripts.render.mujoco_render import export_mujoco_video  # if you have a local util
        except Exception:
            export_mujoco_video = None
        if export_mujoco_video is None:
            print("Render=True but no export_mujoco_video available. Skipping.")
        else:
            xml_path = "data/robots/g1/g1_29dof.xml"
            vid_dir = osp.join(render_dir)
            os.makedirs(vid_dir, exist_ok=True)
            for key, motion in motions.items():
                out_mp4 = osp.join(vid_dir, f"{key}.mp4")
                export_mujoco_video(export_payloads[key], out_mp4, 30, xml_path)


# ---------------------------- Example usage ----------------------------
if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    # Example: AMASS
    # convert_amass_to_motion(
    #     input_dir="/path/to/AMASS",
    #     output_dir="output/g1_motion",
    #     render=False,
    #     process_split="train",
    #     num_jobs=8,
    # )

    # Example: GVHMR
    retarget_from_gvhmr(
        input_dir="/home/miku/Documents/VideoSkills/output/GVHMR_output/kungfu_4",
        output_dir="output/g1_motion/kungfu_4",
        render="output/g1_motion/kungfu_4_rendered",
        num_jobs=1,
    )

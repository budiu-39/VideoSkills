import os
import numpy as np
from videoskills.utils import get_args, task_registry
from videoskills.utils.convergence_monitor import ConvergenceMonitor
import wandb
from videoskills.utils.helpers import print_and_save_cfg, class_to_dict
import glob
from scripts.render.constrast_render import render
from smplx import SMPL
from tqdm import tqdm
import subprocess
from scripts.preprocess.convert_gvhmr_isaac import process_folder
from pathlib import Path
import os, shutil, tempfile
from typing import List, Iterable, Tuple
from utils.refine_utils import make_symlink_batch_dir, reset_motion_lib_dir, chunked
import torch
from scripts.retarget.fit_smpl_motion import retarget_from_gvhmr
from scripts.render.vis_motion_rollout import vis_mujoco_offscreen_render



class MotionRefinePipeline:
    def __init__(self, env_cfg, train_cfg, args, log_dir: str):
        self.args = args
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.log_dir = log_dir

        # 收集所有 motion（每个 .npy 即一个 motion）
        self.motion_files = glob.glob(os.path.join(env_cfg.motion.file, f"**/*.npy"), recursive=True)
        if not self.motion_files:
            raise RuntimeError("No motion files found.")

        # 用第一个文件初始化 env/runner
        self.env_cfg.motion.file = self.motion_files
        self.env_cfg.init_state.type = 'hybrid'
        self.env, self.env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=self.env_cfg)
        self.runner, self.train_cfg = task_registry.make_alg_runner(
            env=self.env, name=args.task, args=args, train_cfg=self.train_cfg, log_dir=self.log_dir
        )

        # 训练/监控参数
        self.target_eval_success = getattr(getattr(self.train_cfg, "refine", {}), "target_eval_success", 0.95)
        self.hard_cap   = getattr(getattr(self.train_cfg, "refine", {}), "max_refine_epochs", 20)
        self.interval   = getattr(getattr(self.train_cfg, "refine", {}), "refine_interval", 20)
        self.et_window  = getattr(getattr(self.train_cfg, "refine", {}), "et_window", 20)
        # self.max_it     = self.train_cfg.runner.max_iterations

        # 监控器
        self.monitor = ConvergenceMonitor(N=20, alpha=1, cv_thr=0.08, trend_scale=1e-3, patience=3)

        # W&B
        if self.args.use_wandb and not self.args.dev:
            os.makedirs(os.path.join(self.log_dir, "wandb"), exist_ok=True)
            run_name = self.train_cfg.runner.run_name
            wandb.init(
                project=self.args.wandb_project, name=run_name, dir=self.log_dir,
                config={**vars(self.args), **class_to_dict(self.train_cfg), **class_to_dict(self.env_cfg)}
            )

        self.runner.load()
        self.tmp_root = os.path.join(self.log_dir, "_tmp_refine_batches")
        os.makedirs(self.tmp_root, exist_ok=True)

    # ------------------------ 外部主入口 ------------------------ #
    def run(self, batch_size_easy: int = 18):
        easy_files, hard_files = self.pre_eval_all()

        # 分批 refine（easy 走批、hard 逐个）
        group_size = min(batch_size_easy, self.runner.env.num_envs)
        self.refine_easy(easy_files, group_size)
        self.refine_hard(hard_files)

        if wandb.run is not None:
            wandb.log({"Refine/easy_count": len(easy_files), "Refine/hard_count": len(hard_files)})

        if os.path.isdir(self.tmp_root) and not os.listdir(self.tmp_root):
            os.rmdir(self.tmp_root)

    # ------------------------ 预评估/分类 ------------------------ #

    def pre_eval_all(self) -> Tuple[List[str], List[str]]:
        """
        先跑一次 runner.eval() 得到 success_keys / failed_keys，
        再把 key 通过 `key.split('-')[-1]` -> 文件名 stem -> 映射回 *.npy 的绝对路径。
        返回: (easy_files, hard_files)，均为 *.npy 的路径列表。
        """
        # 1) 跑评估（确保当前 MotionLib 已包含你要评估的 motions）
        eval_out = self.runner.eval(motion_ids=None)

        # 2) 读取 keys（去重保序）
        def unique_keep_order(xs: List[str]) -> List[str]:
            return list(dict.fromkeys(xs)) if xs else []

        success_keys = unique_keep_order(eval_out.get("success_keys", []))
        failed_keys = unique_keep_order(eval_out.get("failed_keys", []))

        # 3) 预构建 stem -> fullpath 映射，stem 就是文件名不含后缀
        #    例如 /.../Aerial_Kick_..._clip3.npy  ->  stem="Aerial_Kick_..._clip3"
        stem2path = {}
        for p in self.motion_files:
            stem2path[Path(p).stem] = p

        # 4) 把 key 映射回 *.npy
        def keys_to_files(keys: List[str]) -> List[str]:
            out, miss = [], []
            for k in keys:
                stem = k.split('-')[-1]  # 取最后一段作为文件名 stem
                fp = stem2path.get(stem, None)  # 在已知的 motion_files 里找
                if fp is not None:
                    out.append(fp)
                else:
                    miss.append(k)
            if miss:
                print(f"[WARN] {len(miss)} keys not matched to any *.npy. Examples: {miss[:5]}")
            return out

        easy_files = keys_to_files(success_keys)
        hard_files = keys_to_files(failed_keys)

        return easy_files, hard_files

    # ------------------------ refine（easy 批处理） ------------------------ #
    def refine_easy(self, files: List[str], group_size: int):
        if not files:
            return
        print(f"[Refine] Easy motions: {len(files)}; batching {group_size} per refine run.")
        for batch_files in chunked(files, group_size):
            batch_dir = make_symlink_batch_dir(batch_files, base_tmp=self.tmp_root)
            try:
                self.runner.load(load_iteration=False)
                reset_motion_lib_dir(self.runner, batch_dir)  # 目录中每个 .npy 即一个 motion
                self.training_loop()
            finally:
                shutil.rmtree(batch_dir, ignore_errors=True)

    # ------------------------ refine（hard 单个） ------------------------ #
    def refine_hard(self, files: List[str]):
        if not files:
            return
        print(f"[Refine] Hard motions: {len(files)}; refining one-by-one.")
        for file in files:
            batch_dir = make_symlink_batch_dir([file], base_tmp=self.tmp_root)
            try:
                self.runner.load(load_iteration=False)
                reset_motion_lib_dir(self.runner, batch_dir)
                self.training_loop()
            finally:
                shutil.rmtree(batch_dir, ignore_errors=True)

    # ------------------------ 可选：训练/收敛监控 ------------------------ #
    def training_loop(self):

        it = 0
        while it * self.interval <= self.hard_cap:
            it += 1
            self.runner.learn(num_learning_iterations=self.interval, init_at_random_ep_len=False)
            self.runner.env.early_termination_distance = (torch.tensor(self.runner.env.cfg.early_termination.distance
                                                               , device=self.runner.env.device) + 0.25 + 0.01 * it) ** 2
            # 最近窗口的 early-termination 代理成功率
            success_proxy = 1 - self.runner.mean_et_rate(k=self.et_window)
            et_vals = self.runner.pop_recent_ET_rate(k=self.et_window)
            success_series = 1.0 - np.array(et_vals, dtype=float)

            converged = self.monitor.update(success_series, success_rate=success_proxy)

            if wandb.run is not None:
                to_log = dict(self.monitor.last_diag)
                to_log.update({'CM/success_proxy': success_proxy,
                               'Refine/et_window': self.et_window,
                               'Refine/interval': self.interval})
                wandb.log(to_log, step=self.runner.current_learning_iteration)

            if converged:
                eval_out = self.runner.rollout()
                if isinstance(eval_out, dict) and 'per_motion_success_rate' in eval_out:
                    if float(min(eval_out['per_motion_success_rate'].values())) >= self.target_eval_success:
                        break

            if it * self.interval >= self.hard_cap:
                eval_out = self.runner.rollout()
                print(f"[EarlyStop] Hit hard cap {self.hard_cap}.")
                break


def config(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    log_dir = print_and_save_cfg(env_cfg, train_cfg, filename="config.yaml")

    return log_dir, env_cfg, train_cfg


if __name__ == '__main__':
    args = get_args()
    if args.headless:
        if os.environ.get("DISPLAY", "") == "":
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            os.environ["PYGLET_HEADLESS"] = "True"
    log_dir, env_cfg, train_cfg = config(args)

    # 1.GVHMR
    # folder = args.folder
    # folder = Path(folder)
    # gvhmr_root = (Path(__file__).resolve().parents[1] / 'GVHMR').resolve()
    # if args.gvhmr_output is not None:
    #     gvhmr_output_dir = args.gvhmr_output
    # else:
    #     gvhmr_output_dir = os.path.join(log_dir, 'gvhmr_output')
    # mp4_paths = sorted(
    #     [p.resolve() for p in folder.glob("*.mp4")] +
    #     [p.resolve() for p in folder.glob("*.MP4")]
    # )
    # print(f"Found {len(mp4_paths)} .mp4 files in {folder}")
    # for mp4_path in tqdm(mp4_paths):
    #     command = ["python", "tools/demo/demo.py", "--video", str(mp4_path)]
    #     command += ["--output_root", gvhmr_output_dir]
    #     if args.static_cam:
    #         command += ["-s"]
    #     print(f"Running: {' '.join(command)}")
    #     try:
    #         subprocess.run(command, env=dict(os.environ), cwd=str(gvhmr_root), check=True)
    #     except subprocess.CalledProcessError as e:
    #         print(f"[WARN] GVHMR failed on {mp4_path} (rc={e.returncode}). Skip this clip.")
    #         continue
    #
    # # 2.preprocess (smpl) or retarget (gvhmr)
    #
    # if args.task == 'smpl':
    #     motion_data_dir = os.path.join(log_dir, 'preprocessed_data')
    #     result = process_folder(gvhmr_output_dir, motion_data_dir)
    #     env_cfg.motion.file = motion_data_dir
    # elif args.task == 'g1':
    #     motion_data_dir = os.path.join(log_dir, 'retarget_result')
    #     retarget_result_render_dir = os.path.join(motion_data_dir, 'rendered_videos')
    #     retarget_from_gvhmr(
    #         input_dir=gvhmr_output_dir,
    #         output_dir=motion_data_dir,
    #         render_dir=retarget_result_render_dir,
    #         num_jobs=1,
    #     )
    #     env_cfg.motion.file = motion_data_dir
    # else:
    #     raise NotImplementedError(f"Task {args.task} not implemented for refine.py")
    #
    # # 3.refinement
    # pipeline = MotionRefinePipeline(env_cfg, train_cfg, args, log_dir)
    # pipeline.run(batch_size_easy=18)

    # 4.rendering
    render_failed = False
    if args.task == 'smpl':
        render(f'{log_dir}/rollouts/succeed', f'{log_dir}/renders/succeed'
               , True, gvhmr_output_dir)
        if render_failed:
            render(f'{log_dir}/rollouts/failed', f'{log_dir}/renders/failed'
                   , True, gvhmr_output_dir)
    elif args.task == 'g1':
        pkl_files = [f for f in os.listdir(gvhmr_output_dir) if f.endswith('.pkl')]
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {gvhmr_output_dir}")
        for pkl_file in pkl_files:
            file_path = os.path.join(gvhmr_output_dir, pkl_file)
            motion_data = joblib.load(file_path)
            # motion_traj = next(iter(motion_data.values()))
            print(f"Rendering motion from: {pkl_file}")
            vis_mujoco_offscreen_render(motion_data, motion_key=pkl_file, humanoid_model_file=train_cfg.asset.file,
                                        out_dir=output_dir)

#

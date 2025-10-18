import os, sys
import numpy as np
from videoskills.utils import get_args, task_registry
from videoskills.utils.convergence_monitor import ConvergenceMonitor
import wandb
from videoskills.utils.helpers import print_and_save_cfg, class_to_dict
import glob
from smplx import SMPL
from tqdm import tqdm
import subprocess
from scripts.preprocess.gvhmr2smpl import process_folder
from pathlib import Path
import os, shutil, tempfile
from typing import List, Iterable, Tuple
from utils.refine_utils import make_symlink_batch_dir, reset_motion_lib_dir, chunked
import torch
from videoskills import LEGGED_GYM_ROOT_DIR

class MotionRefinePipeline:
    def __init__(self, env_cfg, train_cfg, args, log_dir: str):
        self.args = args
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.log_dir = log_dir

        # 收集所有 motion（每个 .npy 即一个 motion）
        if args.motion_file is not None:
            motion_file = 'dataset/smpl_motion/' + args.motion_file
        else:
            motion_file = env_cfg.motion.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.motion_files = glob.glob(os.path.join(motion_file, f"**/*.npy"), recursive=True)
        if not self.motion_files:
            raise RuntimeError("No motion files found.")

        # 用第一个文件初始化 env/runner
        self.env_cfg.motion.file = self.motion_files
        self.env_cfg.init_state.type = 'hybrid'
        self.env_cfg.motion.rotate_motion = False
        self.env, self.env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=self.env_cfg)
        self.runner, self.train_cfg = task_registry.make_alg_runner(
            env=self.env, name=args.task, args=args, train_cfg=self.train_cfg, log_dir=self.log_dir
        )

        # 训练/监控参数
        self.target_eval_success = getattr(getattr(self.train_cfg, "refine", {}), "target_eval_success", 0.95)
        self.hard_cap   = getattr(getattr(self.train_cfg, "refine", {}), "max_refine_epochs", 400)
        self.interval   = getattr(getattr(self.train_cfg, "refine", {}), "refine_interval", 20)
        self.et_window  = getattr(getattr(self.train_cfg, "refine", {}), "et_window", 20)
        # self.max_it     = self.train_cfg.runner.max_iterations

        if self.args.dev:
            self.hard_cap = 20

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

        self.runner.save_interval = 1000000 # refine 不存 checkpoint
        self.runner.load()
        self.tmp_root = os.path.join(self.log_dir, "_tmp_refine_batches")
        os.makedirs(self.tmp_root, exist_ok=True)

    # ------------------------ 外部主入口 ------------------------ #
    def run(self, batch_size_easy: int = 18):
        easy_files, hard_files = self.pre_eval_all()
        sequential_mode = (getattr(self.args, "sequential", False)
                           or (hasattr(self.args, "accelerate") and not self.args.accelerate))
        group_size = min(batch_size_easy, self.runner.env.num_envs)
        self.plan_groups_and_order(easy_files, hard_files, group_size, sequential_mode=sequential_mode)

        # 分批 refine（easy 走批、hard 逐个）
        if sequential_mode:
            files = easy_files + hard_files
            print(f"[Refine] Sequential mode ON (no acceleration). Total motions: {len(files)}")
            self.refine_hard(files)
        else:
            print(f"[Refine] Accelerated mode (default). Easy batch size = {group_size}")
            self.refine_easy(easy_files, group_size)
            self.refine_hard(hard_files)

        if wandb.run is not None:
            wandb.log({"Refine/easy_count": len(easy_files), "Refine/hard_count": len(hard_files)})

    # ------------------------ 预评估/分类 ------------------------ #

    def pre_eval_all(self):
        """跑一次 eval，得到 easy/hard 的 *.npy 列表（顺序保持与 eval 输出一致）"""
        eval_out = self.runner.eval(motion_ids=None)

        def unique_keep_order(xs):
            return list(dict.fromkeys(xs)) if xs else []

        success_keys = unique_keep_order(eval_out.get("success_keys", []))
        failed_keys = unique_keep_order(eval_out.get("failed_keys", []))

        # 关键：直接从 MotionLib 拿 key <-> filepath（零歧义，不用 split('-')）
        lib = self.runner.env._motion_lib
        key2path = {k: f for k, f in zip(lib._motion_keys, lib._motion_files)}

        miss = [k for k in success_keys + failed_keys if k not in key2path]
        if miss:
            print(f"[WARN] {len(miss)} eval keys不在当前MotionLib里（可能目录不同步）: {miss[:5]}")

        easy_files = [key2path[k] for k in success_keys if k in key2path]
        hard_files = [key2path[k] for k in failed_keys if k in key2path]

        print(f"[PreEval] easy={len(easy_files)}, hard={len(hard_files)}")
        return easy_files, hard_files

    # --- 在类里新增：列出所有 motion、分组，以及 refine 执行顺序（保存到log_dir） ---
    def plan_groups_and_order(self, easy_files, hard_files, group_size, sequential_mode=False):
        """
        输出三部分：
          1) 全量 motions 列表
          2) easy/hard 分组（文件级）
          3) refine 执行顺序（批次/逐个）
        同时保存到 {log_dir}/refine_plan_{timestamp}.json 和 .csv
        """
        import json, time, csv
        ts = time.strftime("%Y%m%d-%H%M%S")

        # 1) 全量 motions（来自初始化收集）
        all_motions = list(dict.fromkeys(self.motion_files))

        # 2) 分组
        groups = {
            "easy": easy_files,
            "hard": hard_files
        }

        # 3) 执行顺序：顺序模式=全部逐个；加速模式=easy按批次→hard逐个
        schedule = []
        if sequential_mode:
            for i, f in enumerate(easy_files + hard_files, 1):
                schedule.append({"step": i, "type": "single", "files": [f]})
        else:
            # easy 批次
            if easy_files:
                for i in range(0, len(easy_files), group_size):
                    batch = easy_files[i:i + group_size]
                    schedule.append({"step": len(schedule) + 1, "type": "batch", "files": batch})
            # hard 逐个
            for f in hard_files:
                schedule.append({"step": len(schedule) + 1, "type": "single", "files": [f]})

        # 打印到控制台（简洁版）
        print("\n========== [Plan] Motions / Groups / Refine Order ==========")
        print(f"Total motions: {len(all_motions)}")
        print(f"Easy: {len(groups['easy'])} | Hard: {len(groups['hard'])}")
        print("Refine order (前10步预览):")
        for item in schedule[:10]:
            kind = "BATCH" if item["type"] == "batch" else "SINGLE"
            print(
                f"  Step {item['step']:>3} [{kind}]  x{len(item['files'])}  e.g., {os.path.basename(item['files'][0])}")
        print("...（完整顺序已保存到文件）")

        # 保存 JSON
        plan_json = {
            "total": len(all_motions),
            "all_motions": all_motions,
            "groups": groups,
            "schedule": schedule,
            "mode": "sequential" if sequential_mode else "accelerated",
            "group_size_easy": None if sequential_mode else group_size,
        }
        json_path = os.path.join(self.log_dir, f"refine_plan_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(plan_json, f, indent=2, ensure_ascii=False)

        # 保存 CSV（按执行顺序展开）
        csv_path = os.path.join(self.log_dir, f"refine_plan_{ts}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "type", "file"])
            for item in schedule:
                for f in item["files"]:
                    w.writerow([item["step"], item["type"], f])

        # W&B 记录（可选）
        if getattr(self.args, "use_wandb", False) and not getattr(self.args, "dev", False):
            import wandb
            wandb.log({
                "Plan/total_motions": len(all_motions),
                "Plan/easy_count": len(groups["easy"]),
                "Plan/hard_count": len(groups["hard"]),
                "Plan/steps": len(schedule)
            })

        print(f"[Plan] Saved: {json_path}")
        print(f"[Plan] Saved: {csv_path}")
        return schedule, json_path, csv_path

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
                        print(f"[Converged+Eval OK] min success={min(eval_out['per_motion_success_rate'].values()):.3f} ≥ {self.target_eval_success:.3f}")
                        break
                    else:
                        print(f"[Converged+Eval FAIL] min success={min(eval_out['per_motion_success_rate'].values()):.3f} < {self.target_eval_success:.3f}")

            if it * self.interval >= self.hard_cap:
                eval_out = self.runner.rollout()
                print(f"[EarlyStop] Hit hard cap {self.hard_cap}.")
                break


def config(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    log_dir = print_and_save_cfg(env_cfg, train_cfg, filename="config.yaml")

    return log_dir, env_cfg, train_cfg

def render(log_dir, render_failed=False):
    rollout_dir = os.path.join(log_dir, 'refine_results')
    rollout_success_dir = os.path.join(rollout_dir, 'succeed')
    # gvhmr_output_dir = os.path.join(log_dir, 'gvhmr_results')
    if args.task == 'smpl':
        from scripts.render.constrast_render import render as smpl_render
        smpl_render(rollout_success_dir, f'{log_dir}/render_results/succeed', True)
        if render_failed:
            rollout_failed_dir = os.path.join(log_dir, 'refine_results/failed')
            smpl_render(rollout_failed_dir, f'{log_dir}/render_results/failed', True)
    # elif args.task == 'g1' and os.environ.get("DISPLAY", "") != "":
    #     from scripts.render.vis_motion_rollout import mujoco_render as g1_render
    #     humanoid_model_file = 'data/robots/g1/g1_29dof.xml'
    #     g1_render(rollout_success_dir, f'{log_dir}/render_results/succeed', True, gvhmr_output_dir, \
    #                   humanoid_model_file)  #, retarget_result_render_dir)
    #     if render_failed:
    #         rollout_failed_dir = os.path.join(log_dir, 'refine_results/failed')
    #         g1_render(rollout_failed_dir, f'{log_dir}/render_results/failed', True,
    #                       gvhmr_output_dir, humanoid_model_file) #, retarget_result_render_dir)

if __name__ == '__main__':
    args = get_args()
    args.resume = True
    if args.headless:
        if os.environ.get("DISPLAY", "") == "":
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            os.environ["PYGLET_HEADLESS"] = "True"

    log_dir, env_cfg, train_cfg = config(args)
    if args.render_run:
        render(args.render_run)
        sys.exit(0)

    # 1.preprocess (smpl) or retarget (gvhmr)
    # motion_data_dir = os.path.join(log_dir, 'preprocessed_data')
    # result = process_folder(gvhmr_output_dir, motion_data_dir)

    # 2.refinement
    pipeline = MotionRefinePipeline(env_cfg, train_cfg, args, log_dir)
    pipeline.run(batch_size_easy=18)


    # 3.rendering
    render_failed = True
    render(log_dir, render_failed)




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


def config(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    log_dir = print_and_save_cfg(env_cfg, train_cfg, filename="config.yaml")

    return log_dir, env_cfg, train_cfg

def train(env_cfg, train_cfg, args):

    motion_files = glob.glob(os.path.join(*env_cfg.motion.file.split('/'), f"**/*.npy"), recursive=True)
    env_cfg.motion.file = motion_files[0]
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg,
                                                      log_dir=log_dir)

    monitor = ConvergenceMonitor(
        N=20,
        alpha=1,
        cv_thr=0.08,
        trend_scale=1e-3,
        patience=3,
        # target_success=getattr(getattr(train_cfg, "refine", {}), "success_proxy", None)  # e.g., 0.9
    )

    if args.use_wandb and not args.dev:
        os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)
        run_name = train_cfg.runner.run_name
        wandb.init(project=args.wandb_project, name=run_name,
                   dir=log_dir,
                   config={**vars(args), **class_to_dict(train_cfg), **class_to_dict(env_cfg)})

    target_eval_success = getattr(getattr(train_cfg, "refine", {}), "target_eval_success", 0.95)
    hard_cap   = getattr(getattr(train_cfg, "refine", {}), "max_refine_epochs", 20)
    interval   = getattr(getattr(train_cfg, "refine", {}), "refine_interval", 20)
    et_window  = getattr(getattr(train_cfg, "refine", {}), "et_window", 20)
    max_it     = train_cfg.runner.max_iterations
    runner.load()
    #TODO: introduce a loop here, to load the motion and set the dataset
    total = len(motion_files)
    for i, file in enumerate(motion_files, start=1):
        runner.load(load_iteration=False)
        runner.reset_motion_lib(file)
        for it in range(0, max_it + 1, interval):
            runner.learn(num_learning_iterations=interval, init_at_random_ep_len=False)

            # 最近这段训练中新结束的 episodic returns
            if hasattr(runner, 'pop_recent_episode_rewards'):
                recent_rewards = runner.pop_recent_mean_rewards()
            else:
                recent_rewards = list(getattr(runner, 'rewbuffer', []))[-monitor.N:]

            success_proxy = 1 - runner.mean_et_rate(k=et_window)
            et_vals = runner.pop_recent_ET_rate(k=et_window)
            success_series = 1.0 - np.array(et_vals, dtype=float)

            # 调用收敛监测
            converged = monitor.update(success_series, success_rate=success_proxy)

            # 统一把诊断项写入 wandb（包括 slope/cv/是否通过等）
            if wandb.run is not None:
                to_log = dict(monitor.last_diag)
                to_log.update({
                    'CM/success_proxy': success_proxy,
                    'Refine/et_window': et_window,
                    'Refine/interval': interval,
                })
                wandb.log(to_log, step=runner.current_learning_iteration)

            if converged:
                runner.rollout = True
                eval_out = runner.eval()  # 你的 eval 里建议返回 dict（success_rate 等）
                if isinstance(eval_out, dict) and 'success_rate' in eval_out:
                    # if wandb.run is not None:
                    #     wandb.log({f'Eval/{k}': v for k, v in eval_out.items() if isinstance(v, (int, float))}, step=it)
                    if float(eval_out['success_rate']) >= target_eval_success:
                        print(f"[Converged+Eval OK] success={eval_out['success_rate']:.3f} ≥ {target_eval_success:.3f}")
                        # runner.save()
                        print(f"[Progress] Done {i}/{total} files, remaining {total - i}.")
                        break
                    else:
                        print(f"[Converged+Eval FAIL] success={eval_out['success_rate']:.3f} < {target_eval_success:.3f}")

            if it >= hard_cap:
                runner.rollout = True
                eval_out = runner.eval()
                # runner.save()
                print(f"[EarlyStop] Hit hard cap {hard_cap}.")
                print(f"[Progress] Done {i}/{total} files, remaining {total - i}.")
                break


if __name__ == '__main__':
    args = get_args()
    if args.headless:
        if os.environ.get("DISPLAY", "") == "":
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            os.environ["PYGLET_HEADLESS"] = "True"
    log_dir, env_cfg, train_cfg = config(args)

    # GVHMR
    folder = args.folder
    folder = Path(folder)
    gvhmr_root = (Path(__file__).resolve().parents[1] / 'GVHMR').resolve()
    gvhmr_output_dir = os.path.join(log_dir, 'gvhmr_output')
    mp4_paths = sorted(list(folder.glob("*.mp4")) + list(folder.glob("*.MP4")))
    print(f"Found {len(mp4_paths)} .mp4 files in {folder}")
    for mp4_path in tqdm(mp4_paths):
        command = ["python", "tools/demo/demo.py", "--video", str(mp4_path)]
        command += ["--output_root", gvhmr_output_dir]
        if args.static_cam:
            command += ["-s"]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, env=dict(os.environ),cwd=str(gvhmr_root), check=True)

    # preprocess or retarget
    motion_data_dir = os.path.join(log_dir, 'motion_data')
    if args.task == 'smpl':
        result = process_folder(gvhmr_output_dir, motion_data_dir)

    env_cfg.motion.file = motion_data_dir
    # TODO: test
    env_cfg.motion.file = 'logs/smpl_ppo/refinement_folder_136_resume_Sep12_17-50-41/motion_data/gvhmr_output'

    # refinement
    train(env_cfg, train_cfg, args)

    # rendering  # 需要把 render 改成在服务器上也能跑
    if args.task == 'smpl':
        render(f'{log_dir}/rollouts/succeed', f'{log_dir}/renders/succeed', True)


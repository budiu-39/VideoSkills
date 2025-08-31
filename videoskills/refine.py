import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from videoskills.utils import get_args, task_registry
from videoskills.utils.convergence_monitor import ConvergenceMonitor
import wandb
from videoskills.utils.helpers import print_and_save_cfg, class_to_dict
sys.path.append(os.getcwd())

import sys, os, inspect
print("argv[0] :", sys.argv[0])
print("sys.path[0] :", sys.path[0])

def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    log_dir = print_and_save_cfg(env_cfg, train_cfg, filename="config.yaml")
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg,
                                                          log_dir=log_dir)

    monitor = ConvergenceMonitor(
        N=100,
        alpha=0.15,
        cv_thr=0.08,
        trend_scale=1e-3,
        patience=3,
        target_success=getattr(getattr(train_cfg, "refine", {}), "target_success", None)  # 如 0.9
    )


    if args.use_wandb and not args.dev:
        os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)
        run_name = train_cfg.runner.run_name
        wandb.init(project=args.wandb_project, name=run_name,
                   dir=log_dir,
                   config={**vars(args), **class_to_dict(train_cfg), ** class_to_dict(env_cfg)})

    pre_cnt = 0
    target_success = getattr(getattr(train_cfg, "refine", {}), "target_success", 1.0)  # 自定阈值
    hard_cap = getattr(getattr(train_cfg, "refine", {}), "max_refine_epochs", 500)
    eval_interval = getattr(getattr(train_cfg, "refine", {}), "eval_interval", 50)
    max_it = train_cfg.runner.max_iterations

    for it in range(0, max_it + 1, eval_interval):
        # 1) 训练
        runner.learn(num_learning_iterations=eval_interval, init_at_random_ep_len=True)

        # 2) 预检测：连续3次稳定就触发 eval
        recent_rewards = runner.rewbuffer
        if monitor.update(recent_rewards):
            eval_out = runner.eval()

        # 3) 连续3次稳定 → 正式评估
        if pre_cnt >= 3:
             # 你已修改为返回 dict
            sr = float(eval_out.get("success_rate", 0.0))

            # 4) 如果 eval 成功（达阈值），执行 rollout；否则清零计数，继续训练
            if sr >= target_success:
                print(f"[Gate→Eval OK] success_rate={sr:.3f} ≥ {target_success:.3f} → rollout")
                do_rollout(runner, seconds=30, normalize_obs=True, record=True)  # 自行决定是否停止或继续
                break  # 若 rollout 后要结束训练；若想继续训练可去掉这行
            else:
                print(f"[Gate→Eval FAIL] success_rate={sr:.3f} < {target_success:.3f} → continue training")
                pre_cnt = 0  # 失败重置，避免频繁触发 eval

        # 5) 硬上限
        if it >= hard_cap:
            print(f"[EarlyStop] Hit hard cap {hard_cap}.")
            break


if __name__ == '__main__':
    args = get_args()
    train(args)

import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from videoskills.utils import get_args, task_registry
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

    if args.use_wandb and not args.dev:
        os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)
        run_name = train_cfg.runner.run_name
        wandb.init(project=args.wandb_project, name=run_name,
                   dir=log_dir,
                   config={**vars(args), **class_to_dict(train_cfg), ** class_to_dict(env_cfg)})

    for it in range(0, train_cfg.runner.max_iterations + 1, train_cfg.runner.eval_interval):
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.eval_interval, init_at_random_ep_len=True)
        ppo_runner.eval(current_motion_id)

if __name__ == '__main__':
    args = get_args()
    train(args)

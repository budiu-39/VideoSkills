import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from videoskills.envs import *
from videoskills.utils import get_args, task_registry
import wandb
from utils.helpers import class_to_dict

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg, log_dir = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    train_cfg_dict = class_to_dict(train_cfg)
    env_cfg_dict = class_to_dict(env_cfg)
    if args.use_wandb and not args.dev:
        os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)
        run_name = train_cfg.runner.run_name
        wandb.init(project=args.wandb_project, name=run_name,
                   dir=log_dir,
                   config={**vars(args), **train_cfg_dict, **env_cfg_dict}, sync_tensorboard=True)

    for it in range(0, train_cfg.runner.max_iterations + 1, train_cfg.runner.eval_interval):
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.eval_interval, init_at_random_ep_len=True)
        ppo_runner.eval()

if __name__ == '__main__':
    args = get_args()
    train(args)

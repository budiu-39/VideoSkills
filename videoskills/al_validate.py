import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from videoskills.utils import get_args, task_registry
import wandb
from videoskills.utils.helpers import print_and_save_cfg, class_to_dict
from utils.refine_utils import make_symlink_batch_dir, reset_motion_lib_dir, chunked
from videoskills.utils.helpers import parse_motion_file_path
from videoskills import LEGGED_GYM_ROOT_DIR
from videoskills.al.active_learner import ActiveLearner
sys.path.append(os.getcwd())
import torch
import json

import sys, os, inspect
print("argv[0] :", sys.argv[0])
print("sys.path[0] :", sys.path[0])

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    env_cfg.motion.file = parse_motion_file_path(env_cfg, train_cfg, only_failed_key=False)
    log_dir = print_and_save_cfg(env_cfg, train_cfg, filename="config.yaml")
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg,
                                                          log_dir=log_dir)

    if args.load_motion_sampling_state:
        motionlib_state_file = os.path.join(LEGGED_GYM_ROOT_DIR,'logs', train_cfg.runner.experiment_name,
                                               train_cfg.runner.load_run, "motion_sampling_state.pkl")
        env._motion_lib.load_sampling_state(motionlib_state_file)

    if args.use_wandb and not args.dev:
        os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)
        run_name = train_cfg.runner.run_name
        wandb.init(project=args.wandb_project, name=run_name,
                   dir=log_dir,
                   config={**vars(args), **class_to_dict(train_cfg), ** class_to_dict(env_cfg)})
    if args.dev:
        train_cfg.runner.eval_interval = 10

    samples_json_file = "dataset/motion_embeds/kungfu.json"
    with open(samples_json_file, "r", encoding="utf-8") as f:
        motion_embeds = json.load(f)
    reference_json_file = "dataset/motion_embeds/AMASS.json"
    with open(reference_json_file, "r", encoding="utf-8") as f:
        ref_motion_embeds = json.load(f)

    train_dir = "dataset/smpl_motion/MotionMillion/kungfu_clean_train"
    test_dir = "dataset/smpl_motion/MotionX++/kungfu_clean_test"
    gt_dir = "dataset/smpl_motion/MotionX++/kungfu_clean_test"
    ppo_runner.env._load_gt_motion(gt_dir)

    for it in range(0, train_cfg.runner.max_iterations + 1, train_cfg.runner.eval_interval):
        # if it > 0:
        reset_motion_lib_dir(ppo_runner, train_dir)
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.eval_interval, init_at_random_ep_len=True)
        # ppo_runner.eval(log=False)
        reset_motion_lib_dir(ppo_runner, test_dir)
        ppo_runner.eval()




if __name__ == '__main__':
    args = get_args()
    train(args)

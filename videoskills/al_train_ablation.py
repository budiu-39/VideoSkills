import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from videoskills.utils import get_args, task_registry
import wandb
from videoskills.utils.helpers import print_and_save_cfg, class_to_dict
from utils.refine_utils import make_symlink_batch_dir, reset_motion_lib_dir, chunked, build_key_to_path_index
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

    train_keys_txt = "../MotionStreamer/data/AMASS_new/active_subset_train_kmeans/picked_names_origin.txt"
    test_keys_txt = "../MotionStreamer/data/AMASS_new/active_subset_test/test_names.txt"
    motion_root = "dataset/smpl_motion/AMASS_train_fixed_height"

    key_to_path = build_key_to_path_index(
        amass_root=motion_root,
        exts=(".npy",)  # 如果你训练用的是 npy，就只保留 npy
    )

    train_motion_paths = []
    with open(train_keys_txt, "r") as f:
        for line in f:
            key = line.strip()
            train_motion_paths.append(key_to_path[key])

    test_motion_paths = []
    with open(test_keys_txt, "r") as f:
        for line in f:
            key = line.strip()
            test_motion_paths.append(key_to_path[key])


    for it in range(0, train_cfg.runner.max_iterations + 1, train_cfg.runner.eval_interval):
        # for train_batch_dir in subset:
        #     print(f"loading {train_batch_dir}")
        #     train_batch_dir = os.path.join(subset_file, train_batch_dir)
        # reset_motion_lib_dir(ppo_runner, env_cfg.motion.file)
        reset_motion_lib_dir(ppo_runner, train_motion_paths)
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.eval_interval, init_at_random_ep_len=True)
        # if ppo_runner.env.cfg.early_termination.distance[0] < 0.69:
        #     ppo_runner.env.early_termination_distance = (torch.tensor(ppo_runner.env.cfg.early_termination.distance
        #                                                              , device=ppo_runner.env.device) + 0.25/5) ** 2
        result = ppo_runner.eval(log=False)
        print('Evaluation result: ', result)

        success_keys = result.get("success_keys", [])
        failed_keys = result.get("failed_keys", [])

        with open(f"{ppo_runner.log_dir}/failed_keys_it{it}_train.txt", "w", encoding="utf-8") as f:
            for item in failed_keys:
                key, frame = item
                f.write(f"{key},{frame}\n")  # 写入 Key,Frame


        with open(f"{ppo_runner.log_dir}/success_keys_it{it}_train.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, success_keys)))

        print(f"Saved {len(success_keys)} success keys and {len(failed_keys)} failed keys to TXT files.")
        reset_motion_lib_dir(ppo_runner, test_motion_paths)
        result = ppo_runner.eval()
        print('Evaluation result: ', result)

        success_keys = result.get("success_keys", [])
        failed_keys = result.get("failed_keys", [])

        with open(f"{ppo_runner.log_dir}/failed_keys_it{it}_test.txt", "w", encoding="utf-8") as f:
            for item in failed_keys:
                key, frame = item
                f.write(f"{key},{frame}\n")  # 写入 Key,Frame


        with open(f"{ppo_runner.log_dir}/success_keys_it{it}_test.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, success_keys)))

        print(f"Saved {len(success_keys)} success keys and {len(failed_keys)} failed keys to TXT files.")




if __name__ == '__main__':
    args = get_args()
    train(args)

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
    al = ActiveLearner(motion_embeds, ref_motion_embeds)
    selected_keys_batch = []

    train_batch_dir = "/home/miku/Documents/VideoSkills/dataset/smpl_motion/control1_amass_al"
    test_batch_dir = "/home/miku/Documents/VideoSkills/dataset/smpl_motion/control1_amass_al"

    for it in range(0, train_cfg.runner.max_iterations + 1, train_cfg.runner.eval_interval):
        selected_keys, _, _ = al.select_by_reference_density(n_select = 300)
        print(al.summary())
        motion_files = [os.path.join(train_batch_dir, f"{k}.npy") for k in selected_keys]
        selected_success_keys, _ = al.random_select(mode='success', n_select = 100)
        if len(selected_success_keys) > 0:
            for k in selected_success_keys:
                motion_files.append(os.path.join(train_batch_dir, f"{k}.npy"))
        reset_motion_lib_dir(ppo_runner, motion_files)
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.eval_interval, init_at_random_ep_len=True)
        # if ppo_runner.env.cfg.early_termination.distance[0] < 0.69:
        #     ppo_runner.env.early_termination_distance = (torch.tensor(ppo_runner.env.cfg.early_termination.distance
        #                                                              , device=ppo_runner.env.device) + 0.25/5) ** 2
        result = ppo_runner.eval(log=False)
        al.add_results(result['success_keys'], result['failed_keys'])
        reset_motion_lib_dir(ppo_runner, test_batch_dir)
        ppo_runner.eval()

        with open(f"{log_dir}/selected_keys_iteration_{it}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, selected_keys)))




if __name__ == '__main__':
    args = get_args()
    train(args)
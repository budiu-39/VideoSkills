
import os

from videoskills.envs import *
from videoskills.utils import  get_args, export_policy_as_jit, task_registry, Logger
from videoskills.utils.helpers import parse_motion_file_path
from utils.refine_utils import reset_motion_lib_dir, build_key_to_path_index

import numpy as np
import torch


def eval(args):

    # env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    env_cfg.motion.file = parse_motion_file_path(env_cfg, train_cfg, only_failed_key=False)
    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 4)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # env_cfg.env.test = True
    env_cfg.env.eval_mode = True # set to a large number for testing

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)

    ### 临时更改 motion lib
    # train_keys_txt = "logs/smpl_ppo/amass_rollout/failed_keys.txt"
    # motion_root = "dataset/smpl_motion/AMASS_train_fixed_height"
    # key_to_path = build_key_to_path_index(
    #     amass_root=motion_root,
    #     exts=(".npy",)  # 如果你训练用的是 npy，就只保留 npy
    # )
    #
    # train_motion_paths = []
    # with open(train_keys_txt, "r") as f:
    #     for line in f:
    #         key = line.strip()
    #         train_motion_paths.append(key_to_path[key])
    #
    # reset_motion_lib_dir(ppo_runner, train_motion_paths)


    result = ppo_runner.eval(rollout=True)
    print('Evaluation result: ', result)

    success_keys = result.get("success_keys", [])
    failed_keys = result.get("failed_keys", [])

    with open(f"{ppo_runner.log_dir}/failed_keys.txt", "w", encoding="utf-8") as f:
        for item in failed_keys:
            # 兼容性处理：防止 item 只是 key 字符串
            if isinstance(item, (list, tuple)):
                key, frame = item
                f.write(f"{key},{frame}\n") # 写入 Key,Frame
            else:
                f.write(f"{item}\n")

    with open(f"{ppo_runner.log_dir}/success_keys.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, success_keys)))

    print(f"Saved {len(success_keys)} success keys and {len(failed_keys)} failed keys to TXT files.")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    eval(args)

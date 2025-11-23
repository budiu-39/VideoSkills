
import os

from videoskills.envs import *
from videoskills.utils import  get_args, export_policy_as_jit, task_registry, Logger
from videoskills.utils.helpers import parse_motion_file_path
from rsl_rl.algorithms.distill_dagger_z import (
    rollout_dagger, train_step, beta_schedule,
    build_student, load_teacher, build_env_and_cfg, DAggerCfg,
    build_decoder_and_prior
)
from rsl_rl.network.episode_encoder import EpisodeEncoder
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
    device = args.rl_device
    device = torch.device(device)

    # env_cfg.env.test = True
    env_cfg.env.eval_mode = True # set to a large number for testing

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    student = build_student(env, train_cfg, device)
    decoder, prior = build_decoder_and_prior(train_cfg, device)
    env.set_z_prior(prior)
    env.set_z_decoder(decoder)

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    ckpt = torch.load(ppo_runner.resume_path)
    prior.load_state_dict(ckpt["prior_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    # student.load_state_dict(ckpt["model_state_dict"])
    # policy = ppo_runner.get_inference_policy(device=env.device)

    # ppo_runner.alg.actor_critic.load_state_dict(student, strict=False)
    decoder.eval()  # 主要是为了 rms
    result = ppo_runner.eval()
    print('Evaluation result: ', result)

    success_keys = result.get("success_keys", [])
    failed_keys = result.get("failed_keys", [])

    with open(f"{ppo_runner.log_dir}/success_keys.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, success_keys)))
    with open(f"{ppo_runner.log_dir}/failed_keys.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, failed_keys)))

    print(f"Saved {len(success_keys)} success keys and {len(failed_keys)} failed keys to TXT files.")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    eval(args)

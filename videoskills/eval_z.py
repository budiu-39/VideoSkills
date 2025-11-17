
import os

from videoskills.envs import *
from videoskills.utils import  get_args, export_policy_as_jit, task_registry, Logger
from videoskills.utils.helpers import parse_motion_file_path
from rsl_rl.algorithms.distill_dagger_window_z import (
    rollout_dagger, train_step, beta_schedule, ReplayBuf,
    build_student, load_teacher, build_env_and_cfg, DAggerCfg,
    EpisodeCtxBuf  # 若在此文件内
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

    # env_cfg.env.test = True
    env_cfg.env.eval_mode = True # set to a large number for testing

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    @torch.no_grad()
    def z_provider(env, env_ids):
        enc.eval()
        ctx, pstats, mids, mask = env.build_context_tensor(env_ids)
        ctx_keys = ctx_buf.add(ctx, pstats, mids, mask)
        z, mu_q, lv_q = enc(ctx, mask=mask)
        # 维护 EMA 原型
        for i, mid in enumerate(mids):
            z_i = z[i].detach()
            if mid not in proto_ema:
                proto_ema[mid] = z_i
            else:
                proto_ema[mid] = 0.98 * proto_ema[mid] + 0.02 * z_i
        aux = {"ctx_keys": ctx_keys}
        return z, aux

    env.set_z_provider(z_provider)

    # encoder
    enc = EpisodeEncoder(in_channels=train_cfg.policy.context_dim, d_model=256, d_z=128).to('cuda')  # C_ctx=上下文通道数
    ctx_buf = EpisodeCtxBuf()
    proto_ema = {}
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    log_root = os.path.dirname(ppo_runner.log_dir)
    path = os.path.join(log_root, args.load_run)
    ckpt = torch.load(path)
    enc.load_state_dict(ckpt["encoder_state_dict"])
    # policy = ppo_runner.get_inference_policy(device=env.device)
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

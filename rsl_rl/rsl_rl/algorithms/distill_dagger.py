# videoskills/distill_dagger_phc_to_transformer.py
import os, time, argparse, math, random
from dataclasses import dataclass
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import num_obs_dm

from videoskills.utils import task_registry, get_args as get_train_args
from videoskills.utils.helpers import print_and_save_cfg, class_to_dict, parse_motion_file_path, dict_to_class
from rsl_rl.storage.rollout_buffer import ReplayBuf

# A100 友好设置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# 导入策略
from rsl_rl.modules import ActorCritic                     # 老师：PHC/MLP
from rsl_rl.modules.actor_critic_attention import ActorCritic_Attention  # 学生：Transformer


@dataclass
class DAggerCfg:
    # 交互
    num_envs: int = 512
    steps_per_env: int = 32         # 每次 DAgger 迭代 rollout 步数
    max_iters: int = 1000           # DAgger 迭代总数

    # β 混合策略（执行动作时：a = beta*teacher + (1-beta)*student）
    beta_start: float = 1.0
    beta_end: float = 0.1
    beta_decay: str = "linear"      # "linear" 或 "exp"
    beta_exp_k: float = 0.002       # exp: beta = beta_end + (beta_start-beta_end)*exp(-k*t)

    # 数据集与优化
    replay_capacity: int = 2_000_00      # 样本上限（T*B 级别）
    batch_size: int = 8192             # 每次优化的样本数（越大越稳定，但显存占用高）
    epochs_per_iter: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas = (0.9, 0.95)

    # 损失
    kl_coef: float = 1.0
    mse_coef: float = 1.0
    value_coef: float = 0.0

    # AMP
    use_mixed_precision: bool = True
    amp_dtype = torch.bfloat16

    # 日志/保存
    log_interval: int = 10
    save_interval: int = 100


def build_env_and_cfg(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    env_cfg.motion.file = parse_motion_file_path(env_cfg, train_cfg, only_failed_key=False)
    log_dir = print_and_save_cfg(env_cfg, train_cfg, filename="config_dagger.yaml")
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    return env, env_cfg, train_cfg, log_dir


def load_teacher(train_cfg, ckpt_path, env, device):

    policy_cfg = class_to_dict(train_cfg['policy'])

    num_actor_obs  = env.num_obs - 138
    num_critic_obs = env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs - 138
    num_actions    = env.num_actions

    teacher = ActorCritic(
        policy_cfg['actor_input_dim'],
        policy_cfg['critic_input_dim'],
        num_actions,

        **policy_cfg
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    teacher.load_state_dict(state, strict=False)
    teacher.eval()

    if hasattr(teacher, "set_update_rms"):
        teacher.set_update_rms(False)
    for p in teacher.parameters():
        p.requires_grad = False

    teacher.obs_dim = num_actor_obs
    teacher.critic_obs_dim = num_critic_obs
    return teacher


def build_student(env, train_cfg, device):
    policy_cfg = class_to_dict(train_cfg.policy)
    policy_cfg.setdefault('n_heads', 4)
    policy_cfg.setdefault('depth', 4)
    policy_cfg.setdefault('num_bodies', 24)
    policy_cfg.setdefault('fixed_std', True)     # 蒸馏期间固定学生 std，KL 更稳定
    policy_cfg.setdefault('init_noise_std', 0.055)

    student = ActorCritic(
        policy_cfg['actor_input_dim'],
        policy_cfg['critic_input_dim'],
        env.num_actions,
        **policy_cfg
    ).to(device)

    #
    # student = ActorCritic_Attention(
    #     env.num_obs,
    #     env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs,
    #     env.num_actions,
    #     **policy_cfg
    # ).to(device)
    return student





@torch.no_grad()
def teacher_outputs(teacher, obs, critic_obs):
    mu_q, lv_q, std_q = teacher.update_latent_distribution(obs)
    v = teacher.evaluate(critic_obs)
    return mu_q, lv_q, v


def kl_gaussians(mu_t, std_t, mu_s, std_s, eps=1e-8):
    std_t = std_t.clamp_min(eps)
    std_s = std_s.clamp_min(eps)
    var_t, var_s = std_t*std_t, std_s*std_s
    kl = torch.log(std_s/std_t) + (var_t + (mu_t - mu_s)**2) / (2.0 * var_s) - 0.5
    return kl.sum(dim=-1)


def rollout_dagger(env, teacher, student, steps_per_env, beta, device):
    """学生主导，β-混合执行；老师用来标注当前obs。返回新采样的 (obs, mu_t, std_t, v_t)。"""
    obs = env.get_observations().to(device)
    priv = env.get_privileged_observations()
    critic_obs = (priv if priv is not None else obs).to(device)

    obs_list, mu_list, std_list, val_list = [], [], [], []

    for _ in range(steps_per_env):
        # 老师标签
        with torch.no_grad():
            # proprioception
            obs_front = obs[:, :358]
            # action + task obs
            obs_back = obs[:, -645:]
            # 拼回去（如果你需要）
            obs_teacher = torch.cat((obs_front, obs_back), dim=1)
            mu_t, std_t, v_t = teacher_outputs(teacher, obs_teacher, obs_teacher)

        # 学生分布
        with torch.no_grad():
            # obs_front = obs[:, :1432]
            # # action + task obs
            # obs_back = obs[:, -645:]
            # obs_student = torch.cat((obs_front, obs_back), dim=1)
            obs_student = obs[:, :-645]
            student.update_latent_distribution(obs_student)
            # mu_s = student.action_mean


        # β-混合动作 方案 1 按概率选老师/学生
        a_teacher = teacher.act_inference(obs_teacher)
        a_student = student.act_inference(obs_student)
        mask = (torch.rand(obs.shape[0], 1, device=device) < beta)
        act = torch.where(mask, a_teacher, a_student)

        # β-混合动作 方案 2
        # act = beta * mu_t + (1.0 - beta) * mu_s


        # 记录（obs+老师标签）
        obs_list.append(obs)
        mu_list.append(mu_t)
        std_list.append(std_t)
        val_list.append(v_t)

        # 交互
        with torch.no_grad():
            obs, priv, _, _, _ = env.step(act)
            obs = obs.to(device)
            critic_obs = (priv.to(device) if priv is not None else obs)

    return (torch.cat(obs_list, dim=0),
            torch.cat(mu_list, dim=0),
            torch.cat(std_list, dim=0),
            torch.cat(val_list, dim=0))


def train_student_epoch(student, optimizer, rb: ReplayBuf, cfg: DAggerCfg):
    student.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_mixed_precision)
    autocast = torch.cuda.amp.autocast

    total_loss = total_kl = total_mse = total_vmse = 0.0
    steps = 0

    for _ in range(cfg.epochs_per_iter):
        for obs, mu_t, std_t, v_t in rb.sample_batches(cfg.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_mixed_precision, dtype=cfg.amp_dtype):
                # obs_front = obs[:, :1432]
                # # action + task obs
                # obs_back = obs[:, -645:]
                # obs_student = torch.cat((obs_front, obs_back), dim=1)
                obs_student = obs[:, :-645]
                mu_s, lv_q, std_s = student.update_latent_distribution(obs_student)
                kl_loss  = kl_gaussians(mu_t, std_t, mu_s, std_s).mean()
                mse_loss = F.mse_loss(mu_s, mu_t)
                v_loss = torch.tensor(0.0, device=obs.device)
                if cfg.value_coef > 0:
                    v_pred = student.evaluate(obs_student)
                    v_loss = F.mse_loss(v_pred, v_t)

                # loss = cfg.kl_coef*kl_loss + cfg.mse_coef*mse_loss + cfg.value_coef*v_loss
                loss = cfg.mse_coef*mse_loss + cfg.value_coef*v_loss

            if cfg.use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            total_loss += float(loss.item())
            total_kl   += float(kl_loss.item())
            total_mse  += float(mse_loss.item())
            total_vmse += float(v_loss.item())
            steps += 1
        torch.cuda.empty_cache()

    if steps == 0:
        return 0.0, 0.0, 0.0, 0.0
    return total_loss/steps, total_kl/steps, total_mse/steps, total_vmse/steps


def beta_schedule(t, cfg: DAggerCfg):
    if cfg.beta_decay == "linear":
        return max(cfg.beta_end, cfg.beta_start + (cfg.beta_end - cfg.beta_start) * (t / cfg.max_iters))
    # exponential
    return cfg.beta_end + (cfg.beta_start - cfg.beta_end) * math.exp(-cfg.beta_exp_k * t)



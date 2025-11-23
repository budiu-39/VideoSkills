# videoskills/distill_dagger_phc_to_transformer.py
import os, time, argparse, math, random
from dataclasses import dataclass
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import num_obs_dm

from videoskills.utils import get_args as get_train_args
from videoskills.utils.helpers import print_and_save_cfg, class_to_dict, parse_motion_file_path, dict_to_class
import torch
from typing import Dict, Iterable
from rsl_rl.network.frame_z import FrameEncoderMLP, FramePrior, FrameDecoder
from rsl_rl.storage.rollout_buffer import ReplayBuf
from rsl_rl.backbone.mlp import MLPBackbone


# A100 友好设置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# 导入策略
from rsl_rl.modules.actor_critic_mlp import ActorCriticMLP                   # 老师：PHC/MLP
from rsl_rl.modules.actor_critic import ActorCritic                     # 老师：PHC/MLP
from rsl_rl.modules.actor_critic_attention import ActorCritic_Attention  # 学生：Transformer
from rsl_rl.network.film import FiLMNetwork
from rsl_rl.backbone.body_graph_z import BodyGraphZ as BodyGraphZ_enc
from rsl_rl.backbone.body_graph_z_decoder import BodyGraphZ

# torch.autograd.set_detect_anomaly(True)

@dataclass
class DAggerCfg:
    # 交互
    num_envs: int = 512
    steps_per_env: int = 32         # 每次 DAgger 迭代 rollout 步数
    max_iters: int = 10000           # DAgger 迭代总数

    # β 混合策略（执行动作时：a = beta*teacher + (1-beta)*student）
    beta_start: float = 1.0
    beta_end: float = 0.1
    beta_decay: str = "linear"      # "linear" 或 "exp"
    beta_exp_k: float = 0.002       # exp: beta = beta_end + (beta_start-beta_end)*exp(-k*t)

    # 数据集与优化

    batch_size: int = 2048               # 每次优化的样本数（越大越稳定，但显存占用高）
    batch_segments: int = 256
    epochs_per_iter: int = 5
    lr: float = 5e-4
    weight_decay: float = 0
    betas = (0.9, 0.95)
    save_k_batches_rollout: int = 6
    replay_capacity: int = batch_segments * 32 * save_k_batches_rollout     # 样本上限（T*B 级别）

    # 损失
    prior_coef: float = 0.05
    act_coef: float = 1.0
    mse_coef: float = 1.0
    value_coef: float = 0.0
    ar1_coef: float = 0.05

    # AMP
    use_mixed_precision: bool = True
    amp_dtype = torch.bfloat16

    # 日志/保存
    log_interval: int = 10
    save_interval: int = 100


def build_env_and_cfg(args):
    from videoskills.utils.task_registry import task_registry
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    env_cfg.motion.file = parse_motion_file_path(env_cfg, train_cfg, only_failed_key=False)
    log_dir = print_and_save_cfg(env_cfg, train_cfg, filename="config_dagger.yaml")
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    return env, env_cfg, train_cfg, log_dir


def load_teacher(train_cfg, ckpt_path, env, device):
    policy_cfg = class_to_dict(train_cfg['policy'])

    num_actor_obs  = env.num_obs
    num_critic_obs = env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs
    num_actions    = env.num_actions

    teacher = ActorCriticMLP(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        **policy_cfg
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    teacher.load_state_dict(state, strict=False)
    teacher.eval()
    teacher.set_update_rms(False)

    if hasattr(teacher, "set_update_rms"):
        teacher.set_update_rms(False)
    for p in teacher.parameters():
        p.requires_grad = False

    teacher.obs_dim = num_actor_obs
    teacher.critic_obs_dim = num_critic_obs
    return teacher


def build_student(env, train_cfg, device):
    """
    用 FiLMNetwork 作为 actor 的特征提取；critic 先用 MLP。
    约定 obs 学生侧拼接为: [proprio | act_tail | phase(sin,cos) | z]
    """
    # 1) 超参
    cfg = train_cfg.policy
    d_model      = cfg.d_model
    depth_actor  = cfg.depth_actor
    depth_critic = cfg.depth_critic
    fixed_std    = cfg.fixed_std
    init_std     = cfg.init_noise_std

    # 2) 维度（推荐从 cfg 明确传入；否则按 env 推断）
    proprio_dim = cfg.proprioception_dim
    task_dim = cfg.task_dim
    action_dim  = env.num_actions
    phase_dim   = cfg.phase_dim
    z_dim       = cfg.z_dim
    encoder_obs_dim = proprio_dim + task_dim

    enc_backbone= MLPBackbone(in_dim=encoder_obs_dim, hidden=(3072, 1024, 512)).to(device)  # C_ctx=上下文通道数
    critic_backbone = MLPBackbone(in_dim=encoder_obs_dim, hidden=(512, 512)).to(device)

    # 4) 实例化新版 ActorCritic（它会在内部做 obs RMS、actor_head/critic_head 与分布）
    student = ActorCritic(
        num_actor_obs=encoder_obs_dim,  # 保持兼容
        num_critic_obs=encoder_obs_dim,
        num_actions=z_dim,
        actor_network=enc_backbone,
        critic_network=critic_backbone,
        d_model=512,
        init_noise_std=init_std,
        fixed_std=fixed_std,
    ).to(device)

    student.proprioception_dim = proprio_dim
    student.action_dim = action_dim
    student.phase_dim = phase_dim
    student.z_dim = z_dim
    student.encoder_obs_dim = encoder_obs_dim
    student.task_dim = task_dim

    return student

def build_decoder_and_prior(train_cfg, device):
    cfg = train_cfg.policy
    proprio_dim = cfg.proprioception_dim
    action_dim  = cfg.num_actions
    z_dim       = cfg.z_dim

    decoder_backbone = BodyGraphZ(
        obs_dim=proprio_dim,
        z_dim=z_dim,
        num_bodies=24,
        d_model=256,
        n_heads=4,
        depth=4,
    )
    decoder = FrameDecoder(decoder_backbone, d_model=256, action_dim=action_dim).to(device)
    prior = FramePrior(proprio_dim, d_z=z_dim, hidden=(2048, 1024, 256)).to(device)

    # decoder_backbone = MLPBackbone(in_dim=proprio_dim + z_dim, hidden=(2048, 1024, 512)).to(device)
    # decoder = FrameDecoder(decoder_backbone, d_model=512, action_dim=action_dim).to(device)
    # prior = FramePrior(proprio_dim, d_z=z_dim, hidden=(2048, 1024, 512)).to(device)

    return decoder, prior



@torch.no_grad()
def teacher_outputs(teacher, obs, critic_obs):
    obs_t = obs[..., :teacher.obs_dim]
    critic_t = critic_obs[..., :teacher.critic_obs_dim]

    teacher.update_distribution(obs_t)
    mu = teacher.action_mean
    std = teacher.action_std
    v = teacher.evaluate(critic_t)
    return mu, std, v


def kl_gaussians(mu_t, std_t, mu_s, std_s, eps=1e-8):
    std_t = std_t.clamp_min(eps)
    std_s = std_s.clamp_min(eps)
    var_t, var_s = std_t*std_t, std_s*std_s
    kl = torch.log(std_s/std_t) + (var_t + (mu_t - mu_s)**2) / (2.0 * var_s) - 0.5
    return kl.sum(dim=-1)


@torch.no_grad()
def rollout_dagger(env, teacher, student, decoder, prior, steps_per_env, beta, device):
    obs = env.get_observations().to(device)
    priv = env.get_privileged_observations()
    critic_obs = (priv if priv is not None else obs).to(device)
    student.eval()
    student.set_update_rms(False)
    decoder.eval()
    prior.eval()

    obs_list, mu_list, std_list, val_list = [], [], [], []
    reset_list = []
    for _ in range(steps_per_env):
        mu_t, std_t, v_t = teacher_outputs(teacher, obs, critic_obs)

        # === PULSE: 计算 z ===
        enc_obs = obs[:, :student.encoder_obs_dim]  # encoder 输入去掉 act_tail
        z = student.act(enc_obs)  # q(z|obs)
        proprio = obs[:, :student.proprioception_dim]

        # === 生成学生动作 ===
        decoder_obs = torch.cat([proprio, z], dim=-1)
        mu_s, log_std_s = decoder(decoder_obs)
        std_s = log_std_s.exp()
        eps = torch.randn_like(std_s)
        a_student = mu_s + eps * std_s
        a_teacher = teacher.act(obs)
        mask = (torch.rand(obs.shape[0], 1, device=device) < beta)
        act = torch.where(mask, a_teacher, a_student)

        obs_list.append(obs)
        mu_list.append(mu_t)
        std_list.append(std_t)
        val_list.append(v_t)

        obs, priv, _, reset_buf, _ = env.step(act)
        obs = obs.to(device)
        critic_obs = (priv.to(device) if priv is not None else obs)

        reset_list.append(reset_buf)

    return (
        torch.cat(obs_list, dim=0),
        torch.cat(mu_list, dim=0),
        torch.cat(std_list, dim=0),
        torch.cat(val_list, dim=0),
        torch.cat(reset_list, dim=0),
    )


import torch
import torch.nn.functional as F
from torch import nn

def kl_normal(mu_q, lv_q, mu_p, lv_p):
    var_q = torch.exp(lv_q)
    var_p = torch.exp(lv_p)
    return 0.5 * ((lv_p - lv_q) + (var_q + (mu_q - mu_p)**2) / var_p - 1.0).sum(dim=-1)

def compute_ar1_from_rollout(z_flat, reset_flat, T, B, phi=0.99):
    """
    z_flat:      [T*B, z_dim]    所有时间步的 latent（按 time-major: step0 env0..B-1, step1 env0..B-1, ...）
    reset_flat:  [T*B]           每一步 step_t → step_{t+1} 是否 reset（来自 env.step 的 reset_buf）
    T:           steps_per_env
    B:           num_envs
    """
    device = z_flat.device
    z_dim = z_flat.shape[-1]

    # 还原成 [T, B, z_dim], [T, B]
    z_seq = z_flat.view(T, B, z_dim)
    reset_seq = reset_flat.view(T, B).bool()

    # z_{t+1} - phi * z_t, 只在 t = 0..T-2 有意义
    error = z_seq[1:] - phi * z_seq[:-1]      # [T-1, B, z_dim]

    # reset_seq[t] 表示从 t 这一步迈向 t+1 时是否 reset
    # 有 reset 的 pair 不做 AR(1)
    valid_mask = ~reset_seq[:-1]              # [T-1, B]

    if not valid_mask.any():
        return torch.tensor(0.0, device=device)

    error_valid = error[valid_mask]           # [N_pairs, z_dim]
    ar1 = torch.norm(error_valid, dim=-1).mean()
    return ar1

def train_step_segment(student, optimizer, seg_buf, cfg, decoder, prior):
    student.train(); decoder.train(); prior.train()
    student.set_update_rms(True)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_mixed_precision)
    autocast = torch.cuda.amp.autocast

    tot_loss = tot_act_loss = tot_kl_lat = tot_ar1 = 0.0
    steps = 0

    for _ in range(cfg.epochs_per_iter):
        if len(seg_buf) == 0:
            break

        # 像 ReplayBuf.sample_batches 那样，遍历整个 buffer
        for obs_b, mu_t, std_t, v_t, reset_b in seg_buf.iter_segment_batches(cfg.batch_segments):
            T = cfg.steps_per_env
            # B = 当前 batch 里 segment 的数量
            B = obs_b.shape[0] // T

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_mixed_precision, dtype=cfg.amp_dtype):
                # posterior q(z|obs)
                enc_obs = obs_b[:, :student.encoder_obs_dim]
                mu_q, lv_q, std_q = student.update_latent_distribution(enc_obs)
                eps = torch.randn_like(std_q)
                z = mu_q + eps * std_q

                # prior p(z|proprio)
                proprio = obs_b[:, :student.proprioception_dim]
                mu_p, lv_p = prior(proprio)
                kl_lat = kl_normal(mu_q, lv_q, mu_p, lv_p).mean()

                # decoder 蒸馏
                decoder_obs = torch.cat([proprio, mu_q], dim=-1)
                mu_s, log_std = decoder(decoder_obs)
                act_loss = torch.norm(mu_s - mu_t, dim=-1).mean()

                # AR(1) over latent（每个 batch 内部：T x B）
                ar1_loss = torch.tensor(0.0, device=obs_b.device)
                if getattr(cfg, "ar1_coef", 0.0) > 0:
                    ar1_loss = compute_ar1_from_rollout(
                        mu_q, reset_b, T, B,
                        phi=getattr(cfg, "ar1_phi", 0.99)
                    )

                loss = cfg.act_coef * act_loss + cfg.prior_coef * kl_lat + cfg.ar1_coef * ar1_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            tot_loss += loss.item()
            tot_act_loss += act_loss.item()
            tot_kl_lat += kl_lat.item()
            tot_ar1 += (ar1_loss.item() if isinstance(ar1_loss, torch.Tensor) else float(ar1_loss))
            steps += 1

    return dict(
        total=tot_loss / max(steps, 1),
        beh_kl=tot_act_loss / max(steps, 1),
        kl_lat=tot_kl_lat / max(steps, 1),
        ar1=tot_ar1 / max(steps, 1),
    )


def train_on_batch(student, optimizer, cfg, decoder , prior, rollout_batch):
    student.train()
    decoder.train()
    prior.train()
    student.set_update_rms(True)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_mixed_precision)
    autocast = torch.cuda.amp.autocast

    obs_flat, mu_t_flat, std_t_flat, v_t_flat, reset_flat = rollout_batch
    device = obs_flat.device

    T = cfg.steps_per_env
    B = cfg.num_envs   # 记得在 main 里把 cfg.num_envs = env.num_envs

    tot_loss = tot_act_loss = tot_kl_lat = tot_ar1 = 0.0
    steps = 0

    for _ in range(cfg.epochs_per_iter):
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.use_mixed_precision, dtype=cfg.amp_dtype):
            # === 1) posterior q(z | obs) ===
            enc_obs = obs_flat[:, :student.encoder_obs_dim]
            mu_q, lv_q, std_q = student.update_latent_distribution(enc_obs)
            eps = torch.randn_like(std_q)
            z = mu_q + eps * std_q   # 也可以只用 mu_q

            # === 2) prior p(z | proprio) ===
            proprio = obs_flat[:, :student.proprioception_dim]
            mu_p, lv_p = prior(proprio)
            kl_lat = cfg.prior_coef * kl_normal(mu_q, lv_q, mu_p, lv_p).mean()

            # === 3) decoder 蒸馏动作 ===
            # 这里用 mu_q 当 latent 输入，避免采样噪声对蒸馏造成干扰
            decoder_obs = torch.cat([proprio, mu_q], dim=-1)
            mu_s, log_std = decoder(decoder_obs)
            act_loss = torch.norm(mu_s - mu_t_flat, dim=-1).mean()

            ar1_loss = torch.tensor(0.0, device=device)
            if getattr(cfg, "ar1_coef", 0.0) > 0.0:
                # 用 mu_q 或 z 都行，这里用 mu_q 更“干净”
                ar1_loss = cfg.ar1_coef * compute_ar1_from_rollout(
                    mu_q, reset_flat, T, B, phi=getattr(cfg, "ar1_phi", 0.99)
                )


            loss = act_loss + kl_lat + ar1_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()


        tot_loss += loss.item()
        tot_act_loss += act_loss.item()
        tot_kl_lat += kl_lat.item()
        tot_ar1 += (ar1_loss.item() if isinstance(ar1_loss, torch.Tensor) else float(ar1_loss))
        steps += 1

    return dict(
        total=tot_loss / steps,
        beh_kl=tot_act_loss / steps,
        kl_lat=tot_kl_lat / steps,
        ar1=tot_ar1 / steps,
    )


def train_step(student, optimizer, rb, cfg,
               decoder , prior,
               ):
    student.train(); decoder.train(); prior.train()
    student.set_update_rms(True)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_mixed_precision)
    autocast = torch.cuda.amp.autocast
    tot_loss = tot_act_loss = tot_kl_lat = 0.0
    steps = 0

    for _ in range(cfg.epochs_per_iter):
        for obs_b, mu_t, std_t, v_t, *_ in rb.sample_batches(cfg.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_mixed_precision, dtype=cfg.amp_dtype):
                # === PULSE: 计算 q(z|obs), p(z|humanoid_obs) ===
                enc_obs = obs_b[:, :student.encoder_obs_dim]
                mu_q, lv_q, std_q = student.update_latent_distribution(enc_obs)
                eps = torch.randn_like(std_q)
                z = mu_q + eps * std_q

                prior_obs = obs_b[:, :student.proprioception_dim]  # humanoid_obs
                mu_p, lv_p = prior(prior_obs)

                kl_lat = kl_normal(mu_q, lv_q, mu_p, lv_p).mean()
                # === 学生输入 = [proprio, z] ===
                proprio = obs_b[:, :student.proprioception_dim]
                decoder_obs = torch.cat([proprio, mu_q], dim=-1)
                mu_s, log_std = decoder(decoder_obs)
                std_s = log_std.exp()

                # === 蒸馏损失 ===
                # 方案A ：mode-covering loss
                # var_t, var_s = std_t**2, std_s**2
                # kl_beh = torch.log(std_s/std_t) + (var_t + (mu_t - mu_s)**2) / (2*var_s) - 0.5
                # act_loss = kl_beh.sum(dim=-1).mean()

                # 方案B ：RMSE
                act_loss = torch.norm(mu_s - mu_t, dim=-1).mean()

                loss = cfg.act_coef * act_loss + cfg.prior_coef * kl_lat

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            tot_loss += loss.item()
            tot_act_loss += act_loss.item()
            tot_kl_lat += kl_lat.item()
            steps += 1

    return dict(
        total=tot_loss/steps,
        beh_kl=tot_act_loss/steps,
        kl_lat=tot_kl_lat/steps,
    )



def beta_schedule(t, cfg: DAggerCfg):
    if cfg.beta_decay == "linear":
        return max(cfg.beta_end, cfg.beta_start + (cfg.beta_end - cfg.beta_start) * (t / cfg.max_iters))
    # exponential
    return cfg.beta_end + (cfg.beta_start - cfg.beta_end) * math.exp(-cfg.beta_exp_k * t)

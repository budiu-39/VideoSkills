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
    num_envs: int = 1024
    steps_per_env: int = 32         # 每次 DAgger 迭代 rollout 步数
    max_iters: int = 1000           # DAgger 迭代总数

    # β 混合策略（执行动作时：a = beta*teacher + (1-beta)*student）
    beta_start: float = 1.0
    beta_end: float = 0.1
    beta_decay: str = "linear"      # "linear" 或 "exp"
    beta_exp_k: float = 0.002       # exp: beta = beta_end + (beta_start-beta_end)*exp(-k*t)

    # 数据集与优化
    replay_capacity: int = 2_000_000      # 样本上限（T*B 级别）
    batch_size: int = 16384               # 每次优化的样本数（越大越稳定，但显存占用高）
    epochs_per_iter: int = 1
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

    num_actor_obs  = env.num_obs
    num_critic_obs = env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs
    num_actions    = env.num_actions

    teacher = ActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
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

    return teacher

def build_student(env, train_cfg, device):
    policy_cfg = class_to_dict(train_cfg.policy)
    policy_cfg.setdefault('d_model', 256)
    policy_cfg.setdefault('n_heads', 4)
    policy_cfg.setdefault('depth', 4)
    policy_cfg.setdefault('num_bodies', 24)
    policy_cfg.setdefault('fixed_std', True)     # 蒸馏期间固定学生 std，KL 更稳定
    policy_cfg.setdefault('init_noise_std', 0.055)

    student = ActorCritic_Attention(
        env.num_obs,
        env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs,
        env.num_actions,
        **policy_cfg
    ).to(device)
    return student


class ReplayBuf:
    """
    聚合缓冲（DAgger）：数据常驻 CPU，随机抽样时才搬到 GPU。
    避免每次训练把所有块 cat 到一个超大 GPU 张量导致 OOM。
    """
    def __init__(self, capacity, device, dtype=torch.float32, pin_memory=True):
        self.device = device
        self.capacity = capacity
        self.pin_memory = pin_memory
        self.dtype = dtype

        self.obs_chunks = []
        self.mu_chunks  = []
        self.std_chunks = []
        self.val_chunks = []
        self.size = 0  # 样本总数

    def _to_cpu(self, x):
        x = x.detach().to('cpu', dtype=self.dtype, non_blocking=False)
        if self.pin_memory:
            try:
                x = x.pin_memory()
            except:
                pass
        return x

    def add(self, obs, mu, std, val):
        # 所有新数据搬到 CPU 存（避免 GPU 长驻）
        self.obs_chunks.append(self._to_cpu(obs))
        self.mu_chunks.append(self._to_cpu(mu))
        self.std_chunks.append(self._to_cpu(std))
        self.val_chunks.append(self._to_cpu(val))
        self.size += obs.shape[0]

        # 截断容量：从最早的块开始删
        while self.size > self.capacity and len(self.obs_chunks) > 0:
            popped_n = self.obs_chunks[0].shape[0]
            self.obs_chunks.pop(0); self.mu_chunks.pop(0); self.std_chunks.pop(0); self.val_chunks.pop(0)
            self.size -= popped_n

    def __len__(self):
        return self.size

    def sample_batches(self, batch_size):
        """
        随机索引而不是全量 cat。
        思路：先给每个块一个 “范围映射”，然后在全局范围内采样索引，再落到具体块与偏移。
        """
        if self.size == 0:
            return
        # 构建累计长度前缀（仅整数列表，CPU 轻量）
        lens = [t.shape[0] for t in self.obs_chunks]
        import numpy as np
        cumsum = np.cumsum([0] + lens)  # len = n_chunks+1
        N = cumsum[-1]

        # 随机打乱全局索引并按 batch 切分
        idx_global = torch.randperm(N)  # 在 CPU
        for s in range(0, N, batch_size):
            t = min(s + batch_size, N)
            sel = idx_global[s:t].numpy()

            # 将全局索引映射到 (chunk_id, offset)
            # 二分定位：np.searchsorted(cumsum, i, side='right')-1
            import bisect
            obs_list = []; mu_list = []; std_list = []; val_list = []
            for g in sel:
                cid = bisect.bisect_right(cumsum, g) - 1
                off = g - cumsum[cid]
                obs_list.append(self.obs_chunks[cid][off:off+1])
                mu_list.append(self.mu_chunks[cid][off:off+1])
                std_list.append(self.std_chunks[cid][off:off+1])
                val_list.append(self.val_chunks[cid][off:off+1])

            # 小批拼接（仍在 CPU），然后一次性搬到 GPU
            obs_mb  = torch.cat(obs_list, dim=0).to(self.device, non_blocking=True)
            mu_mb   = torch.cat(mu_list,  dim=0).to(self.device, non_blocking=True)
            std_mb  = torch.cat(std_list, dim=0).to(self.device, non_blocking=True)
            val_mb  = torch.cat(val_list, dim=0).to(self.device, non_blocking=True)
            yield obs_mb, mu_mb, std_mb, val_mb



@torch.no_grad()
def teacher_outputs(teacher, obs, critic_obs):
    teacher.update_distribution(obs)
    mu = teacher.action_mean
    std = teacher.action_std
    v = teacher.evaluate(critic_obs)
    return mu, std, v


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
            mu_t, std_t, v_t = teacher_outputs(teacher, obs, critic_obs)

        # 学生分布
        with torch.no_grad():
            student.update_distribution(obs)
            # mu_s = student.action_mean


        # β-混合动作 方案 1 按概率选老师/学生
        a_teacher = teacher.act_inference(obs)
        a_student = student.act_inference(obs)
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
                student.update_distribution(obs)
                mu_s, std_s = student.action_mean, student.action_std
                kl_loss  = kl_gaussians(mu_t, std_t, mu_s, std_s).mean()
                mse_loss = F.mse_loss(mu_s, mu_t)
                v_loss = torch.tensor(0.0, device=obs.device)
                if cfg.value_coef > 0:
                    v_pred = student.evaluate(obs)
                    v_loss = F.mse_loss(v_pred, v_t)

                loss = cfg.kl_coef*kl_loss + cfg.mse_coef*mse_loss + cfg.value_coef*v_loss

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


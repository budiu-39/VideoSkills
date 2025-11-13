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
import torch
from typing import Dict, Iterable


# A100 友好设置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# 导入策略
from rsl_rl.modules.actor_critic import ActorCritic as ActorCriticMLP                   # 老师：PHC/MLP
from rsl_rl.modules.actor_critic_new import ActorCritic                     # 老师：PHC/MLP
from rsl_rl.modules.actor_critic_attention import ActorCritic_Attention  # 学生：Transformer
from rsl_rl.network.film import FiLMNetwork, MLPBackbone



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
    replay_capacity: int = 2_000_000      # 样本上限（T*B 级别）
    batch_size: int = 8192               # 每次优化的样本数（越大越稳定，但显存占用高）
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
    action_dim  = env.num_actions
    phase_dim   = cfg.phase_dim
    z_dim       = cfg.z_dim

    if proprio_dim is None:
        # 兜底推断：原始 env.obs 去掉末尾 action_tail（若你的构成不同请改这里）
        proprio_dim = int(env.num_obs) - int(action_dim)

    # 学生 actor 的“输入长度”= 拼接后的总维度
    num_actor_obs = proprio_dim + action_dim + phase_dim + z_dim

    # critic 用 env 的 privileged obs；没有就用与 actor 相同
    num_critic_obs = getattr(env, "num_privileged_obs", None) or num_actor_obs

    # 3) 构建 actor/critic 网络后端
    actor_backbone = FiLMNetwork(
        dims=(proprio_dim, action_dim, phase_dim, z_dim),
        hid=d_model,              # 必须与 ActorCritic.d_model 一致
        depth_film=cfg.depth_film,
        backbone=MLPBackbone(in_dim=proprio_dim + action_dim + phase_dim,
                             hid=d_model, depth=depth_actor)
    )  # 输出 [B, d_model]  → 由 ActorCritic 的 actor_head 映射到动作均值

    critic_backbone = MLPBackbone(in_dim=num_critic_obs, hid=d_model, depth=depth_critic)

    # 4) 实例化新版 ActorCritic（它会在内部做 obs RMS、actor_head/critic_head 与分布）
    student = ActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=env.num_actions,
        actor_network=actor_backbone,
        critic_network=critic_backbone,
        d_model=d_model,
        init_noise_std=init_std,
        fixed_std=fixed_std,
    ).to(device)

    student.proprioception_dim = proprio_dim
    student.action_dim = action_dim
    student.phase_dim = phase_dim
    student.z_dim = z_dim

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
        self.ctxkey_chunks = []
        self.phase_chunks = []
        self.size = 0  # 样本总数

    def _to_cpu(self, x, dtype=None):
        x = x.detach().to('cpu', non_blocking=False)
        if dtype is not None:
            x = x.to(dtype)
        if self.pin_memory:
            try:
                x = x.pin_memory()
            except:
                pass
        return x

    def add(self, obs, mu, std, val, ctx_keys, phase):
        self.obs_chunks.append(self._to_cpu(obs, self.dtype))
        self.mu_chunks.append(self._to_cpu(mu, self.dtype))
        self.std_chunks.append(self._to_cpu(std, self.dtype))
        self.val_chunks.append(self._to_cpu(val, self.dtype))
        self.ctxkey_chunks.append(self._to_cpu(ctx_keys, dtype=torch.long))  # ★
        self.phase_chunks.append(self._to_cpu(phase, self.dtype))

        self.size += obs.shape[0]
        while self.size > self.capacity and len(self.obs_chunks) > 0:
            popped_n = self.obs_chunks[0].shape[0]
            self.obs_chunks.pop(0);
            self.mu_chunks.pop(0);
            self.std_chunks.pop(0)
            self.val_chunks.pop(0);
            self.ctxkey_chunks.pop(0)
            self.size -= popped_n

    def __len__(self):
        return self.size

    def sample_batches(self, batch_size: int):
        """
        随机索引批量采样；返回五元组：
        (obs_mb, mu_mb, std_mb, val_mb, ctxkey_mb)
        """
        if self.size == 0:
            return
        lens = [t.shape[0] for t in self.obs_chunks]
        import numpy as np, bisect
        cumsum = np.cumsum([0] + lens)  # len = n_chunks+1
        N = cumsum[-1]

        idx_global = torch.randperm(N)  # CPU
        for s in range(0, N, batch_size):
            t = min(s + batch_size, N)
            sel = idx_global[s:t].numpy()

            obs_list, mu_list, std_list, val_list, key_list, phase_list = [], [], [], [], [], []
            for g in sel:
                cid = bisect.bisect_right(cumsum, g) - 1
                off = g - cumsum[cid]
                obs_list.append(self.obs_chunks[cid][off:off+1])
                mu_list.append(self.mu_chunks[cid][off:off+1])
                std_list.append(self.std_chunks[cid][off:off+1])
                val_list.append(self.val_chunks[cid][off:off+1])
                key_list.append(self.ctxkey_chunks[cid][off:off+1])
                phase_list.append(self.phase_chunks[cid][off:off + 1])

                # 仍在 CPU cat，再一次性搬 GPU
            obs_mb  = torch.cat(obs_list, dim=0).to(self.device, non_blocking=True)
            mu_mb   = torch.cat(mu_list,  dim=0).to(self.device, non_blocking=True)
            std_mb  = torch.cat(std_list, dim=0).to(self.device, non_blocking=True)
            val_mb  = torch.cat(val_list, dim=0).to(self.device, non_blocking=True)
            ctxkey_mb = torch.cat(key_list, dim=0).to(self.device, non_blocking=True)
            phase_mb = torch.cat(phase_list, dim=0).to(self.device, non_blocking=True)

            yield obs_mb, mu_mb, std_mb, val_mb, ctxkey_mb, phase_mb


# === in EpisodeCtxBuf ===
class EpisodeCtxBuf:
    def __init__(self):
        self._ctx = {}
        self._prior = {}
        self._mask = {}
        # 新增：tuple mid <-> int key_id
        self._mid2id = {}  # dict[tuple,int]
        self._id2mid = {}  # dict[int,tuple]
        self._next_id = 0

    @torch.no_grad()
    def add(self, ctx, prior, mids, mask):
        """
        mids: List[tuple]，长度 = B
        返回:
          key_ids: LongTensor[B]，可写入 env.current_ctx_key / ReplayBuf
        """
        B = len(mids)
        key_ids = []
        ctx = ctx.detach().to('cpu').contiguous()
        prior = prior.detach().to('cpu').contiguous()
        mask = mask.detach().to('cpu').contiguous().to(torch.bool)

        for i in range(B):
            mid = mids[i]
            if mid not in self._mid2id:
                kid = self._next_id
                self._next_id += 1
                self._mid2id[mid] = kid
                self._id2mid[kid] = mid
            else:
                kid = self._mid2id[mid]

            self._ctx[kid] = ctx[i]
            self._prior[kid] = prior[i]
            self._mask[kid] = mask[i]
            key_ids.append(kid)

        return torch.tensor(key_ids, dtype=torch.long)

    @torch.no_grad()
    def get_by_keys(self, key_ids: torch.Tensor):
        k_list = key_ids.to('cpu').tolist()
        ctxs, priors, masks = [], [], []
        for k in k_list:
            ctxs.append(self._ctx[k]);
            priors.append(self._prior[k]);
            masks.append(self._mask[k])
        return (torch.stack(ctxs, 0),
                torch.stack(priors, 0),
                torch.stack(masks, 0))

    # 供 proto_loss 查回 tuple mid
    def id_to_mid(self, key_ids: torch.Tensor):
        return [self._id2mid[int(k)] for k in key_ids.to('cpu').tolist()]


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


def rollout_dagger(env, teacher, student, steps_per_env, beta, device):
    obs = env.get_observations().to(device)
    priv = env.get_privileged_observations()
    critic_obs = (priv if priv is not None else obs).to(device)

    obs_list, mu_list, std_list, val_list, key_list, phase_list = [], [], [], [], [], []   # ★

    for _ in range(steps_per_env):
        with torch.no_grad():
            mu_t, std_t, v_t = teacher_outputs(teacher, obs, critic_obs)

        with torch.no_grad():
            phase_local = env.local_phase_buf
            z_step = env.z_global_buf.to(device)  # [B,d_z] 由 z_provider 维护
            obs_student = build_student_obs_from_parts(obs, student, phase_local, z_step)
            student.update_distribution(obs_student)
            a_student = student.act_inference(obs_student)
        a_teacher = teacher.act_inference(obs)
        mask = (torch.rand(obs.shape[0], 1, device=device) < beta)
        act = torch.where(mask, a_teacher, a_student)

        # ★ 在 step 之前/之后都可以，只要与你的 env.current_ctx_key 更新逻辑一致
        ctx_key_now = env.current_ctx_key.to(device=device)   # [B] Long
        key_list.append(ctx_key_now.clone())

        obs_list.append(obs_student); mu_list.append(mu_t); std_list.append(std_t); val_list.append(v_t)
        phase_list.append(phase_local)

        with torch.no_grad():
            obs, priv, _, _, _ = env.step(act)
            obs = obs.to(device)
            critic_obs = (priv.to(device) if priv is not None else obs)

    return (torch.cat(obs_list, dim=0),
            torch.cat(mu_list, dim=0),
            torch.cat(std_list, dim=0),
            torch.cat(val_list, dim=0),
            torch.cat(key_list, dim=0),
            torch.cat(phase_list, dim=0))


import torch
import torch.nn.functional as F
from torch import nn

def _kl_gaussians(mu_p, std_p, mu_q, std_q):
    # KL( N(mu_p,std_p^2) || N(mu_q,std_q^2) )，逐样本逐维
    var_p, var_q = std_p**2, std_q**2
    return 0.5 * ( (var_p/var_q) + ((mu_q - mu_p)**2)/var_q - 1.0 + torch.log(var_q/var_p) ).sum(dim=-1)

def train_step(student, optimizer_student, rb, cfg,
               enc=None, prior=None, optimizer_enc=None,
               ctx_buf=None, beta_kl:float=0.0, alpha_proto:float=0.0, proto_ema=None,
               behavior_loss:str="kl",   # "kl" 或 "mse"
               use_value_distill:bool=False,
               use_mixed_precision:bool=True,
               amp_dtype=torch.bfloat16):
    """
    behavior_loss:
      - "kl": KL(teacher || student)（推荐）
      - "mse": MSE(μ_s, μ_t)
    """
    student.train()
    if enc is not None: enc.train()
    if prior is not None: prior.train()

    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    autocast = torch.cuda.amp.autocast

    tot, tot_kl_beh, tot_mse_beh, tot_v, tot_kl_lat, tot_proto = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    steps = 0

    for _ in range(cfg.epochs_per_iter):
        for obs_b, mu_t, std_t, v_t, ctxkey_b, phase_b in rb.sample_batches(cfg.batch_size):
            # 老师标签不带梯度
            mu_t   = mu_t.detach()
            std_t  = std_t.detach()
            v_t    = v_t.detach()

            optimizer_student.zero_grad(set_to_none=True)
            if optimizer_enc is not None:
                optimizer_enc.zero_grad(set_to_none=True)

            with autocast(enabled=use_mixed_precision, dtype=amp_dtype):
                # 1) 取回上下文 → 编码 z（带梯度）
                ctx_cpu, prior_cpu, mask_cpu = ctx_buf.get_by_keys(ctxkey_b)
                ctx_mb   = ctx_cpu.to(obs_b.device, non_blocking=True).requires_grad_(True)  # [B,C,K]
                prior_mb = prior_cpu.to(obs_b.device, non_blocking=True)
                mask_mb  = mask_cpu.to(obs_b.device, non_blocking=True)
                z, mu_q, lv_q = enc(ctx_mb, mask=mask_mb)

                # 先验
                if prior is not None:
                    mu_p, lv_p = prior(prior_mb)
                else:
                    mu_p = torch.zeros_like(mu_q); lv_p = torch.zeros_like(lv_q)

                # 稳定：限制 logvar
                lv_q = lv_q.clamp_(-8.0, 8.0)
                lv_p = lv_p.clamp_(-8.0, 8.0)

                # 潜变量 KL（对条件先验）
                std_q = (0.5*lv_q).exp()
                std_p = (0.5*lv_p).exp()
                kl_lat = _kl_gaussians(mu_q, std_q, mu_p, std_p).mean()

                z_det = z.detach()

                # （可选）原型一致性：需要你手上有 motion_id→proto_ema 的映射；没有可为 0
                proto_loss = torch.zeros((), device=obs_b.device)
                # 若你已提供 motion_id，可在此构造 proto 张量并计算 F.mse_loss(z, proto)

                # 2) 拼 obs：base 不参与梯度，z 参与
                proprio = obs_b[..., :student.proprioception_dim].detach()
                act_tail = obs_b[..., -student.action_dim:].detach()
                base = torch.cat([proprio, act_tail], dim=-1)
                obs_student = torch.cat([base, phase_b, z_det], dim=-1)

                # 3) 学生前向 & 行为蒸馏
                student.update_distribution(obs_student)
                mu_s, std_s = student.action_mean, student.action_std

                if behavior_loss == "kl":
                    # KL(teacher || student)
                    loss_beh_kl = _kl_gaussians(mu_t, std_t, mu_s, std_s).mean()
                    loss_beh_mse = torch.tensor(0.0, device=obs_b.device)
                    loss_beh = loss_beh_kl
                else:
                    loss_beh_mse = F.mse_loss(mu_s, mu_t)
                    loss_beh_kl = torch.tensor(0.0, device=obs_b.device)
                    loss_beh = loss_beh_mse

                # （可选）value 蒸馏
                loss_v = torch.tensor(0.0, device=obs_b.device)
                if use_value_distill and hasattr(student, "evaluate"):
                    v_pred = student.evaluate(obs_student)
                    loss_v = F.mse_loss(v_pred, v_t)

                if alpha_proto > 0.0 and (proto_ema is not None) and (ctx_buf is not None):
                    mids_batch = ctx_buf.id_to_mid(ctxkey_b)  # List[tuple]
                    proto_list = []
                    miss = 0
                    for mid in mids_batch:
                        if mid in proto_ema:
                            proto_list.append(proto_ema[mid].to(obs_b.device))
                        else:
                            # 若首次出现还没EMA，就用当前 z 作为“原型”（避免 NaN）
                            # 或者用 detach 的 mu_q 也行
                            proto_list.append(z.new_zeros(z.size(1)).to(obs_b.device))
                            miss += 1
                    if len(proto_list) > 0:
                        proto = torch.stack(proto_list, dim=0)  # [B, d_z]
                        # 通常用 L2，或 cos 距离
                        proto_loss = torch.mean(torch.sum((z - proto) ** 2, dim=-1))

                # 总损失
                loss = loss_beh + cfg.value_coef*loss_v +  beta_kl*kl_lat + alpha_proto*proto_loss
                # loss = cfg.value_coef*loss_v + alpha_proto*proto_loss
                # 反传
                if use_mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer_student)
                    nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    if optimizer_enc is not None:
                        scaler.unscale_(optimizer_enc)
                        nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                        if prior is not None:
                            nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
                    scaler.step(optimizer_student)
                    if optimizer_enc is not None:
                        scaler.step(optimizer_enc)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    if enc is not None:
                        nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                        if prior is not None:
                            nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
                    optimizer_student.step()
                    if optimizer_enc is not None:
                        optimizer_enc.step()
                # for attr in ("distribution", "action_mean", "action_std", "pre_tanh_value"):
                #     if hasattr(student, attr):
                #         setattr(student, attr, None)

            # 统计
            tot       += float(loss.item()); steps += 1
            tot_kl_beh+= float(loss_beh_kl.item())
            tot_mse_beh+=float(loss_beh_mse.item())
            tot_v     += float(loss_v.item())
            if enc is not None:
                tot_kl_lat += float((beta_kl*kl_lat).item())
                tot_proto  += float((alpha_proto*proto_loss).item())

    if steps == 0:
        return dict(total=0.0, beh_kl=0.0, beh_mse=0.0, v=0.0, kl_lat=0.0, proto=0.0)

    return dict(
        total   = tot/steps,
        beh_kl  = tot_kl_beh/steps,
        beh_mse = tot_mse_beh/steps,
        v       = tot_v/steps,
        kl_lat  = tot_kl_lat/steps,
        proto   = tot_proto/steps
    )




def beta_schedule(t, cfg: DAggerCfg):
    if cfg.beta_decay == "linear":
        return max(cfg.beta_end, cfg.beta_start + (cfg.beta_end - cfg.beta_start) * (t / cfg.max_iters))
    # exponential
    return cfg.beta_end + (cfg.beta_start - cfg.beta_end) * math.exp(-cfg.beta_exp_k * t)

def build_student_obs_from_parts(obs, student, phase_b, z_b):
    proprio = obs[..., :student.proprioception_dim]
    act_tail = obs[..., -student.action_dim:]          # 若 obs 末尾就是 action 相关
    base = torch.cat([proprio, act_tail], dim=-1)
    return torch.cat([base, phase_b, z_b], dim=-1)     # [B, proprio+action+phase+z]
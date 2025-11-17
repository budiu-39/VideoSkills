import os, time
from videoskills.utils import get_args
from rsl_rl.algorithms.distill_dagger_window_z import (
    rollout_dagger, train_step, beta_schedule, ReplayBuf,
    build_student, load_teacher, build_env_and_cfg, DAggerCfg,
    EpisodeCtxBuf  # 若在此文件内
)
from rsl_rl.network.episode_encoder import EpisodeEncoder
from videoskills.utils.helpers import class_to_dict
import yaml
import torch
from videoskills.utils.task_registry import task_registry
import wandb

# A100 友好设置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# 导入策略
def main():

    # 复用训练参数解析
    args = get_args()
    device = args.rl_device
    max_iters = args.max_iterations
    teacher_ckpt = args.teacher_ckpt
    device = torch.device(device)
    eval_runner = None

    # 构建环境/配置
    env, env_cfg, train_cfg, log_dir = build_env_and_cfg(args)
    os.makedirs(log_dir, exist_ok=True)

    with open(args.teacher_config, 'r') as f:
        cfg = yaml.safe_load(f)
    teacher_config = cfg.get('train_cfg', {})

    # 老师/学生
    teacher = load_teacher(teacher_config, teacher_ckpt, env, device)
    student = build_student(env, train_cfg, device)

    @torch.no_grad()
    def z_provider(env, env_ids):
        enc.eval()
        prior.eval() if prior is not None else None
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
    # TODO: d_model 是 dimension 中间层的维度
    #  （这里好像弄错了！）
    enc = EpisodeEncoder(in_channels=train_cfg.policy.context_dim, d_model=256, d_z=64).to(device)  # C_ctx=上下文通道数
    prior = None  # 先不用先验，稳了再开：prior = PriorNet(in_dim=S_prior, d_z=32).to(device)
    optimizer_enc = torch.optim.AdamW(
        list(enc.parameters()) + ([] if prior is None else list(prior.parameters())),
        lr=5e-4, betas=(0.9, 0.95), weight_decay=1e-4
    )
    ctx_buf = EpisodeCtxBuf()
    proto_ema = {}
    beta_kl = 0.0  # KL 退火：先 0，稳定后慢慢升
    alpha_proto = 0.0  # 原型一致性：先 0，稳定后 0.01

    # 优化器：先只训 actor/backbone；如需蒸馏 value 再加 critic
    optim_params = list(student.actor_network.parameters()) + list(student.actor_head.parameters())
    if args.distill_value:
        optim_params += list(student.critic_network.parameters()) + list(student.critic_head.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=DAggerCfg.lr, betas=DAggerCfg.betas, weight_decay=DAggerCfg.weight_decay)

    cfg = DAggerCfg()
    if max_iters is not None:
        cfg.max_iters = int(max_iters)

    # 预热
    with torch.inference_mode():
        env.reset()

    # 聚合缓冲
    rb = ReplayBuf(cfg.replay_capacity, device)

    t_last = time.time()

    if args.use_wandb:  # and not args.dev:
        os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)
        run_name = train_cfg.runner.run_name
        wandb.init(project=args.wandb_project, name=run_name,
                   dir=log_dir,
                   config={**vars(args), **class_to_dict(train_cfg), ** class_to_dict(env_cfg)})

    for it in range(1, cfg.max_iters + 1):
        beta = beta_schedule(it, cfg)

        # 1) 学生主导 rollout，老师打标签
        # TODO: 要补充一个config!
        obs, mu_t, std_t, v_t, ctx_key_step, phase  = rollout_dagger(env, teacher, student, cfg.steps_per_env, beta, device)
        rb.add(obs, mu_t, std_t, v_t, ctx_key_step, phase)

        # 2) 在聚合数据上训练学生若干 epoch
        if it > 50:  beta_kl = min(0.01, beta_kl + 0.0002)

        if it > 50:
            alpha_proto = min(0.01, alpha_proto + 0.0005)

        if it > 400: beta_kl = max(0.001, beta_kl * 0.99)

        out = train_step(
            student, optimizer, rb, cfg,
            enc=enc, prior=prior, optimizer_enc=optimizer_enc,
            ctx_buf=ctx_buf, beta_kl=beta_kl, alpha_proto=alpha_proto, proto_ema=proto_ema
        )
        loss, klv, msev, vls, enc_loss, proto_loss = (
            out["total"], out["beh_kl"], out["beh_mse"], out["v"], out["kl_lat"], out["proto"]
        )
        # 3) 日志/保存
        if it % cfg.log_interval == 0 or it == 1:
            dt = time.time() - t_last
            steps_per_iter = cfg.steps_per_env * env.num_envs
            print(f"[DAgger] it={it:05d}  beta={beta:.3f}  |  loss={loss:.4f} (kl={klv:.4f}, mse={msev:.4f}, v={vls:.4f})"
                  f"  |  data={rb.size}  |  fps≈{int(steps_per_iter/cfg.log_interval/dt)}")
            t_last = time.time()

        if it % cfg.save_interval == 0 or it == cfg.max_iters:
            path = os.path.join(log_dir, f"dagger_student_{it}.pt")
            torch.save({
                "model_state_dict": student.state_dict(),
                "encoder_state_dict": enc.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_enc_state_dict": optimizer_enc.state_dict(),
                "iter": it,
                "cfg": cfg.__dict__,
            }, path)
            print(f"[DAgger] saved => {path}")

        # 监控一下聚合的上下文数量/原型数量
        if it % 50 == 0:
            print(f"[Z_Provider] ctx_buf.size={len(ctx_buf._ctx)}, proto_ema={len(proto_ema)}")

        if it % 100 == 0:
            print(f"\n[DAgger] Eval at iteration {it} ...")

            if eval_runner is None:
                import copy
                eval_cfg = copy.deepcopy(train_cfg)
                eval_cfg.runner.policy_class_name = 'ActorCritic'

                eval_cfg.runner.init_storage = False

                # 创建 eval_runner
                eval_runner, _ = task_registry.make_alg_runner(
                    env=env, name=args.task, args=args, train_cfg=eval_cfg, log_dir=log_dir
                )
                print("[DAgger] Created new eval_runner")
            eval_runner.current_learning_iteration = it
            eval_runner.alg.actor_critic.load_state_dict(student.state_dict(), strict=False)
            eval_runner.eval()

    print("[DAgger] done.")


if __name__ == "__main__":
    main()

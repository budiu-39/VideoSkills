import os, time
from videoskills.utils import get_args
from rsl_rl.algorithms.distill_dagger_z import (
    rollout_dagger, train_step, beta_schedule,
    build_student, load_teacher, build_env_and_cfg, DAggerCfg,
    build_decoder_and_prior, train_on_batch, train_step_segment
)
from rsl_rl.storage.rollout_buffer import ReplayBuf
from rsl_rl.storage.rollout_clip_buffer import SegmentReplayBuf
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

    # 老师/学生, decoder, prior, optimizer
    teacher = load_teacher(teacher_config, teacher_ckpt, env, device)
    student = build_student(env, train_cfg, device)
    decoder, prior = build_decoder_and_prior(train_cfg, device)
    env.set_z_prior(prior)
    env.set_z_decoder(decoder)

    optim_params = list(student.actor_network.parameters()) + list(student.actor_head.parameters()) +   \
        list(decoder.parameters()) + ([] if prior is None else list(prior.parameters()))
    distill_value = False
    if distill_value:
        optim_params += list(student.critic_network.parameters()) + list(student.critic_head.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=DAggerCfg.lr, betas=DAggerCfg.betas, weight_decay=DAggerCfg.weight_decay)

    cfg = DAggerCfg()
    cfg.num_envs = env.num_envs
    cfg.steps_per_env = cfg.steps_per_env

    if max_iters is not None:
        cfg.max_iters = int(max_iters)

    # 预热
    with torch.inference_mode():
        env.reset()

    rb = ReplayBuf(
        capacity=cfg.replay_capacity,
        device=device,
    )
    t_last = time.time()

    if args.use_wandb:  # and not args.dev:
        os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)
        run_name = train_cfg.runner.run_name
        wandb.init(project=args.wandb_project, name=run_name,
                   dir=log_dir,
                   config={**vars(args), **class_to_dict(train_cfg), ** class_to_dict(env_cfg)})

        wandb.define_metric("iteration")
        wandb.define_metric("*", step_metric="iteration")

    for it in range(1, cfg.max_iters + 1):
        beta = beta_schedule(it, cfg)

        # 1) 学生主导 rollout，老师打标签
        # TODO: 要补充一个config!

        obs_flat, mu_flat, std_flat, v_flat, reset_flat = rollout_dagger(env, teacher, student, decoder,
                                                                         prior, cfg.steps_per_env, beta, device)

        rb.add(obs_flat, mu_flat, std_flat, v_flat)

        stats = train_step(
            student, optimizer, rb, cfg,
            decoder=decoder, prior=prior
        )

        loss = stats["total"]
        klv = stats.get("beh_kl", 0.0)
        kl_lat = stats.get("kl_lat", 0.0)
        ar1 = stats.get("ar1", 0.0)

        # 3) 日志/保存
        log_interval = train_cfg.runner.log_interval
        if it % log_interval == 0 or it == 1:
            dt = time.time() - t_last
            fps = cfg.steps_per_env * env.num_envs / (log_interval * dt)
            print(f"[DAgger] it={it:05d} | β={beta:.3f} "
                  f"| loss={loss:.3f} (beh={klv:.3f}, lat={kl_lat:.3f}, ar1={ar1: .3f}) "
                  f"| fps≈{fps:.1f}")
            t_last = time.time()

        if args.use_wandb:
            wandb.log({
                "iteration": it,
                "loss/total": loss,
                "loss/behavior_KL": klv,
                "loss/latent_KL": kl_lat,
                "loss/ar1": ar1,
                "beta": beta,
                "fps": cfg.steps_per_env * env.num_envs / (time.time() - t_last + 1e-6)
            }, step=it)


        if it % train_cfg.runner.save_interval == 0 or it == cfg.max_iters:
            ckpt_name = f"dagger_student_{it}.pt"
            path = os.path.join(log_dir, ckpt_name)

            torch.save({
                "model_state_dict": student.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "prior_state_dict": prior.state_dict(),  # ← 补上 prior
                "optimizer_state_dict": optimizer.state_dict(),
                "iter": it,
                "cfg": cfg.__dict__,
            }, path)
            print(f"[DAgger] saved => {path}")

        if it % train_cfg.runner.eval_interval == 0:
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
            decoder.eval() # 主要是为了 rms
            eval_runner.eval()
            decoder.train()

    print("[DAgger] done.")


if __name__ == "__main__":
    main()

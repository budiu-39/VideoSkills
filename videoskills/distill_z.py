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

    student = build_student(env, train_cfg, device)
    decoder, prior = build_decoder_and_prior(train_cfg, device)
    env.set_z_prior(prior)
    env.set_z_decoder(decoder)

    # ========================== 新增：Eval 模式分支 ==========================
    if getattr(args, 'dev', False):
        print("\n[DAgger Z] 🚀 Switch to EVALUATION mode (Skipping training)...")

        # 1. 确定 Checkpoint 路径
        ckpt_path = getattr(args, 'vae_ckpt', None)
        if ckpt_path is None or str(ckpt_path) == '-1':
            raise ValueError("[DAgger Z Eval] Please provide a checkpoint path via --checkpoint.")

        print(f"[DAgger Z Eval] Loading weights from: {ckpt_path}")
        loaded_dict = torch.load(ckpt_path, map_location=device)

        # 2. 加载 Student (Actor/Critic)
        # 兼容不同的保存 key
        if 'model_state_dict' in loaded_dict:
            student.load_state_dict(loaded_dict['model_state_dict'])
        elif 'student_state_dict' in loaded_dict:
            student.load_state_dict(loaded_dict['student_state_dict'])
        else:
            student.load_state_dict(loaded_dict)  # 尝试直接加载

        # 3. 加载 Decoder
        if 'decoder_state_dict' in loaded_dict:
            decoder.load_state_dict(loaded_dict['decoder_state_dict'])
            print(" | Loaded Decoder weights")
        else:
            print("[Warning] No 'decoder_state_dict' found in checkpoint!")

        # 4. 加载 Prior
        if prior is not None and 'prior_state_dict' in loaded_dict:
            prior.load_state_dict(loaded_dict['prior_state_dict'])
            print(" | Loaded Prior weights")

        # 5. 切换到评估模式 (影响 Dropout/BatchNorm 等)
        student.eval()
        decoder.eval()
        if prior is not None:
            prior.eval()

        # 6. 创建 Eval Runner
        import copy
        eval_cfg = copy.deepcopy(train_cfg)
        eval_cfg.runner.init_storage = False
        # 确保 runner 识别正确的 policy 类名 (通常 student 是 ActorCritic_Attention，但 runner 可能默认配置不同)
        # 这里的 hack 是为了让 runner 能够实例化，实际上我们会把 student 实例直接替换进去
        eval_cfg.runner.policy_class_name = 'ActorCritic'

        eval_runner, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=eval_cfg, log_dir=log_dir
        )

        # 7. 注入 Student 模型
        # 直接替换实例，确保使用的是带有 Attention 的 student
        eval_runner.alg.actor_critic = student

        # 再次确保 env 里的 decoder 是最新的 (虽然上面设过了，但为了保险)
        env.set_z_decoder(decoder)
        env.set_z_prior(prior)

        print("[DAgger Z Eval] Starting evaluation loop...")
        eval_runner.eval(log=True)

        return  # 结束程序
    # ========================================================================

    with open(args.teacher_config, 'r') as f:
        cfg = yaml.safe_load(f)
    teacher_config = cfg.get('train_cfg', {})

    # 老师/学生, decoder, prior, optimizer
    teacher = load_teacher(teacher_config, teacher_ckpt, env, device)

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

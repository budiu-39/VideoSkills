import os, time, argparse, math, random
from videoskills.utils import task_registry, get_args as get_train_args
from rsl_rl.algorithms.distill_dagger import (rollout_dagger, train_student_epoch, beta_schedule, ReplayBuf,
                                              load_teacher, build_student, build_env_and_cfg, DAggerCfg)


import torch
import copy
from videoskills.utils import task_registry
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

def main():

    # 复用训练参数解析
    train_args = get_train_args()
    device = train_args.rl_device
    max_iters = train_args.max_iterations
    teacher_ckpt = train_args.teacher_ckpt
    device = torch.device(device)
    eval_runner = None

    # 构建环境/配置
    env, env_cfg, train_cfg, log_dir = build_env_and_cfg(train_args)
    os.makedirs(log_dir, exist_ok=True)

    # 老师/学生
    teacher = load_teacher(teacher_ckpt, env, device)
    student = build_student(env, train_cfg, device)

    # 优化器：先只训 actor/backbone；如需蒸馏 value 再加 critic
    optim_params = list(student.actor_backbone.parameters()) + list(student.actor_head.parameters())
    if train_args.distill_value:
        optim_params += list(student.critic_backbone.parameters()) + list(student.critic_head.parameters())
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

    for it in range(1, cfg.max_iters + 1):
        beta = beta_schedule(it, cfg)

        # 1) 学生主导 rollout，老师打标签
        obs, mu_t, std_t, v_t = rollout_dagger(env, teacher, student, cfg.steps_per_env, beta, device)
        rb.add(obs, mu_t, std_t, v_t)

        # 2) 在聚合数据上训练学生若干 epoch
        loss, klv, msev, vls = train_student_epoch(student, optimizer, rb, cfg)

        # 3) 日志/保存
        if it % cfg.log_interval == 0 or it == 1:
            dt = time.time() - t_last
            steps_per_iter = cfg.steps_per_env * env.num_envs
            print(f"[DAgger] it={it:05d}  beta={beta:.3f}  |  loss={loss:.4f} (kl={klv:.4f}, mse={msev:.4f}, v={vls:.4f})"
                  f"  |  data={rb.size}  |  fps≈{int(steps_per_iter/cfg.log_interval/dt)}")
            t_last = time.time()

        if it % cfg.save_interval == 0 or it == cfg.max_iters:
            path = os.path.join(log_dir, f"dagger_student_{it}.pt")
            torch.save({"student_state_dict": student.state_dict(),
                        "iter": it,
                        "cfg": cfg.__dict__}, path)
            print(f"[DAgger] saved => {path}")

        if it % 20 == 0:
            print(f"\n[DAgger] Eval at iteration {it} ...")

            if eval_runner is None:
                import copy
                eval_cfg = copy.deepcopy(train_cfg)
                eval_cfg.runner.policy_class_name = 'ActorCritic_Attention'

                eval_runner, _ = task_registry.make_alg_runner(
                    env=env,
                    name=train_args.task,
                    args=train_args,
                    train_cfg=eval_cfg,
                    log_dir=log_dir
                )
                print("[DAgger] Created new eval_runner")

            eval_runner.alg.actor_critic.load_state_dict(student.state_dict(), strict=False)
            eval_runner.eval()

    print("[DAgger] done.")


if __name__ == "__main__":
    main()

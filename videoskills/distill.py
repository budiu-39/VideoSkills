import os, time, argparse, math, random
from videoskills.utils import task_registry, get_args
from rsl_rl.algorithms.distill_dagger import (rollout_dagger, train_student_epoch, beta_schedule, ReplayBuf,
                                              build_student, load_teacher, build_env_and_cfg, DAggerCfg)
from videoskills.utils.helpers import class_to_dict
import yaml
import torch
import copy
from videoskills.utils import task_registry
import wandb

import gc
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

    # ========================== 新增：Eval 模式分支 ==========================
    # 检查是否开启了 play 模式 (通常 videoskills.utils.get_args 会包含 --play 参数)
    # 或者你可以手动添加参数解析
    if getattr(args, 'play', False):
        print("\n[DAgger] 🚀 Switch to EVALUATION mode (Skipping training)...")

        # 1. 确定学生模型权重路径
        # 假设通过 --checkpoint 传入，或者在这里硬编码路径进行测试
        # 如果 args 没有 checkpoint 属性，你需要手动指定 student_load_path
        student_load_path = getattr(args, 'checkpoint', None)

        # 如果命令行没传 checkpoint，可以在这里临时硬编码调试：
        # student_load_path = "logs/your_experiment/dagger_student_1000.pt"

        if student_load_path is None or str(student_load_path) == '-1':
            raise ValueError("[DAgger Eval] Please provide a checkpoint path via --checkpoint or hardcode it.")

        print(f"[DAgger Eval] Loading student weights from: {student_load_path}")
        loaded_dict = torch.load(student_load_path, map_location=device)

        # 2. 加载权重 (兼容不同的保存格式)
        if 'student_state_dict' in loaded_dict:
            student.load_state_dict(loaded_dict['student_state_dict'])
        elif 'model_state_dict' in loaded_dict:
            student.load_state_dict(loaded_dict['model_state_dict'])
        else:
            # 尝试直接加载 state_dict
            student.load_state_dict(loaded_dict)

        # 3. 创建 Eval Runner
        # 复用之前的配置，但关闭 storage 初始化以节省显存
        eval_cfg = copy.deepcopy(train_cfg)
        eval_cfg.runner.init_storage = False

        # 使用 task_registry 创建 runner，传入现有的 env
        eval_runner, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=eval_cfg, log_dir=log_dir
        )

        # 4. 将加载好权重的学生模型同步给 Runner 的 ActorCritic
        # 注意：eval_runner 内部默认是 Teacher 类型的 Policy 结构，
        # 如果 Student 结构(Attention)与 Teacher 不同，确保 Runner 能兼容 Student 的 forward
        eval_runner.alg.actor_critic = student  # 直接替换为学生实例最稳妥
        # 或者如果结构完全一致：
        # eval_runner.alg.actor_critic.load_state_dict(student.state_dict(), strict=False)

        print("[DAgger Eval] Starting evaluation loop...")
        eval_runner.eval(log=True)  # 调用 runner_eval.py 中的 eval

        return  # 评估结束后直接退出程序

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

        if it % cfg.save_interval == 0 or it == cfg.max_iters or it == 30:
            path = os.path.join(log_dir, f"dagger_student_{it}.pt")
            torch.save({"student_state_dict": student.state_dict(),
                        "iter": it,
                        "cfg": cfg.__dict__}, path)
            print(f"[DAgger] saved => {path}")

        if it % 50 == 0:
            print(f"\n[DAgger] Eval at iteration {it} ...")

            if eval_runner is None:
                import copy
                eval_cfg = copy.deepcopy(train_cfg)
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

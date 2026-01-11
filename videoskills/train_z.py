import os
import sys
import wandb
import argparse

# 引入项目路径
sys.path.append(os.getcwd())

from videoskills.utils import get_args, task_registry
from videoskills.utils.helpers import print_and_save_cfg, class_to_dict, parse_motion_file_path
from videoskills import LEGGED_GYM_ROOT_DIR

# 引入构建 Decoder/Prior 的辅助函数
from rsl_rl.algorithms.distill_dagger_z import build_decoder_and_prior
import torch

# A100/H100 优化
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


def get_args_z():
    """扩展标准参数，增加 vae_ckpt 参数"""
    parser = argparse.ArgumentParser(description="Train Z-Policy")
    # 复用原来的参数定义
    # 注意：这里假设 get_args 返回的是 parser 对象或者你需要手动添加
    # 为了简单，直接调用 get_args() 获取 args 对象后，再手动解析或硬编码路径也可以
    # 但推荐如下方式扩展：

    # 临时覆盖 sys.argv 解析逻辑，或者直接使用标准 get_args 并通过额外方式指定 ckpt
    args = get_args()
    return args


def train_z(args):
    # 1. 获取配置
    env_cfg, train_cfg = task_registry.get_cfgs(args)

    # 强制开启 use_z 模式 (以防配置文件没写)
    if hasattr(train_cfg, 'policy'):
        train_cfg.policy.use_z = True
    else:
        # 如果是字典格式
        train_cfg['policy']['use_z'] = True

    # 解析 motion file
    env_cfg.motion.file = parse_motion_file_path(env_cfg, train_cfg, only_failed_key=False)

    # 保存配置
    log_dir = print_and_save_cfg(env_cfg, train_cfg, filename="config_z.yaml")

    # 2. 创建环境
    # 注意：环境类必须是 LeggedRobotHoiZ (通过配置文件指定 task name 关联)
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    device = args.rl_device

    # ==========================================================================
    # 3. [关键步骤] 构建并加载预训练的 VAE (Decoder & Prior)
    # ==========================================================================
    print(f"\n[Train Z] Loading VAE components from: {args.vae_ckpt}")
    decoder, prior = build_decoder_and_prior(train_cfg, device)

    ckpt = torch.load(args.vae_ckpt, map_location=device)

    # 处理可能的 key 名字不匹配 (例如是否有 'module.' 前缀)
    decoder_state = ckpt.get('decoder_state_dict', ckpt)
    prior_state = ckpt.get('prior_state_dict', ckpt)

    decoder.load_state_dict(decoder_state)
    prior.load_state_dict(prior_state)

    # 4. [关键步骤] 冻结 VAE 参数 (Evaluation Mode + No Grad)
    # 我们只训练新的 Policy (Encoder)，保持 Latent Space 不变
    decoder.eval()
    decoder.requires_grad_(False)
    for p in decoder.parameters(): p.requires_grad = False

    prior.eval()
    prior.requires_grad_(False)
    for p in prior.parameters(): p.requires_grad = False

    # 5. [关键步骤] 将 VAE 注入到环境中
    # 环境在 step 过程中会调用 compute_z_action -> decoder
    env.set_z_decoder(decoder)
    env.set_z_prior(prior)
    print("[Train Z] VAE injected into environment and frozen.")

    # ==========================================================================
    # 6. 创建 PPO Runner
    # ==========================================================================
    # Runner 会根据 train_cfg.policy.use_z = True 自动构建输出为 z_dim 的 ActorCritic
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_dir=log_dir
    )

    # 初始化 WandB
    if args.use_wandb:
        os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)
        wandb.init(
            project=args.wandb_project,
            name=train_cfg.runner.run_name,
            dir=log_dir,
            config={**vars(args), **class_to_dict(train_cfg), **class_to_dict(env_cfg)}
        )

    # 7. 开始训练循环
    for it in range(0, train_cfg.runner.max_iterations + 1, train_cfg.runner.eval_interval):

        # 确保 VAE 始终处于 eval 模式 (防止 PPO runner 内部可能的 train() 调用影响到 VAE 的 BatchNorm)
        decoder.eval()
        prior.eval()

        # 训练
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.eval_interval, init_at_random_ep_len=False)

        result = ppo_runner.eval()
        print('Evaluation result: ', result)

        # 保存结果 keys
        success_keys = result.get("success_keys", [])
        failed_keys = result.get("failed_keys", [])

        with open(f"{ppo_runner.log_dir}/failed_keys_it{it}.txt", "w", encoding="utf-8") as f:
            for item in failed_keys:
                if isinstance(item, (list, tuple)):
                    f.write(f"{item[0]},{item[1]}\n")
                else:
                    f.write(f"{item}\n")

        with open(f"{ppo_runner.log_dir}/success_keys_it{it}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, success_keys)))

        print(f"Saved {len(success_keys)} success keys and {len(failed_keys)} failed keys.")


if __name__ == '__main__':
    # 获取参数
    args = get_args()
    train_z(args)
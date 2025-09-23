from videoskills.envs import *
from videoskills.utils import get_args, task_registry
from videoskills.utils.helpers import parse_motion_file_path
import torch

def play_hoi(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args)

    # 只给 HOI 用：确定 motion 文件（可按你现有逻辑选择失败样本或全量）
    env_cfg.motion.file = parse_motion_file_path(env_cfg, train_cfg, only_failed_key=False)

    # 播放期禁用随机项/困难地形等
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 32)
    env_cfg.early_termination.enabled = False
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    # 创建 HOI 环境（确保 args.task 指向 HOI 任务，比如 "LeggedRobotHoi" 对应的注册名）
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # 可选：手动指定要播放的 motion_ids（默认 None 则按对象类别自动采样）
    motion_ids = None
    # motion_ids = torch.arange(env.num_envs, device=env.device) % env._motion_lib.get_num_motions()

    env.play_hoi(motion_ids=motion_ids,
                 random_start=False,
                 real_time=True,
                 max_loops=1,
                 sleep_when_render=True)

if __name__ == "__main__":
    args = get_args()
    play_hoi(args)

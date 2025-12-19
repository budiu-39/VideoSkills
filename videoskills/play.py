import os
import time

from isaacgym import gymtorch, gymapi

from videoskills.utils import get_args, task_registry
from videoskills.utils.helpers import parse_motion_file_path
import torch
# ================= 🔧 播放配置 =================
PLAY_REFERENCE_MOTION = True  # True: 播放 GT 参考动作; False: 策略控制
PLAY_SPEED = 0.5  # 播放速度: 1.0=实时, 0.5=慢动作, 2.0=快进
INITIAL_MOTION_ID = 0  # 启动时默认播放的动作 ID


# ==============================================

def play(args):
    # ---------------------------------------------------------
    # 1. 配置加载与环境初始化
    # ---------------------------------------------------------
    # 使用 get_cfgs 避免未指定 run_name 时的报错
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    env_cfg.motion.file = parse_motion_file_path(env_cfg, train_cfg, only_failed_key=False)

    # 针对可视化进行环境设置覆盖
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)  # 限制环境数量以保证流畅度
    env_cfg.early_termination.enabled = False  # 关闭提前终止，保证播完

    # 简化地形与物理随机化，专注于动作观察
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    # 创建环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # ---------------------------------------------------------
    # 2. 策略与键盘事件初始化
    # ---------------------------------------------------------
    ppo_runner = None
    obs = None

    if not PLAY_REFERENCE_MOTION:
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        ppo_runner.alg.set_eval()
        obs = env.get_observations()
    else:
        train_cfg.runner.resume = False

    # 注册键盘监听：按 N 切换动作
    if env.viewer:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_N, "NEXT_MOTION")

    # ---------------------------------------------------------
    # 3. 状态管理与辅助函数
    # ---------------------------------------------------------
    num_total_motions = env._motion_lib.num_motions()

    # 使用字典来管理在闭包中修改的状态
    state = {
        "current_id": INITIAL_MOTION_ID,
        "motion_ids_tensor": None,
        "duration": 0.0,
        "total_frames": 0,
        "motion_name": "Unknown"
    }

    def get_motion_name(mid):
        """尝试从 MotionLib 中获取动作文件名"""
        lib = env._motion_lib
        # 尝试访问文件名列表
        if hasattr(lib, "_motion_files") and mid < len(lib._motion_files):
            file_path = lib._motion_files[mid]
            return os.path.splitext(os.path.basename(file_path))[0]
        # 备选方案：尝试访问名称列表
        if hasattr(lib, "get_motion_names"):
            names = lib.get_motion_names()
            if mid < len(names): return names[mid]
        return f"Motion_{mid}"

    def load_motion(mid):
        """切换动作的核心逻辑"""
        # 限制 ID 范围 (循环切换)
        mid = mid % num_total_motions
        state["current_id"] = mid
        state["motion_name"] = get_motion_name(mid)

        # 1. 创建全环境统一的 Motion ID Tensor
        motion_ids = torch.tensor([mid] * env.num_envs, device=env.device, dtype=torch.long)
        state["motion_ids_tensor"] = motion_ids

        # 2. 强制重置环境到该动作起点
        env.reset_with_motion_ids(motion_ids)

        # 3. 更新动作时长信息
        # 注意：get_motion_length 返回的是秒
        state["duration"] = env._motion_lib.get_motion_length(motion_ids)[0].item()
        state["total_frames"] = int(state["duration"] / env.dt)

        # 4. 如果是策略模式，重置后需要刷新观测
        if not PLAY_REFERENCE_MOTION:
            nonlocal obs
            obs = env.get_observations()

        # 打印切换信息
        print(f"\n{'=' * 50}")
        print(f"[INFO] Switched to Motion ID: {mid} / {num_total_motions - 1}")
        print(f"[INFO] Name: {state['motion_name']}")
        print(f"[INFO] Duration: {state['duration']:.2f}s ({state['total_frames']} frames)")
        print(f"{'=' * 50}\n")

    # 初始加载
    load_motion(state["current_id"])

    print(f"Start Playing... Mode: {'REFERENCE MOTION (GT)' if PLAY_REFERENCE_MOTION else 'POLICY'}")
    print("Press 'N' in the viewer to switch to the next motion.")

    # ---------------------------------------------------------
    # 4. 主循环
    # ---------------------------------------------------------
    while True:
        loop_start_time = time.time()

        # --- A. 键盘事件处理 ---
        if env.viewer:
            events = env.gym.query_viewer_action_events(env.viewer)
            for evt in events:
                if evt.action == "NEXT_MOTION" and evt.value > 0:  # value > 0 表示按键按下
                    load_motion(state["current_id"] + 1)

        # --- B. 获取当前进度 ---
        # 读取第一个环境的进度即可
        current_frame = env.episode_length_buf[0].item()
        current_time = current_frame * env.dt

        # --- C. 动作控制 (GT 覆盖 或 策略推理) ---
        if PLAY_REFERENCE_MOTION:
            # === Kinematic Override (强制覆盖物理状态) ===
            # 1. 覆盖关节位置和 Root 状态
            env.dof_pos[:] = env.ref_dof_pos
            env.dof_vel[:] = 0.0  # 清零速度以防物理爆炸

            env.root_states[:, 0:3] = env.ref_root_pos
            env.root_states[:, 3:7] = env.ref_root_rot
            env.root_states[:, 7:13] = 0.0

            # 2. 刷新仿真器状态
            env_ids_int32 = env.robot_actor_ids
            env.gym.set_actor_root_state_tensor_indexed(
                env.sim,
                gymtorch.unwrap_tensor(env.root_states),
                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
            )
            env.gym.set_dof_state_tensor_indexed(
                env.sim,
                gymtorch.unwrap_tensor(env.dof_state),
                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
            )

            # 3. 发送零动作 (仅为了推进时间步)
            action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        else:
            # === Policy Inference ===
            obs = ppo_runner.alg.obs_mean_std(obs)
            action = ppo_runner.alg.actor_critic.act_inference(obs)

        # --- D. 物理步进 ---
        # step 会更新时间、计算下一帧的参考动作 (ref_*) 并渲染画面
        obs, _, rews, dones, infos = env.step(action.detach())

        # --- E. UI 信息打印 ---
        progress = (current_frame / state["total_frames"]) * 100 if state["total_frames"] > 0 else 0
        info_str = f"Motion: {state['motion_name']} [{state['current_id']}]"

        # 使用 \r 回车符覆盖当前行
        print(
            f"\r{info_str:<40} | Prog: {progress:5.1f}% | Frame: {current_frame:4d}/{state['total_frames']:4d} | Time: {current_time:5.2f}s",
            end="")

        # --- F. 自动循环 ---
        # 如果超过动作长度，重置回当前动作开头
        if current_frame >= state["total_frames"] - 2:
            env.reset_with_motion_ids(state["motion_ids_tensor"])
            if not PLAY_REFERENCE_MOTION:
                obs = env.get_observations()

        # --- G. 速度控制 ---
        target_frame_time = env.dt / PLAY_SPEED
        elapsed_time = time.time() - loop_start_time
        if elapsed_time < target_frame_time:
            time.sleep(target_frame_time - elapsed_time)


if __name__ == '__main__':
    args = get_args()
    play(args)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np

from isaacgym import gymapi, gymtorch

from videoskills.utils.motionlib.motion_lib import MotionLib
from videoskills.envs.smpl.smpl_config import SMPLRobotCfg
from videoskills import LEGGED_GYM_ROOT_DIR
import torch
import time

# ----------------------------------------------------------------------
# 1. 路径和设备配置
# ----------------------------------------------------------------------
asset_path  = SMPLRobotCfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
asset_root  = os.path.dirname(asset_path)
asset_file  = os.path.basename(asset_path)
MOTION_DIR  = SMPLRobotCfg.motion.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# 2. 初始化 Isaac Gym
# ----------------------------------------------------------------------
gym = gymapi.acquire_gym()
sim_params            = gymapi.SimParams()
sim_params.dt         = 1.0 / 60.0
sim_params.up_axis    = gymapi.UP_AXIS_Z
sim_params.gravity    = gymapi.Vec3(0.0, 0.0, -9.81)

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
assert sim, "Failed to create sim"

# 地面
plane_params          = gymapi.PlaneParams()
plane_params.normal   = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# 单环境
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

# 载入 Humanoid 资产
asset_options               = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
actor_handle = gym.create_actor(env, asset, gymapi.Transform(), "humanoid", 0, 1)

# ----------------------------------------------------------------------
# 3. 收集全部 motion 文件
# ----------------------------------------------------------------------
def get_all_motion_files(directory, ext=".npy"):
    files = []
    for root, _, fns in os.walk(directory):
        files += [os.path.join(root, f) for f in fns if f.endswith(ext)]
    return sorted(files)

motion_files = get_all_motion_files(MOTION_DIR)
assert motion_files, f"No motion files found in {MOTION_DIR}"

# ----------------------------------------------------------------------
# 4. 构建 MotionLib
# ----------------------------------------------------------------------
body_names     = gym.get_asset_rigid_body_names(asset)
dof_names      = gym.get_asset_dof_names(asset)
dof_body_ids   = np.arange(1, len(body_names)).tolist()
dof_offsets    = np.linspace(0, len(dof_names), len(body_names)).astype(int)

key_body_ids = [
    gym.find_actor_rigid_body_handle(env, actor_handle, n) for n in body_names
]
motion_lib = MotionLib(
    motion_file = motion_files,
    dof_body_ids = dof_body_ids,
    dof_offsets  = dof_offsets,
    key_body_ids = torch.tensor(key_body_ids),
    device       = device,
)

tot_len = motion_lib.get_motion_length(
    torch.arange(len(motion_files), device=device)
).sum().item()
print(f"Loaded {len(motion_files)} motions ({tot_len:.2f}s total).")

# ----------------------------------------------------------------------
# 5. Viewer 与相机
# ----------------------------------------------------------------------
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
assert viewer, "Failed to create viewer"

follow_offset = np.array([0.0, -3.0, 0.7])   # 相机距离角色的相对偏移
lookat_shift  = np.array([0.0, 0.0, 0.4])   # 盯着角色胸口

# ----------------------------------------------------------------------
# 6. 播放所有 motions
# ----------------------------------------------------------------------
for m_id in range(len(motion_files)):
    motion_len = motion_lib.get_motion_length(
        torch.tensor([m_id], device=device)
    )[0].item()
    print(f"\nPlaying motion {m_id}  ({m_len:.2f}s)")

    t = 0.0
    wall_start_t = time.time()

    while not gym.query_viewer_has_closed(viewer) and t < motion_len:
        frame_start = time.time()
        state = motion_lib.get_motion_state(
            torch.tensor([m_id], device=device),
            torch.tensor([t], device=device)
        )
        root_pos, root_rot = state["root_pos"], state["root_rot"]
        root_vel, root_ang_vel = state["root_vel"], state["root_ang_vel"]
        dof_pos,  dof_vel      = state["dof_pos"],  state["dof_vel"]

        # ---- 写入 root ----
        root_ptr = gym.acquire_actor_root_state_tensor(sim)
        root_tensor = gymtorch.wrap_tensor(root_ptr).view(-1, 13)
        root_tensor[0, 0:3]   = root_pos[0]
        root_tensor[0, 3:7]   = root_rot[0]
        root_tensor[0, 7:10]  = root_vel[0]
        root_tensor[0, 10:13] = root_ang_vel[0]
        gym.set_actor_root_state_tensor_indexed(
            sim, root_ptr,
            gymtorch.unwrap_tensor(torch.tensor([0], dtype=torch.int32)),
            1
        )

        # ---- 写入 DOF ----
        ds = np.zeros(dof_pos.shape[1], dtype=[('pos', np.float32), ('vel', np.float32)])
        ds['pos'] = dof_pos[0].cpu().numpy()
        ds['vel'] = dof_vel[0].cpu().numpy()
        gym.set_actor_dof_states(env, actor_handle, ds, gymapi.STATE_ALL)

        # ---- 相机跟随 ----
        root_np = root_pos[0].cpu().numpy()
        cam_pos = gymapi.Vec3(*(root_np + follow_offset))
        cam_lookat = gymapi.Vec3(*(root_np + lookat_shift))
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_lookat)

        # ---- 模拟 + 渲染 ----
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        elapsed = time.time() - frame_start
        if elapsed < sim_params.dt:
            time.sleep(sim_params.dt - elapsed)

        t = time.time() - wall_start_t  #
        # time.sleep(sim_params.dt)  # 想放慢播放速度时可启用

# ----------------------------------------------------------------------
# 7. 清理
# ----------------------------------------------------------------------
print("\nAll motions played successfully.")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

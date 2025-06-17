import os
import time
import numpy as np

import time

from isaacgym import gymapi, gymtorch
import torch
from videoskills.utils.motionlib.motion_lib import MotionLib
from videoskills.envs.smpl.smpl_config import SMPLRobotCfg
from videoskills import LEGGED_GYM_ROOT_DIR
from scripts.poselib.skeleton.skeleton3d import SkeletonTree

# 配置路径和设备
asset_path = SMPLRobotCfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
asset_root = os.path.dirname(asset_path)
asset_file = os.path.basename(asset_path)
MOTION_DIR = SMPLRobotCfg.motion.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 Isaac Gym
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
assert sim is not None, "Failed to create sim"

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
pose = gymapi.Transform()
actor_handle = gym.create_actor(env, asset, pose, "humanoid", 0, 1)

# 加载所有 motion 文件
def get_all_motion_files(directory, ext=".npy"):
    return sorted([
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.endswith(ext)
    ])

motion_files = get_all_motion_files(MOTION_DIR)
assert len(motion_files) > 0, f"No motion files found in {MOTION_DIR}"

# 读取 asset 元信息
body_names = gym.get_asset_rigid_body_names(asset)
dof_names = gym.get_asset_dof_names(asset)
dof_body_ids = np.arange(1, len(body_names)).tolist()
dof_offsets = np.linspace(0, len(dof_names), len(body_names)).astype(int)
key_body_ids = []
for body_name in SMPLRobotCfg.motion.keybodys:
    body_id = gym.find_actor_rigid_body_handle(env, actor_handle, body_name)
    assert body_id != -1
    key_body_ids.append(body_id)

# 初始化 MotionLib
motion_lib = MotionLib(
    motion_file=motion_files,
    dof_body_ids=dof_body_ids,
    dof_offsets=dof_offsets,
    key_body_ids=key_body_ids,
    device=device
)

motion_lengths = motion_lib.get_motion_length(torch.arange(len(motion_files), device=device))
total_duration = motion_lengths.sum().item()
print(f"Loaded {len(motion_files)} motions with a total length of {total_duration:.3f}s.")

# 创建 viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
assert viewer is not None, "Failed to create viewer"

# 播放所有 motion
for motion_id in range(len(motion_files)):
    motion_len = motion_lib.get_motion_length(torch.tensor([motion_id], device=device))[0].item()
    print(f"\nPlaying motion ID {motion_id} with duration {motion_len:.2f}s")

    t = 0.0
    while not gym.query_viewer_has_closed(viewer) and t < motion_len:
        t_tensor = torch.tensor([t], device=device)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, _, _, _, _= \
            motion_lib.get_motion_state(torch.tensor([motion_id], device=device), t_tensor)

        # 设置 root state
        root_tensor_ptr = gym.acquire_actor_root_state_tensor(sim)
        root_tensor = gymtorch.wrap_tensor(root_tensor_ptr).view(-1, 13)
        root_tensor[0, 0:3] = root_pos[0]
        root_tensor[0, 3:7] = root_rot[0]
        root_tensor[0, 7:10] = root_vel[0]
        root_tensor[0, 10:13] = root_ang_vel[0]
        env_ids = torch.tensor([0], dtype=torch.int32, device="cpu")
        env_ids_tensor = gymtorch.unwrap_tensor(env_ids)
        gym.set_actor_root_state_tensor_indexed(sim, root_tensor_ptr, env_ids_tensor, 1)

        # 设置 DOF 状态
        dof_state = np.zeros(dof_pos.shape[1], dtype=[('pos', np.float32), ('vel', np.float32)])
        dof_state['pos'] = dof_pos[0].cpu().numpy()
        dof_state['vel'] = dof_vel[0].cpu().numpy()
        gym.set_actor_dof_states(env, actor_handle, dof_state, gymapi.STATE_ALL)

        # 模拟并渲染
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        t += sim_params.dt
        time.sleep(0.1)

print("\nAll motions played successfully.")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

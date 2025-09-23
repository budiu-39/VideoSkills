
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import torch
from videoskills.envs.base.legged_robot_config import LeggedRobotCfg
from videoskills.envs.base.legged_robot_imi import LeggedRobotImi
from videoskills.utils.torch_utils import to_torch
import trimesh
from videoskills.utils.motion_lib_hoi import MotionLibHoi
import time

class LeggedRobotHoi(LeggedRobotImi):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        # self.motion_file = os.listdir(self.cfg.motion.file)
        self.motion_file = self.cfg.motion.file
        self.object_name = [motion_example.split('/')[-1].split('_')[2].split('.')[0] for motion_example in self.motion_file]
        self.object_density = self.cfg.object.object_density
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        env_obj_names = [self.object_name[i % len(self.object_name)] for i in range(self.num_envs)]
        # 将名字映射到 MotionLibHoi 的词表索引
        self.env_object_ids = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
        for i, name in enumerate(env_obj_names):
            # motion_lib.object_vocab 由 motion 库构建（名字必须一致）
            obj_id = self._motion_lib.object_vocab.get(name, None)
            self.env_object_ids[i] = obj_id



    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_target(env_id, env_ptr)
        return

    def _load_motion(self, motion_file):

        self._motion_lib = MotionLibHoi(motion_file=motion_file,
                                     dof_body_ids=self.dof_body_ids,
                                     dof_offsets=self.dof_offsets,
                                     key_body_ids=self.body_ids,
                                     rotate_motion=self.cfg.motion.rotate_motion,
                                     device=self.device)


    def _create_envs(self):

        self._target_handles = []
        self._load_target_asset()
        super()._create_envs()

        return

    def _load_target_asset(self):  # smplx
        asset_root = "dataset/behave/objects_centroid_mean"
        self._target_asset = []
        points_num = []
        self.object_points = []
        for i, object_name in enumerate(self.object_name):

            asset_file = object_name + ".urdf"
            obj_file = asset_root + '/objects/' + object_name + '/' + object_name + '.obj'
            max_convex_hulls = 64
            density = self.object_density

            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01

            asset_options.density = density
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.max_convex_hulls = max_convex_hulls
            asset_options.vhacd_params.max_num_vertices_per_ch = 64
            asset_options.vhacd_params.resolution = 300000

            self._target_asset.append(self.gym.load_asset(self.sim, asset_root, asset_file, asset_options))

            mesh_obj = trimesh.load(obj_file, force='mesh')
            obj_verts = mesh_obj.vertices
            center = np.mean(obj_verts, 0)
            object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2024)

            object_points = to_torch(object_points - center)

            while object_points.shape[0] < 1024:
                object_points = torch.cat([object_points, object_points[:1024 - object_points.shape[0]]], dim=0)
            self.object_points.append(to_torch(object_points))

        self.object_points = torch.stack(self.object_points, dim=0)
        return

    def _build_target_tensors(self):
        num_actors = self.gym.get_actor_count(self.envs[0])
        self._target_states = self.root_states.view(self.num_envs, num_actors, 13)[..., 1, :]

        self.tar_actor_ids = self.robot_actor_ids + 1

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]
        return

    def _init_buffers(self):
        super()._init_buffers()
        self._build_target_tensors()

        return

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()

        target_handle = self.gym.create_actor(env_ptr, self._target_asset[env_id % len(self.object_name)], default_pose,
                                              self.object_name[env_id % len(self.object_name)], col_group, col_filter,
                                              segmentation_id)

        props = self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        for p_idx in range(len(props)):
            props[p_idx].restitution = 0.05
            props[p_idx].friction = 0.6
            props[p_idx].rolling_friction = 0.01
            props[p_idx].torsion_friction = 0.01
            if self.object_name[env_id % len(self.object_name)] == 'plasticbox' or self.object_name[
                env_id % len(self.object_name)] == 'trashcan':
                props[p_idx].rest_offset = 0.015
            else:
                props[p_idx].rest_offset = 0.002
        self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, props)

        self._target_handles.append(target_handle)
        # self.gym.set_actor_scale(env_ptr, target_handle, self.ball_size)

        return


    def _reset_robot(self, env_ids):
        super()._reset_robot(env_ids)  # 内部调用 _reset_ref_state_init
        self._reset_target(env_ids)

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        env_ids_int32 = self.tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def sample_motions(self, env_ids):
        obj_ids = self.env_object_ids[env_ids]  # [B]
        motion_ids = self._motion_lib.sample_motions_by_object(obj_ids)  # [B]
        self._sampled_motion_ids[env_ids] = motion_ids

    def _reset_target(self, env_ids):
        # 计算这些 env 对应的 motion_id 与 time
        motion_ids = self._sampled_motion_ids[env_ids]  # [B]
        motion_times = self._motion_start_times[env_ids] + self.episode_length_buf[env_ids] * self.dt  # [B]

        hoi = self._motion_lib.get_hoi_state(motion_ids, motion_times)

        self._target_states[env_ids, :3] = hoi["obj_pos"] + self.pos_offset['root'][env_ids]
        self._target_states[env_ids, 3:7] = hoi["obj_rot"]  # xyzw
        self._target_states[env_ids, 7:10] = hoi["obj_pos_vel"]
        self._target_states[env_ids, 10:13] = hoi["obj_rot_vel"]
        return


    @torch.no_grad()
    def play_hoi(self,
                 motion_ids: torch.Tensor = None,
                 random_start: bool = False,
                 real_time: bool = True,
                 max_loops: int = 1,
                 sleep_when_render: bool = True):
        """
        HOI 专用播放（仅按参考轨迹逐帧写入人物与物体状态，不走策略网络）。

        Args:
            motion_ids: [num_envs] 每个 env 播放的轨迹 ID。None 则按对象类别采样。
            random_start: True 从随机时间开始；False 从 0 开始。
            real_time: True 尽量按 dt 实时播放（需渲染）；False 尽快跑。
            max_loops: 轨迹播完循环次数。
            sleep_when_render: 渲染时是否用 sleep 做节流。
        """
        # 播放期：固定、可复现
        self.eval_mode = True
        self.early_termination = False

        env_ids = torch.arange(self.num_envs, device=self.device)

        # 1) 选择要播的 motion
        if motion_ids is None:
            # HOI: 按对象类别采样对应的动作
            self.sample_motions(env_ids)                 # -> 填充 self._sampled_motion_ids
            motion_ids = self._sampled_motion_ids.clone()
        else:
            assert motion_ids.shape[0] == self.num_envs
            self._sampled_motion_ids[:] = motion_ids

        # 2) 初始化到起始帧
        self.reset_with_motion_ids(motion_ids, random=random_start)
        # 清掉提前终止/timeout
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0
        self.early_termination_buf[:] = 0

        # 3) 获取各 env 轨迹时长(秒)，确定一次 loop 的步数
        motion_lens = self._motion_lib.get_motion_length(motion_ids)    # [num_envs]
        steps_per_loop = int(float(motion_lens.max().item()) / self.dt) + 1

        t0 = time.perf_counter()

        for loop in range(max_loops):
            # 从头来一遍
            self.episode_length_buf[:] = 0
            self._motion_start_times[:] = 0.0

            for step_idx in range(steps_per_loop):
                # 当前参考时间
                progress = (self.episode_length_buf.to(torch.float) + 1) * self.dt     # [num_envs]
                motion_times = progress + self._motion_start_times                     # [num_envs]

                # a) 取人物参考状态
                motion_state = self._motion_lib.get_motion_state(self._sampled_motion_ids, motion_times)
                # b) 取物体参考状态（HOI）
                hoi = self._motion_lib.get_hoi_state(self._sampled_motion_ids, motion_times)

                # c) 写人物状态（root / dof / key bodies）
                self._set_env_state(
                    env_ids=env_ids,
                    root_pos=motion_state["root_pos"] + self.pos_offset['root'],
                    root_rot=motion_state["root_rot"],
                    dof_pos=motion_state["dof_pos"],
                    root_vel=motion_state["root_vel"],
                    root_ang_vel=motion_state["root_ang_vel"],
                    dof_vel=motion_state["dof_vel"],
                    key_pos=motion_state["key_pos"] + self.pos_offset['body'],
                    key_rot=motion_state["key_rot"],
                    key_vel=motion_state["key_vel"],
                    key_ang_vel=motion_state["key_ang_vel"]
                )

                # d) 写物体状态（到缓存 _target_states）
                self._target_states[env_ids, :3]  = hoi["obj_pos"] + self.pos_offset['root'][env_ids]
                self._target_states[env_ids, 3:7] = hoi["obj_rot"]          # xyzw
                self._target_states[env_ids, 7:10]  = hoi["obj_pos_vel"]
                self._target_states[env_ids, 10:13] = hoi["obj_rot_vel"]

                # e) 把缓存写回 sim（注意 dof_pos contiguous）
                self._reset_env_tensors(env_ids)   # LeggedRobotHoi 覆盖版会同时推 actor_root（人物+物体）

                # f) 推仿真 + 渲染（为显示；物理不会主导状态）
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.render()

                # g) 推进内部计步器
                self.episode_length_buf += 1

                # h) 实时节流
                if real_time and sleep_when_render and not self.headless:
                    target_elapsed = (loop * steps_per_loop + step_idx + 1) * self.dt
                    now = time.perf_counter() - t0
                    remain = target_elapsed - now
                    if remain > 0:
                        time.sleep(min(remain, self.dt))

        # 让最后一帧停顿一下（看清）
        if not self.headless:
            time.sleep(0.5)

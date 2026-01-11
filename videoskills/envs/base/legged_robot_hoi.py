
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
from videoskills.utils.torch_utils import calc_heading_quat_inv, calc_heading_quat, quat_to_tan_norm, quat_rotate
from videoskills.utils.torch_utils import to_torch, quat_mul, quat_conjugate, quat_to_angle_axis
from videoskills.envs.base.legged_robot_imi import (
    compute_humanoid_observations_jit,
)


class LeggedRobotHoi(LeggedRobotImi):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        # self.motion_file = os.listdir(self.cfg.motion.file)
        self.motion_file = self.cfg.motion.file
        # TODO: Hacky code here for Behave dataset
        if self.cfg.asset.load_object:
            self.mask_interaction_reward = False
            # self.object_name = [motion_example.split('/')[-1].split('_')[2].split('.')[0] for motion_example in self.motion_file]
            self.object_name = [motion_example.split('/')[-1].split('_')[1].split('.')[0] for motion_example in self.motion_file]
        else:
            self.mask_interaction_reward = True
            self.object_name = ["none" for _ in range(len(self.motion_file))]
        self.object_density = self.cfg.object.object_density
        self.reward_weights = self.cfg.rewards.weight
        self.et_counter = {
            "robot": 0,
            "object": 0,
            "ig": 0,
            "contact": 0,
            "total": 0,
        }
        self.reward_subterm_sums = {}
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        env_obj_names = [self.object_name[i % len(self.object_name)] for i in range(self.num_envs)]
        # 将名字映射到 MotionLibHoi 的词表索引
        self.env_object_ids = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
        for i, name in enumerate(env_obj_names):
            # motion_lib.object_vocab 由 motion 库构建（名字必须一致）
            obj_id = self._motion_lib.object_vocab.get(name, None)
            self.env_object_ids[i] = obj_id

        # Physics Init Buffer Variables
        self.physics_state_buffer = None
        self.physics_state_scores = None  # 新增：用于存储分数的 Buffer
        self.physics_init_candidates = 3  # Parameter: candidate number per frame
        self.physics_buffer_initialized = False


    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        if self.cfg.asset.load_object:
            self._build_obj(env_id, env_ptr)
        return

    def _load_motion(self, motion_file):
        self._motion_lib = MotionLibHoi(motion_file=motion_file,
                                     dof_body_ids=self.dof_body_ids,
                                     dof_offsets=self.dof_offsets,
                                     key_body_ids=self.body_ids,
                                     rotate_motion=self.cfg.motion.rotate_motion,
                                     device=self.device)

    def dof_to_body(dof_name: str) -> str:
        if len(dof_name) >= 2 and dof_name[-2:] in ("_x", "_y", "_z"):
            return dof_name[:-2]
        return dof_name

    def _create_envs(self):

        self._obj_handles = []
        self._load_obj_asset()
        super()._create_envs()
        return

    def _load_obj_asset(self):
        if not self.cfg.asset.load_object:
            # Stage 1: 创建一个虚拟的空点云，保证维度对齐 [1, 1024, 3]
            self.object_points = torch.zeros((1, 1024, 3), device=self.device)
            # self.object_name = ["none"]
            return

        asset_root = self.cfg.asset.asset_root

        # --- 修复核心：不再依赖 self._motion_lib ---
        # 直接从 self.object_name (在 __init__ 已生成) 推导唯一物体列表
        # 使用 sorted 确保顺序确定，这通常能与 MotionLib 的 vocab ID 对应上
        unique_obj_names = sorted(list(set(self.object_name)))
        num_unique_objs = len(unique_obj_names)

        # 构建临时映射：Name -> Index (用于构建 _obj_asset 列表)
        name_to_idx = {name: i for i, name in enumerate(unique_obj_names)}

        unique_assets = []
        unique_points = []

        print(f"Loading {num_unique_objs} unique objects from {len(self.object_name)} motions...")

        # 2. 只加载唯一的物体资源
        for i, object_name in enumerate(unique_obj_names):

            # --- Asset 加载 ---
            asset_file = object_name + ".urdf"
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

            # 加载一次 Asset
            asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            unique_assets.append(asset)

            # --- Points 加载 ---
            obj_file = asset_root + '/objects/' + object_name + '/' + object_name + '.obj'
            mesh_obj = trimesh.load(obj_file, force='mesh')
            obj_dir = os.path.dirname(obj_file)
            points_cache_path = os.path.join(obj_dir, 'sampled_points.pt')

            if os.path.exists(points_cache_path):
                pts = torch.load(points_cache_path, map_location=self.device)
            else:
                pts, _ = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2025)
                pts = to_torch(pts, device=self.device)
                torch.save(pts, points_cache_path)

            # 补齐点数
            while pts.shape[0] < 1024:
                pts = torch.cat([pts, pts[:1024 - pts.shape[0]]], dim=0)

            unique_points.append(pts)

        # 3. 生成 self.object_points [Num_Unique, 1024, 3]
        self.object_points = torch.stack(unique_points, dim=0)

        # 4. 生成 self._obj_asset
        # 为每个 motion 匹配对应的 unique asset 引用
        self._obj_asset = []
        for name in self.object_name:
            idx = name_to_idx[name]
            self._obj_asset.append(unique_assets[idx])

        return

    def _build_obj_tensors(self):
        if self.cfg.asset.load_object:
            num_actors = self.gym.get_actor_count(self.envs[0])
            self.obj_states = self.root_states.view(self.num_envs, num_actors, 13)[..., 1, :]
            self.obj_pos = self.obj_states[..., 0:3]
            self.obj_quat = self.obj_states[..., 3:7]
            self.obj_vel = self.obj_states[..., 7:10]
            self.obj_ang_vel = self.obj_states[..., 10:13]
            self.tar_actor_ids = self.robot_actor_ids + 1
        else:
            # AMASS 阶段：创建一个全零的伪 Tensor，保证代码不崩
            self.obj_states = torch.zeros((self.num_envs, 13), device=self.device)
            self.obj_pos = self.obj_states[:, 0:3]
            self.obj_quat = self.obj_states[:, 3:7]
            self.obj_quat[:, 3] = 1.0  # 单位四元数
            self.obj_vel = self.obj_states[:, 7:10]
            self.obj_ang_vel = self.obj_states[:, 10:13]
            self.tar_actor_ids = self.robot_actor_ids
        # bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        # contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        # contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        # self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]


        return

    def _init_buffers(self):
        self.contact_reset = torch.zeros((self.num_envs,2), device=self.device, dtype=torch.long)
        self.kinematic_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ig_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ig = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.body_contact = torch.zeros((self.num_envs, self.num_bodies), device=self.device, dtype=torch.float)

        super()._init_buffers()
        self._build_obj_tensors()

        return

    def _build_obj(self, env_id, env_ptr):
        col_group = env_id
        col_filter = -1
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p = gymapi.Vec3(self.env_origins[env_id, 0], self.env_origins[env_id, 1], 1.0)

        obj_handle = self.gym.create_actor(env_ptr, self._obj_asset[env_id % len(self.object_name)], default_pose,
                                              self.object_name[env_id % len(self.object_name)], col_group, col_filter,
                                              segmentation_id)

        props = self.gym.get_actor_rigid_shape_properties(env_ptr, obj_handle)
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
        self.gym.set_actor_rigid_shape_properties(env_ptr, obj_handle, props)

        self._obj_handles.append(obj_handle)

        return


    def _reset_robot(self, env_ids):
        super()._reset_robot(env_ids)  # 内部调用 _reset_ref_state_init
        self._reset_obj(env_ids)

        # self.contact_reset[env_ids] = 0
        # self.kinematic_reset[env_ids] = 0
        # self.early_termination_buf[env_ids] = 0

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

    def _reset_obj(self, env_ids):
        # 计算这些 env 对应的 motion_id 与 time
        if self._state_init == 'physical' and not self.eval_mode:
            return

        motion_ids = self._sampled_motion_ids[env_ids]  # [B]
        motion_times = self._motion_start_times[env_ids] # + self.episode_length_buf[env_ids] * self.dt  # [B]

        hoi_state = self._motion_lib.get_hoi_state(motion_ids, motion_times)


        self.obj_pos[env_ids] =  hoi_state["obj_pos"] + self.pos_offset['root'][env_ids]
        self.obj_quat[env_ids] = hoi_state["obj_rot"]  # xyzw
        self.obj_vel[env_ids] = hoi_state["obj_pos_vel"]
        self.obj_ang_vel[env_ids] = hoi_state["obj_rot_vel"]
        # self.ref_ig[env_ids] = hoi_state["ig"]
        # self.body_contact[env_ids] = hoi_state["contact_robot"]

        return

    def compute_observations(self):
        # 有 4 类 obs：humanoid, mimic(for humanoid), obj(相对于人类）, interaction
        # TODO：思考一下，这些有必要写成 self 吗？好像没必要，可以节省更多显存。哦哦哦哦有些是有必要写的，为了 rollout
        progress = (self.episode_length_buf.to(torch.float) + 1) * self.dt
        motion_times = progress + self._motion_start_times
        motion_state = self._motion_lib.get_motion_state(self._sampled_motion_ids, motion_times)
        hoi_state = self._motion_lib.get_hoi_state(self._sampled_motion_ids, motion_times)

        self.ref_root_pos[:] =  motion_state["root_pos"] + self.pos_offset['root']
        self.ref_root_rot[:] = motion_state["root_rot"]
        self.ref_dof_pos[:] = motion_state["dof_pos"]
        self.ref_body_pos = motion_state["key_pos"] + self.pos_offset['body']
        self.ref_body_rot = motion_state["key_rot"]
        self.ref_body_vel = motion_state["key_vel"]
        self.ref_body_ang_vel = motion_state["key_ang_vel"]
        self.ref_obj_pos = hoi_state["obj_pos"] + self.pos_offset['root']
        self.ref_obj_rot = hoi_state["obj_rot"]
        self.ref_obj_pos_vel = hoi_state["obj_pos_vel"]
        self.ref_obj_rot_vel = hoi_state["obj_rot_vel"]
        self.ref_ig = hoi_state["ig"]
        self.ref_contact= hoi_state["contact_robot"]

        # 要做的是 环境数 * 人体点数 * 3   对于物体做一个 view 先把变成 环境数 * 物体点数，3
        # ref_obj_rot_extend = hoi_state['obj_rot'].unsqueeze(1).repeat(1, self.object_points.shape[1], 1).view(-1, 4)
        object_points_extend = self.object_points[self.env_object_ids].unsqueeze(0).view(-1, 3)
        # ref_obj_points = (quat_rotate(ref_obj_rot_extend, object_points_extend).view(self.num_envs, -1, 3)
        #               + self.ref_obj_pos.unsqueeze(1))

        # ref_ig = compute_sdf(self.ref_body_pos.view(-1, 52, 3), ref_obj_points).view(-1, 3)
        # self.ref_ig = ref_ig.detach().view(self.num_envs, -1, 3)

        obj_rot_extend = self.obj_quat.unsqueeze(1).repeat(1, self.object_points.shape[1], 1).view(-1, 4)
        obj_points = (quat_rotate(obj_rot_extend, object_points_extend).view(self.num_envs, -1, 3) + self.obj_pos.unsqueeze(1))
        ig = -compute_sdf(self.body_pos.view(-1, 52, 3), obj_points).view(-1, 3)  # 人到物体

        self.ig = ig.detach().view(self.num_envs, -1, 3)


        humanoid_obs = compute_humanoid_observations_jit(self.base_pos, self.base_quat,
                                                         self.body_pos, self.body_rot, self.body_vel, self.body_ang_vel,
                                                         activate_quat_to_tan_norm=self.activate_quat_to_tan_norm)

        mimic_obs = self.compute_mimic_observations()

        if self.cfg.asset.load_object:

            obj_obs = compute_obj_observations_jit(self.base_pos, self.base_quat, self.obj_states,
                                                   self.ref_obj_pos, self.ref_obj_rot, self.ref_obj_pos_vel, self.ref_obj_rot_vel)

            hoi_obs = compute_hoi_observation_jit(self.base_quat.unsqueeze(1).repeat(1, self.num_bodies, 1), self.ref_ig.view(-1, 3)
                                                  , ig)
            self.body_contact = torch.any(torch.abs(self.contact_forces[:, self.body_ids]) > 0.1, dim=-1).float()
        else:
            obj_obs = torch.zeros((self.num_envs, 21), device=self.device) # 根据你的 JIT 函数计算出的维度
            hoi_obs = torch.zeros((self.num_envs, self.num_bodies * 6), device=self.device)
            self.body_contact[:] = 0.0


        self.obs_buf = torch.cat((humanoid_obs, mimic_obs, obj_obs, hoi_obs, self.body_contact, self.actions), dim=-1)



    # @torch.no_grad()
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
        self.eval_mode = False
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
        # self.reset_with_motion_ids(motion_ids, random=random_start)
        # 清掉提前终止/timeout
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0
        self.early_termination_buf[:] = 0

        # 3) 获取各 env 轨迹时长(秒)，确定一次 loop 的步数
        motion_lens = self._motion_lib.get_motion_length(motion_ids)    # [num_envs]
        steps_per_loop = int(float(motion_lens.max().item()) / self.dt) + 1

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

                # self._reset_obj(env_ids)   # 直接写入物体状态
                # d) 写物体状态（到缓存 _target_states）
                self.obj_states[env_ids, :3]  = hoi["obj_pos"] + self.pos_offset['root'][env_ids]
                self.obj_states[env_ids, 3:7] = hoi["obj_rot"]          # xyzw
                self.obj_states[env_ids, 7:10]  = hoi["obj_pos_vel"]
                self.obj_states[env_ids, 10:13] = hoi["obj_rot_vel"]
                self.ig[env_ids] = hoi['ig']
                self.body_contact[env_ids] = hoi['contact_robot']

                # e) 把缓存写回 sim（注意 dof_pos contiguous）
                self._reset_env_tensors(env_ids)   # LeggedRobotHoi 覆盖版会同时推 actor_root（人物+物体）

                # f) 推仿真 + 渲染（为显示；物理不会主导状态）
                self.gym.simulate(self.sim)
                # self.gym.fetch_results(self.sim, True)
                # if (step_idx % 4) == 0:
                self.gym.step_graphics(self.sim)
                self.gym.clear_lines(self.viewer)
                self.render()

                # g) 推进内部计步器
                self.episode_length_buf += 1

                # h) 实时节流
                # if real_time and sleep_when_render and not self.headless:
                #     target_elapsed = (loop * steps_per_loop + step_idx + 1) * self.dt
                #     now = time.perf_counter() - t0                #     remain = target_elapsed - now
                #     if remain > 0:
                #         time.sleep(min(remain, self.dt))

        # 让最后一帧停顿一下（看清）
        if not self.headless:
            time.sleep(0.5)


    def render(self):
        if not self.headless:
            max_vis_envs = self.num_envs
            for i in range(max_vis_envs):
                env_ptr = self.envs[i]  # 当前环境的指针

                body_pos_env = self.body_pos[i].detach().cpu().numpy()  # (52, 3)
                num_lines = body_pos_env.shape[0]

                # ig_env = self.ig[i].cpu().numpy()  # (52, 3)
                # obj_near_env = body_pos_env + ig_env
                # #
                # verts = np.empty((num_lines * 2, 3), dtype=np.float32)
                # # #
                # verts[0::2] = body_pos_env
                # verts[1::2] = obj_near_env
                #
                # # 颜色（蓝色）
                # colors = np.tile(np.array([[0.2, 0.2, 1.0]], dtype=np.float32), (num_lines * 2, 1))
                # self.gym.add_lines(self.viewer, env_ptr, num_lines, verts, colors)

                # 如果要渲染参考 ig
                self.ref_ig[i].cpu().numpy()  # (52, 3)
                obj_near_ref = self.ref_body_pos[i].detach().cpu().numpy() + self.ref_ig[i].cpu().numpy()

                verts_ref = np.empty((num_lines * 2, 3), dtype=np.float32)
                verts_ref[0::2] = body_pos_env
                verts_ref[1::2] = obj_near_ref
                colors_ref = np.tile(np.array([[1.0, 0.2, 0.2]], dtype=np.float32), (num_lines * 2, 1))
                self.gym.add_lines(self.viewer, env_ptr, num_lines, verts_ref, colors_ref)

                for j in range(self.num_bodies):
                    # if j in self.body_no_hand_ids:
                    #     self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL,
                    #                                   gymapi.Vec3(1., 0., 0.))
                    if self.body_contact[i, j] > 0:
                        self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL,
                                                      gymapi.Vec3(1., 0., 0.))
                    elif self.body_contact[i, j] < 0:
                        self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL,
                                                      gymapi.Vec3(0., 0., 1.))
                    else:
                        self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL,
                                                      gymapi.Vec3(1., 1., 1.))

        super().render()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        time_out = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        progress = (self.episode_length_buf.to(torch.float) + 1) * self.dt  # progress is the current ref_motion
        motion_lens = self._motion_lib.get_motion_length(self._sampled_motion_ids)
        ref_out = (progress + self._motion_start_times ) > motion_lens

        # --------- ③ 汇总三个条件 ----------
        # self.reset_buf = fall | time_out | body_too_far
        if not self.cfg.asset.load_object:
            self.object_reset = torch.zeros_like(self.robot_reset)
            self.ig_reset = torch.zeros_like(self.robot_reset)
        kinematic_reset = torch.logical_or(self.robot_reset, self.object_reset)
        self.kinematic_reset = torch.logical_or(self.ig_reset, kinematic_reset)
        contact_fail = torch.any(self.contact_reset > 10, dim=-1) & (self.episode_length_buf > 1)
        self.early_termination_buf = self.kinematic_reset | contact_fail
        self.time_out_buf = time_out | ref_out
        self.reset_buf = self.time_out_buf | self.early_termination_buf

        if not self.early_termination:
            self.reset_buf = time_out | ref_out
            self.time_out_buf = time_out | ref_out

        if self.eval_mode:
            self.reset_buf = ref_out | self.early_termination_buf
            if not self.early_termination:
                self.reset_buf = ref_out
            self.time_out_buf = time_out | ref_out

        if torch.any(self.early_termination_buf):
            total_et = torch.sum(self.early_termination_buf).item()
            self.et_counter["total"] += total_et
            self.et_counter["robot"] += torch.sum(self.robot_reset).item()
            self.et_counter["object"] += torch.sum(self.object_reset).item()
            self.et_counter["ig"] += torch.sum(self.ig_reset).item()
            self.et_counter["contact"] += torch.sum(contact_fail).item()


    def compute_reward(self):
        # TODO: just for develop now
        rew_h = self._reward_humanoid()

        if self.mask_interaction_reward:
            # Stage 1: 屏蔽物体奖励
            rew_obj = torch.ones(self.num_envs, device=self.device)
            rew_ig = torch.ones(self.num_envs, device=self.device)
            rew_cg = torch.ones(self.num_envs, device=self.device)
        else:
            # Stage 2/3: 逐步开启
            rew_obj = self._reward_obj()
            rew_ig = self._reward_ig()
            rew_cg = self._reward_cg()

        if self.mask_interaction_reward:
            self.episode_sums["obj"].zero_()  # 修复点：使用 zero_() 保持张量类型
            self.episode_sums["ig"].zero_()
            self.episode_sums["cg"].zero_()
        else:
            self.episode_sums["obj"] += rew_obj
            self.episode_sums["ig"] += rew_ig
            self.episode_sums["cg"] += rew_cg
        self.episode_sums["humanoid"] += rew_h

        # 最终奖励相乘或相加
        self.rew_buf[:] = rew_obj * rew_ig * rew_cg * rew_h
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        return


    def _reward_imitation(self):
        # reward 是不需要 heading 归一化 的！
        """
        Computes the imitation reward based on the difference between the current and reference body positions and rotations.
        The reward is computed in the heading frame of the root body.
        """
        body_nohand = self.no_hand_body_mask

        pos_err = torch.mean(torch.square(self.body_pos[:, body_nohand] - self.ref_body_pos[:, body_nohand]), dim=1).mean(-1)
        # pos_err = torch.norm(self.body_pos - self.ref_body_pos, p=2, dim=-1).mean(dim=1)
        rot_diff = quat_mul(self.ref_body_rot[:, body_nohand] , quat_conjugate(self.body_rot[:, body_nohand] ))
        diff_global_body_angle = quat_to_angle_axis(rot_diff)[0]
        rot_err = (diff_global_body_angle ** 2).mean(dim=-1)
        vel_err = torch.mean(torch.square(self.body_vel[:, body_nohand]  - self.ref_body_vel[:, body_nohand] ), dim=1).mean(-1)
        ang_vel_err = torch.mean(torch.square(self.body_ang_vel[:, body_nohand]  - self.ref_body_ang_vel[:, body_nohand] ), dim=1).mean(-1)
        self.robot_reset = (self.ref_body_pos[:, body_nohand] - self.body_pos[:, body_nohand]).norm(dim=-1).mean(
            dim=-1) > 0.5
        self.robot_reset *= (self.episode_length_buf > 1)
        # Compute the reward as a weighted sum of the errors
        reward_pos = self.cfg.rewards.task_w.w_pos * torch.exp(-self.cfg.rewards.task_w.k_pos * pos_err)
        reward_rot = self.cfg.rewards.task_w.w_rot * torch.exp(-self.cfg.rewards.task_w.k_rot * rot_err)
        reward_vel = self.cfg.rewards.task_w.w_vel * torch.exp(-self.cfg.rewards.task_w.k_vel * vel_err)
        reward_ang_vel = self.cfg.rewards.task_w.w_ang_vel * torch.exp(-self.cfg.rewards.task_w.k_ang_vel * ang_vel_err)

        reward = reward_pos + reward_rot + reward_vel + reward_ang_vel
        self.extras['reward_pos'] = reward_pos
        self.extras['reward_rot'] = reward_rot
        self.extras['reward_vel'] = reward_vel
        self.extras['reward_ang_vel'] = reward_ang_vel
        self.extras['pos_err'] = pos_err
        self.extras['rot_err'] = rot_err
        self.extras['vel_err'] = vel_err
        self.extras['ang_vel_err'] = ang_vel_err

        return reward


    def _reward_humanoid(self):
        # body pos reward
        w = self.reward_weights
        body_nohand = self.no_hand_body_mask
        dof_nohand = self.no_hand_dof_mask

        ref_ig_norm = self.ref_ig.norm(dim=-1)
        weight_h = (-5 * ref_ig_norm).exp()
        weight_hp = weight_h.clone().detach()
        # 这里好像有问题!
        ankle_toe_ids = [i for i in range(self.num_bodies) if 'Ankle' in self.body_names[i] or 'Toe' in self.body_names[i]]
        weight_hp[:, ankle_toe_ids] = 1

        ep = torch.mean(((self.ref_body_pos[:, body_nohand] - self.body_pos[:, body_nohand]) ** 2).sum(dim=-1)
                        * weight_hp[:, body_nohand], dim=-1)
        rp = torch.exp(-ep * w.p)

        diff_quat_data = normalize(quat_mul(quat_conjugate(self.ref_body_rot[:, body_nohand].reshape(-1, 4)),
                                                   self.body_rot[:, body_nohand].reshape(-1, 4)))
        diff_angle, diff_axis = quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, sum(body_nohand))
        weight_hr = 1 - weight_h

        er = torch.mean(diff[:, :] * weight_hr[:, body_nohand], dim=-1)
        rr = torch.exp(-er * w.r)

        # body pos vel reward
        epv = torch.mean((self.ref_body_vel[:, body_nohand] - self.body_vel[:, body_nohand]).view(self.num_envs, -1) ** 2, dim=-1)
        rpv = torch.exp(-epv * w.pv)

        # body rot vel reward
        erv = torch.mean((self.body_ang_vel[:, body_nohand] - self.ref_body_ang_vel[:, body_nohand]).view(self.num_envs, -1) ** 2, dim=-1)
        rrv = torch.exp(-erv * w.rv)

        # energy penalty
        energy = torch.abs(torch.multiply(self.dof_force_tensor[:, dof_nohand], self.dof_vel[:, dof_nohand])).sum(dim=-1)
        energy = energy.mul(-w.eg1).exp()

        rb = rp * rr * rpv * rrv * energy
        self.robot_reset = (self.ref_body_pos[:, body_nohand] - self.body_pos[:, body_nohand]).norm(dim=-1).mean(dim=-1) > 0.5
        self.robot_reset *= (self.episode_length_buf > 1)

        self._accum_subterm("Humanoid/PositionTerm", rp)
        self._accum_subterm("Humanoid/RotationTerm", rr)
        self._accum_subterm("Humanoid/LinearVelocityTerm", rpv)
        self._accum_subterm("Humanoid/AngularVelocityTerm", rrv)
        self._accum_subterm("Humanoid/EnergyTerm", energy)

        return rb

    def _reward_obj(self):
        w = self.reward_weights
        heading_rot = calc_heading_quat_inv(self.base_quat)
        ref_heading_rot = calc_heading_quat_inv(self.ref_root_rot)

        local_obj_pos = self.obj_pos - self.base_pos
        local_obj_pos[..., -1] = self.obj_pos[..., -1]
        local_obj_pos = quat_rotate(heading_rot, local_obj_pos)


        ref_local_obj_pos = self.ref_obj_pos - self.ref_root_pos
        ref_local_obj_pos[..., -1] =  self.ref_obj_pos[..., -1]
        ref_local_obj_pos = quat_rotate(ref_heading_rot, ref_local_obj_pos)
        eop = torch.mean(((ref_local_obj_pos - local_obj_pos) ** 2), dim=-1)  # * (1 - weight_h.max(dim=-1)[0])
        rop = torch.exp(-eop * w.op)

        # object rot reward
        local_obj_rot = quat_mul(heading_rot, self.obj_quat)
        ref_local_obj_rot = quat_mul(ref_heading_rot, self.ref_obj_rot)
        diff_quat_data = normalize(quat_mul(quat_conjugate(ref_local_obj_rot), local_obj_rot))
        diff_angle, diff_axis = quat_to_angle_axis(diff_quat_data)
        eor = diff_angle
        ror = torch.exp(-eor * w.obj_r)
        # 可能的改进 用二次/余弦形式更平滑
        # eor = torch.mean(diff_angle ** 2, dim=-1)  # 或 1 - cos(diff_angle)
        # ror = torch.exp(-w.obj_r * eor)

        # object pos vel reward
        local_obj_vel = quat_rotate(heading_rot, self.obj_vel)
        ref_local_obj_vel = quat_rotate(ref_heading_rot, self.ref_obj_pos_vel)
        eopv = torch.mean((ref_local_obj_vel - local_obj_vel) ** 2, dim=-1)
        ropv = torch.exp(-eopv * w.opv)

        local_obj_angvel = quat_rotate(heading_rot, self.obj_ang_vel)
        ref_local_obj_angvel = quat_rotate(ref_heading_rot, self.ref_obj_rot_vel)
        eorv = torch.mean((ref_local_obj_angvel - local_obj_angvel) ** 2, dim=-1)
        rorv = torch.exp(-eorv * w.orv)

        # v2 = (self.obj_vel ** 2).sum(dim=-1)  # ||v||^2
        # w2 = (self.obj_ang_vel ** 2).sum(dim=-1)  # ||ω||^2
        # obj_energy = torch.exp(- (w.eg2_lin * v2 + w.eg2_ang * w2))
        # obj_energy = obj_energy.mul(-w.eg2).exp()
        ro = rop * ror * ropv * rorv

        pos_thresh = 0.50  # 位置阈值（米），按需要调整
        rot_thresh = 1.0  # 朝向阈值（弧度），~57°，按需要调整

        pos_err = torch.norm(ref_local_obj_pos - local_obj_pos, dim=-1)  # [N]
        dq = normalize(quat_mul(quat_conjugate(ref_local_obj_rot), local_obj_rot))  # 相对四元数
        angle_err = quat_to_angle_axis(dq)[0] # [N,3] 的角轴角度向量 # 角度幅值（弧度）

        self.object_reset = torch.logical_or(pos_err > pos_thresh, angle_err > rot_thresh)
        self.object_reset *= (self.episode_length_buf > 1)

        self._accum_subterm("Object/PositionTerm", rop)
        self._accum_subterm("Object/RotationTerm", ror)
        self._accum_subterm("Object/LinearVelocityTerm", ropv)
        self._accum_subterm("Object/AngularVelocityTerm", rorv)
        # self._accum_subterm("Object/EnergyTerm", obj_energy)

        return ro

    def _reward_ig(self):
        w = self.reward_weights
        body_nohand = self.no_hand_body_mask

        ig = self.ig[:, body_nohand]
        ref_ig = self.ref_ig[:, body_nohand]
        ### interaction graph reward ###
        with torch.no_grad():
            weight_1 = (1 / torch.clamp((ig**2).sum(dim=-1), min=0.01))
            weight_1 = weight_1 / (weight_1.sum(dim=-1, keepdim=True) + 1e-8)
            weight_2 = (1 / torch.clamp((ref_ig**2).sum(dim=-1), min=0.01))
            weight_2 = weight_2 / (weight_2.sum(dim=-1, keepdim=True) + 1e-8)

        sq_err = ((ig - ref_ig) ** 2).sum(dim=-1)  # (N, B)
        eig = sq_err * (weight_1 + weight_2)  # (N, B)
        eig_env = eig.mean(dim=-1)  # (N,)  用 mean 使尺度与 B 无关

        # --- reward ---
        rig = torch.exp(-0.5 * w.ig * eig_env)  # (N,)

        # --- reset（每个 env 独立；避免双 max 变成全局）---
        rel_err_ref = sq_err.sqrt() / torch.clamp((ref_ig ** 2).sum(dim=-1).sqrt(), min=0.5)  # (N, B)
        rel_err_cur = sq_err.sqrt() / torch.clamp((ig ** 2).sum(dim=-1).sqrt(), min=0.5)  # (N, B)

        th = 4.0
        reset_ig_1 = rel_err_ref.mean(dim=-1) > th  # (N,)
        reset_ig_2 = rel_err_cur.mean(dim=-1) > th  # (N,)
        self.ig_reset = torch.logical_or(reset_ig_1, reset_ig_2) & (self.episode_length_buf > 1)

        # --- 日志：分开记录距离与奖励 ---
        self._accum_subterm("InteractionGraph/WeightedSquaredErrorMean", eig_env)  # 距离

        return rig

    def _reward_cg(self):
        # TODO: 公式还不太能理解
        contact_thres = 0.1
        w = self.reward_weights
        ref_human_contact = self.ref_contact  # 0, 1, -1
        human_contact = self.body_contact[:, self.body_ids]
        left_contact_hand_ids = list(range(17, 33))

        ref_left_contact_hand = ref_human_contact[:, left_contact_hand_ids]
        ref_left_contact_hand_any = torch.any(ref_left_contact_hand > contact_thres, dim=-1).float()
        left_hand_contact = human_contact[:, left_contact_hand_ids].clone()
        left_hand_contact_any = torch.any(left_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        ecg_left = (((ref_left_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(
            left_hand_contact - ref_left_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        rcg_left = 0.5 * (1 + torch.exp(-ecg_left * w.cg_hand)) * (ref_left_contact_hand_any) + (
                    1 - ref_left_contact_hand_any)

        right_contact_hand_ids = list(range(36, 52))

        ref_right_contact_hand = ref_human_contact[:, right_contact_hand_ids]
        ref_right_contact_hand_any = torch.any(ref_right_contact_hand > contact_thres, dim=-1).float()
        right_hand_contact = human_contact[:, right_contact_hand_ids].clone()
        right_hand_contact_any = torch.any(right_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        ecg_right = (((ref_right_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(
            right_hand_contact - ref_right_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        rcg_right = 0.5 * (1 + torch.exp(-ecg_right * w.cg_hand)) * (ref_right_contact_hand_any) + (
                    1 - ref_right_contact_hand_any)

        contact_reset = torch.cat([
            torch.abs(
                ref_left_contact_hand_any.unsqueeze(-1) - left_hand_contact_any) * ref_left_contact_hand_any.unsqueeze(
                -1),
            torch.abs(ref_right_contact_hand_any.unsqueeze(
                -1) - right_hand_contact_any) * ref_right_contact_hand_any.unsqueeze(-1),
        ], dim=-1)

        rcg_hand = rcg_left * rcg_right

        other_ids = [i for i in range(len(self.body_ids)) if
                     i not in left_contact_hand_ids and i not in right_contact_hand_ids]
        ref_other_contact = ref_human_contact[:, other_ids]
        other_contact = human_contact[:, other_ids]
        ecg_other = ((torch.abs(other_contact - ref_other_contact) * (ref_other_contact > contact_thres))).mean(dim=-1)
        rcg_other = torch.exp(-ecg_other * w.cg_other)

        no_contact = torch.abs(human_contact) < contact_thres
        ecg_all = (torch.abs(no_contact + ref_human_contact) * (ref_human_contact < -contact_thres)).mean(dim=-1)
        rcg_all = torch.exp(-ecg_all * w.cg_all)

        contact_all = self.contact_forces.clone().abs().sum(dim=-1).sum(dim=-1)
        contact_energy = contact_all.pow(2).mul(-w.eg3).exp()

        rcg = rcg_hand * rcg_other * rcg_all * contact_energy
        self.contact_reset = (self.contact_reset + contact_reset) * contact_reset

        self._accum_subterm("ContactGraph/LeftHandTerm", rcg_left)
        self._accum_subterm("ContactGraph/RightHandTerm", rcg_right)
        self._accum_subterm("ContactGraph/HandsCombinedTerm", rcg_hand)
        self._accum_subterm("ContactGraph/OtherBodiesTerm", rcg_other)
        self._accum_subterm("ContactGraph/NoContactConsistencyTerm", rcg_all)
        self._accum_subterm("ContactGraph/ContactEnergyTerm", contact_energy)

        return rcg

    def _accum_subterm(self, key: str, val: torch.Tensor):
        # val: (num_envs,) 或可 broadcast 到 (num_envs,)
        if key not in self.reward_subterm_sums:
            self.reward_subterm_sums[key] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reward_subterm_sums[key] += val.float()

    def begin_eval(self, motion_ids=None):
        ml = self._motion_lib
        device = self.device

        # 确定需要评估的 motion_ids
        if motion_ids is None:
            mids = ml.motion_ids  # 所有动作
        else:
            mids = torch.as_tensor(motion_ids, device=device, dtype=torch.long)

        # 1. 按物体类型对动作进行分桶 (Object ID -> Motion IDs)
        self._eval_buckets = {}
        unique_obj_ids = self.env_object_ids.unique().tolist()

        # 将传入的 mids 分类到对应的物体桶中
        # 注意：这里要查 MotionLibHoi 中动作对应的物体名
        for oid in unique_obj_ids:
            name = ml.object_vocab_inv[oid]
            # 找到所有物体名为 name 且在 mids 里的动作
            all_mids_for_obj = ml._motions_by_object[oid]
            # 取交集：只测试指定的 mids
            mask = torch.isin(all_mids_for_obj, mids)
            self._eval_buckets[oid] = all_mids_for_obj[mask]

        # 2. 初始化各物体的指针
        self._eval_cursors = {oid: 0 for oid in unique_obj_ids}
        self._eval_all_done = False

    def next_eval_batch_ids(self):
        num_envs = self.num_envs
        device = self.device
        ml = self._motion_lib

        batch_ids = torch.empty(num_envs, dtype=torch.long, device=device)

        # 记录本轮哪些环境是在执行有效测试（不是为了凑数填充的）
        # 用于后续统计评估是否真的结束
        finished_status = []

        for i in range(num_envs):
            oid = int(self.env_object_ids[i].item())
            bucket = self._eval_buckets[oid]
            cursor = self._eval_cursors[oid]

            if bucket.numel() == 0:
                # 保护：如果该环境的物体根本没有对应的动作
                batch_ids[i] = 0  # 填个占位符
                finished_status.append(True)
                continue

            # 取当前指针指向的动作
            batch_ids[i] = bucket[cursor % bucket.numel()]

            # 指针前进
            self._eval_cursors[oid] += 1

            # 如果该物体的桶刚跑完一次，标记一下
            if self._eval_cursors[oid] >= bucket.numel():
                finished_status.append(True)
            else:
                finished_status.append(False)

        # 判断是否所有桶都至少完整跑过一遍了
        all_buckets_done = True
        for oid in self._eval_buckets:
            if self._eval_cursors[oid] < self._eval_buckets[oid].numel():
                all_buckets_done = False
                break

        return batch_ids, all_buckets_done

    def reset_with_motion_ids(self, motion_ids, random = False):
        """ Reset all environments with given motion ids. (For Evaluation)
            This method is used to reset the environment with specific motion ids, e.g. in the training stage.
        Args:
            motion_ids (torch.Tensor): Tensor of shape [num_envs] containing motion ids to reset the environments with.
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._sampled_motion_ids = motion_ids.clone()
        self._motion_start_times = torch.zeros(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        motion_state = self._motion_lib.get_motion_state(self._sampled_motion_ids[env_ids], self._motion_start_times )

        self._set_env_state(env_ids=env_ids,
                            root_pos=motion_state["root_pos"] + self.pos_offset['root'][env_ids],
                            root_rot=motion_state["root_rot"],
                            dof_pos=motion_state["dof_pos"],
                            root_vel=motion_state["root_vel"],
                            root_ang_vel=motion_state["root_ang_vel"],
                            dof_vel=motion_state["dof_vel"],
                            key_pos=motion_state["key_pos"] + self.pos_offset['body'][env_ids],
                            key_rot=motion_state["key_rot"],
                            key_vel=motion_state["key_vel"],
                            key_ang_vel=motion_state["key_ang_vel"]
                            )

        self._reset_obj(env_ids)
        self._reset_env_tensors(env_ids)
        self.gym.clear_lines(self.viewer)

        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        # self.feet_air_time[env_ids] = 0.  # good idea!
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids])
            # self.extras["episode"]['rew_' + key] = torch.mean(
            #     self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self._refresh_sim_tensors()

        motion_lens = self._motion_lib._motion_lengths[motion_ids]
        self.extras["motion_length"] = motion_lens.clone()
        self.compute_observations()

        # obs, _, _, _, _ = self.step(
        #     torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return self.obs_buf

    def _init_phys_init_state_buffer(self):
        """
        Initializes the physics state buffer with the kinematic reference motion.
        For each motion in the library, we pre-calculate the full trajectory (Robot + Object)
        and store it as the initial 'valid' physics states.
        Structure: self.physics_state_buffer[motion_id] -> Tensor shape (Length, Num_Candidates, State_Dim)
        State Dim = Robot Root (13) + DoF Pos (Num_DoF) + DoF Vel (Num_DoF) + Obj State (13)
        """
        print("Initializing Physics State Buffer...")
        self.physics_state_buffer = {}
        self.physics_state_scores = {} # 新增
        num_motions = self._motion_lib.num_motions()

        # We process motions one by one (or could batch if memory allows, but lengths differ)
        for mid in range(num_motions):
            motion_len_frames = int(self._motion_lib._motion_lengths[mid].item() / self.dt) + 1

            # Create time steps for the entire motion length
            times = torch.arange(motion_len_frames, device=self.device) * self.dt
            # Batch motion id
            mids = torch.full((motion_len_frames,), mid, dtype=torch.long, device=self.device)

            # 1. Get Kinematic Robot State
            motion_state = self._motion_lib.get_motion_state(mids, times)

            # 2. Get Kinematic Object State (HOI)
            hoi_state = self._motion_lib.get_hoi_state(mids, times)

            # 3. Construct the State Vector
            # Robot Root State: [Pos(3), Rot(4), LinVel(3), AngVel(3)] -> 13
            # Note: MotionLib returns raw positions. We usually apply offsets in reset, but here we store RAW relative to ref,
            # and apply offsets when resetting into Env.
            # WAIT: If we store simulated states, they include the offset.
            # To be consistent, let's store states *including* the standard offset applied in reset.
            # However, the offset in reset is per-env (self.pos_offset['root'][env_ids]).
            # Since the buffer is global per motion ID, we should store them *without* the per-environment random offset,
            # or assume a canonical offset (e.g. zero) and add the specific env offset during reset.
            # Strategy: Store Canonical State (Raw MotionLib + Base Offset if any, but usually Base is 0 for library).
            # The reset function adds `self.pos_offset['root']`.

            root_pos = motion_state["root_pos"]  # (T, 3)
            root_rot = motion_state["root_rot"]  # (T, 4)
            root_vel = motion_state["root_vel"]  # (T, 3)
            root_ang_vel = motion_state["root_ang_vel"]  # (T, 3)

            dof_pos = motion_state["dof_pos"]  # (T, num_dof)
            dof_vel = motion_state["dof_vel"]  # (T, num_dof)

            obj_pos = hoi_state["obj_pos"]  # (T, 3)
            obj_rot = hoi_state["obj_rot"]  # (T, 4)
            obj_pos_vel = hoi_state["obj_pos_vel"]  # (T, 3)
            obj_rot_vel = hoi_state["obj_rot_vel"]  # (T, 3)

            # Concatenate: Root(13) + DoF_Pos(N) + DoF_Vel(N) + Obj(13)
            # Root State: Pos, Rot, LinVel, AngVel
            robot_root_state = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
            obj_state = torch.cat([obj_pos, obj_rot, obj_pos_vel, obj_rot_vel], dim=-1)

            full_state = torch.cat([robot_root_state, dof_pos, dof_vel, obj_state], dim=-1)  # (T, D)

            # Expand to (T, Num_Candidates, D)
            # Initialize all candidates with the same kinematic reference
            full_state_expanded = full_state.unsqueeze(1).repeat(1, self.physics_init_candidates, 1)

            self.physics_state_buffer[mid] = full_state_expanded

            scores = torch.zeros((motion_len_frames, self.physics_init_candidates), device=self.device,
                                 dtype=torch.float) + 0.1

            # 【关键修改】
            # 给 Candidate 0 一个"及格分" (例如 50.0，相当于存活了 1.5秒)
            # 这意味着如果仿真出的状态能存活 > 50帧，它就比原始参考更具吸引力
            scores[:, 0] = 50.0

            self.physics_state_scores[mid] = scores

        print(f"Physics Buffer initialized for {num_motions} motions.")

    def _reset_phys_state_init(self, env_ids):
        """
        Resets environment using states from the Physics State Buffer.
        Also updates the buffer with current valid states from environments that lived long enough.
        """
        # 1. Update Buffer with current states from surviving environments
        # Filter envs that have survived > 64 steps
        cond_survived = self.episode_length_buf[env_ids] > 64
        # 这里的随机概率可以调高一点，因为我们现在有优胜劣汰机制了，不用担心 Buffer 被垃圾数据填满
        cond_prob = torch.rand(len(env_ids), device=self.device) < 0.25
        survived_mask = cond_survived & cond_prob
        update_env_ids = env_ids[survived_mask]

        if len(update_env_ids) > 0:

            root_offset = self.pos_offset['root'][update_env_ids]

            # Capture Robot State
            curr_root_pos = self.base_pos[update_env_ids] - root_offset
            curr_root_rot = self.base_quat[update_env_ids]
            curr_root_lin_vel = self.root_states[update_env_ids, 7:10]
            curr_root_ang_vel = self.root_states[update_env_ids, 10:13]

            curr_dof_pos = self.dof_pos[update_env_ids]
            curr_dof_vel = self.dof_vel[update_env_ids]

            # Capture Object State
            curr_obj_pos = self.obj_pos[update_env_ids] - root_offset
            curr_obj_rot = self.obj_quat[update_env_ids]
            curr_obj_lin_vel = self.obj_vel[update_env_ids]
            curr_obj_ang_vel = self.obj_ang_vel[update_env_ids]

            curr_robot_root = torch.cat([curr_root_pos, curr_root_rot, curr_root_lin_vel, curr_root_ang_vel], dim=-1)
            curr_obj_state = torch.cat([curr_obj_pos, curr_obj_rot, curr_obj_lin_vel, curr_obj_ang_vel], dim=-1)
            captured_state = torch.cat(
                [curr_robot_root, curr_dof_pos, curr_dof_vel, curr_obj_state], dim=-1)

            curr_motion_ids = self._sampled_motion_ids[update_env_ids]
            curr_motion_times = self._motion_start_times[update_env_ids] + self.episode_length_buf[
                update_env_ids] * self.dt
            curr_frames = (curr_motion_times / self.dt).long()

            # --- Dynamic Trim Logic ---
            # Formula: 16 / length * 32 = 512 / length
            # Note: length > 64 checked above, so division is safe.
            curr_lengths = self.episode_length_buf[update_env_ids].float()

            curr_scores = self.episode_length_buf[update_env_ids].float() # (N_update,)

            curr_trim_frames = (512.0 / curr_lengths).long()

            unique_mids = torch.unique(curr_motion_ids)
            for mid_tensor in unique_mids:
                mid = int(mid_tensor.item())
                mid_mask = (curr_motion_ids == mid)

                batch_frames = curr_frames[mid_mask]
                batch_states = captured_state[mid_mask]
                batch_scores = curr_scores[mid_mask]  # 当前这批数据的分数
                batch_trim = curr_trim_frames[mid_mask]

                motion_buffer = self.physics_state_buffer[mid]
                score_buffer = self.physics_state_scores[mid]  # 获取分数 Buffer
                max_frame = motion_buffer.shape[0]

                valid_frame_mask = (batch_frames > batch_trim) & (batch_frames < (max_frame - batch_trim))
                if not valid_frame_mask.any(): continue

                final_frames = batch_frames[valid_frame_mask]
                final_states = batch_states[valid_frame_mask]
                final_new_scores = batch_scores[valid_frame_mask]

                # --- 核心修改：优胜劣汰逻辑 ---

                # 1. 获取 Buffer 中对应帧的所有候选者分数
                # current_buffer_scores: (Num_Updates, Num_Candidates)
                current_buffer_scores = score_buffer[final_frames]

                # 2. 找到每个帧中最差的候选者 (Min Value) 及其索引 (Argmin)
                # 注意：因为我们初始化时给了 Index 0 极高分 (10000)，所以 min() 永远不会选中 Index 0
                # 这样就自动实现了“保留原始参考”的功能
                min_scores, min_indices = torch.min(current_buffer_scores, dim=1)

                # 3. 只有当 新分数 > 旧的最低分 时，才执行替换
                replace_mask = final_new_scores > min_scores

                if replace_mask.any():
                    # 筛选出值得更新的
                    update_frames = final_frames[replace_mask]
                    update_indices = min_indices[replace_mask]  # 要覆盖的槽位

                    # 执行写入状态
                    self.physics_state_buffer[mid][update_frames, update_indices] = final_states[replace_mask]
                    # 执行更新分数
                    self.physics_state_scores[mid][update_frames, update_indices] = final_new_scores[replace_mask]

        # 2. Reset the requested environments using the buffer
        # A. Sample Motion IDs
        self.sample_motions(env_ids)

        num_resets = len(env_ids)
        reset_mids = self._sampled_motion_ids[env_ids]

        motion_lens_sec = self._motion_lib._motion_lengths[reset_mids]

        # 计算总帧数 (与 _init_physics_state_buffer 中的逻辑对齐: int(len/dt) + 1)
        motion_lens_frames = (motion_lens_sec / self.dt).long() + 1

        # Sample frames: Full range [0, Len)
        # 只要 buffer 初始化正确，首尾的 Kinematic 数据也是合法的采样点
        rand_factors = torch.rand(num_resets, device=self.device)
        sampled_frames = (rand_factors * motion_lens_frames).long()

        # C. Sample Candidate Index
        rand_candidates = torch.randint(0, self.physics_init_candidates, (num_resets,), device=self.device)

        # D. Gather States from Buffer
        # Since buffer is a Dict, we iterate or gather.
        # Given heterogenous shapes in dict, we likely loop or group by motion ID.
        # Grouping by motion ID is more efficient on GPU than single loops.

        # Placeholders for extracted data
        # Dim: 13 + 2*dof + 13
        state_dim = 13 + 2 * self.num_dofs + 13

        target_states = torch.zeros((num_resets, state_dim), device=self.device)

        unique_mids = torch.unique(reset_mids)
        for umid in unique_mids:
            mask = (reset_mids == umid)
            frames_for_mid = sampled_frames[mask]  # (N_subset,)

            # 获取这些帧的分数分布
            # scores: (N_subset, Num_Candidates)
            scores = self.physics_state_scores[int(umid.item())][frames_for_mid]

            # 转换为概率 (Softmax 或直接归一化)
            # 直接归一化即可，分数越高概率越大
            # 加一个 epsilon 防止除以 0
            probs = scores / (scores.sum(dim=1, keepdim=True) + 1e-6)

            # 采样
            # torch.multinomial 需要 2D 输入，对每一行采样 1 个索引
            cand_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (N_subset,)

            # 获取状态
            states = self.physics_state_buffer[int(umid.item())][frames_for_mid, cand_indices]
            target_states[mask] = states

        # E. Apply States to Environment Variables
        # Split target_states
        # Robot Root (13)
        idx_root = 13
        r_root = target_states[:, 0:idx_root]

        # Robot DoF Pos (Num_Dof)
        idx_dof_pos = idx_root + self.num_dofs
        r_dof_pos = target_states[:, idx_root:idx_dof_pos]

        # Robot DoF Vel (Num_Dof)
        idx_dof_vel = idx_dof_pos + self.num_dofs
        r_dof_vel = target_states[:, idx_dof_pos:idx_dof_vel]

        # Object State (13)
        r_obj = target_states[:, idx_dof_vel:]

        # Apply Offsets
        env_root_offset = self.pos_offset['root'][env_ids]

        # Set Robot State
        self.robot_states[env_ids, 0:3] = r_root[:, 0:3] + env_root_offset  # Pos
        self.robot_states[env_ids, 3:7] = r_root[:, 3:7]  # Rot
        self.robot_states[env_ids, 7:10] = r_root[:, 7:10]  # LinVel
        self.robot_states[env_ids, 10:13] = r_root[:, 10:13]  # AngVel

        self.dof_pos[env_ids] = r_dof_pos
        self.dof_vel[env_ids] = r_dof_vel

        # Set Object State (for _reset_obj usage later)
        self.obj_pos[env_ids] = r_obj[:, 0:3] + env_root_offset
        self.obj_quat[env_ids] = r_obj[:, 3:7]
        self.obj_vel[env_ids] = r_obj[:, 7:10]
        self.obj_ang_vel[env_ids] = r_obj[:, 10:13]

        # Set Tracking Times
        self._motion_start_times[env_ids] = sampled_frames * self.dt



@torch.jit.script
def compute_obj_observations_jit(root_pos, root_rot, obj_states, ref_obj_pos, ref_obj_rot, ref_obj_vel, ref_obj_ang_vel):
    tar_pos = obj_states[:, 0:3]
    tar_rot = obj_states[:, 3:7]
    tar_vel = obj_states[:, 7:10]
    tar_ang_vel = obj_states[:, 10:13]

    heading_rot = calc_heading_quat_inv(root_rot)
    heading_inv_rot = calc_heading_quat(root_rot)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = quat_to_tan_norm(local_tar_rot)

    diff_global_obj_pos = ref_obj_pos - tar_pos
    diff_local_obj_pos_flat = quat_rotate(heading_rot, diff_global_obj_pos)

    local_ref_obj_pos = ref_obj_pos - root_pos  # preserves the body position
    local_ref_obj_pos = quat_rotate(heading_rot, local_ref_obj_pos)


    diff_global_obj_rot = normalize(quat_mul(quat_conjugate(ref_obj_rot), tar_rot))
    diff_local_obj_rot_flat = quat_mul(quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)  # Need to be change of basis
    diff_local_obj_rot_obs = quat_to_tan_norm(diff_local_obj_rot_flat)

    local_ref_obj_rot = quat_mul(heading_rot, ref_obj_rot)
    local_ref_obj_rot = quat_to_tan_norm(local_ref_obj_rot)

    diff_global_vel = ref_obj_vel - tar_vel
    diff_local_vel = quat_rotate(heading_rot, diff_global_vel)

    diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
    diff_local_ang_vel = quat_rotate(heading_rot, diff_global_ang_vel)

    obs = torch.cat([local_tar_vel, local_tar_ang_vel, diff_local_obj_pos_flat, diff_local_obj_rot_obs, diff_local_vel, diff_local_ang_vel], dim=-1)
    return obs



@torch.jit.script
def normalize(x, eps: float = 1e-9):
    # 增加 eps 防止除以 0
    mask = x[..., -1] < 0
    x[mask] = -x[mask]
    return x / x.norm(p=2, dim=-1).clamp(min=eps).unsqueeze(-1)

@torch.jit.script
def compute_sdf(points1, points2):
    # type: (Tensor, Tensor) -> Tensor
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat

@torch.jit.script
def compute_hoi_observation_jit(root_rot, ref_ig, ig):
    N = root_rot.shape[0]
    root_rot_extend = root_rot.view(-1, 4)
    heading_rot = calc_heading_quat_inv(root_rot_extend)

    local_ref_ig = quat_rotate(heading_rot, ref_ig)
    local_ig = quat_rotate(heading_rot, ig)

    diff_ig = local_ref_ig - local_ig

    diff_ig = diff_ig.view(N, -1)
    local_ig = local_ig.view(N, -1)

    return torch.cat([local_ig, diff_ig], dim = -1)
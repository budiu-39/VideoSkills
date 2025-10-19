
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
    compute_mimic_observations_jit,
    compute_amp_observations_jit,
)


class LeggedRobotHoi(LeggedRobotImi):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        # self.motion_file = os.listdir(self.cfg.motion.file)
        self.motion_file = self.cfg.motion.file
        # TODO: Hacky code here for Behave dataset
        # self.object_name = [motion_example.split('/')[-1].split('_')[2].split('.')[0] for motion_example in self.motion_file]
        self.object_name = [motion_example.split('/')[-1].split('_')[1].split('.')[0] for motion_example in self.motion_file]
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



    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

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

    def _load_obj_asset(self):  # smplx
        asset_root = self.cfg.asset.asset_root
        self._obj_asset = []
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

            self._obj_asset.append(self.gym.load_asset(self.sim, asset_root, asset_file, asset_options))

            mesh_obj = trimesh.load(obj_file, force='mesh')
            # obj_verts = mesh_obj.vertices
            # center = np.mean(obj_verts, 0)
            object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2025)

            object_points = to_torch(object_points)

            while object_points.shape[0] < 1024:
                object_points = torch.cat([object_points, object_points[:1024 - object_points.shape[0]]], dim=0)
            self.object_points.append(to_torch(object_points))

        self.object_points = torch.stack(self.object_points, dim=0)
        return

    def _build_obj_tensors(self):
        num_actors = self.gym.get_actor_count(self.envs[0])
        self.obj_states = self.root_states.view(self.num_envs, num_actors, 13)[..., 1, :]

        self.obj_pos = self.obj_states[..., 0:3]
        self.obj_quat = self.obj_states[..., 3:7]
        self.obj_vel = self.obj_states[..., 7:10]
        self.obj_ang_vel = self.obj_states[..., 10:13]

        self.tar_actor_ids = self.robot_actor_ids + 1

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]
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
        motion_ids = self._sampled_motion_ids[env_ids]  # [B]
        motion_times = self._motion_start_times[env_ids] # + self.episode_length_buf[env_ids] * self.dt  # [B]

        hoi_state = self._motion_lib.get_hoi_state(motion_ids, motion_times)

        self.obj_pos[env_ids] =  hoi_state["obj_pos"] + self.pos_offset['root'][env_ids]
        self.obj_quat[env_ids] = hoi_state["obj_rot"]  # xyzw
        self.obj_vel[env_ids] = hoi_state["obj_pos_vel"]
        self.obj_ang_vel[env_ids] = hoi_state["obj_rot_vel"]
        self.ig[env_ids] = hoi_state['ig']
        self.body_contact[env_ids] = hoi_state['contact_robot']

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

        obj_obs = compute_obj_observations_jit(self.base_pos, self.base_quat, self.obj_states,
                                               self.ref_obj_pos, self.ref_obj_rot, self.ref_obj_pos_vel, self.ref_obj_rot_vel)

        hoi_obs = compute_hoi_observation_jit(self.base_quat.unsqueeze(1).repeat(1, self.num_bodies, 1), self.ref_ig.view(-1, 3)
                                              , ig)

        self.body_contact = torch.any(torch.abs(self.contact_forces[:, self.body_ids]) > 0.1, dim=-1).float()

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
                ig_env = self.ig[i].cpu().numpy()  # (52, 3)
                obj_near_env = body_pos_env + ig_env
                #
                num_lines = body_pos_env.shape[0]
                verts = np.empty((num_lines * 2, 3), dtype=np.float32)
                #
                verts[0::2] = body_pos_env
                verts[1::2] = obj_near_env

                # 颜色（蓝色）
                colors = np.tile(np.array([[0.2, 0.2, 1.0]], dtype=np.float32), (num_lines * 2, 1))
                self.gym.add_lines(self.viewer, env_ptr, num_lines, verts, colors)

                # self.ref_ig[i].cpu().numpy()  # (52, 3)
                # obj_near_ref = self.ref_body_pos[i].detach().cpu().numpy() + self.ref_ig[i].cpu().numpy()

                # verts_ref = np.empty((num_lines * 2, 3), dtype=np.float32)
                # verts_ref[0::2] = body_pos_env
                # verts_ref[1::2] = obj_near_ref
                # colors_ref = np.tile(np.array([[1.0, 0.2, 0.2]], dtype=np.float32), (num_lines * 2, 1))
                # self.gym.add_lines(self.viewer, env_ptr, num_lines, verts_ref, colors_ref)

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
        self.rew_buf[:] = 1.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf[:] *= rew
            self.episode_sums[name] += rew
            self.extras[f'reward_{name}'] = rew
        # index = torch.arange(self._curr_reward.shape[0])
        # # # print(self._humanoid_root_states.dtype)
        # self._curr_reward[index, self.progress_buf - self.start_times] = self.rew_buf
        # self._sum_reward[index] += self.rew_buf
        # self._curr_state[index, self.progress_buf - self.start_times, :] = torch.cat([
        #     self._humanoid_root_states,
        #     self._dof_pos,
        #     self._dof_vel,
        #     self._target_states,
        # ], dim=1)
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        return


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
        """
        一次性生成评估批次：
        - 每个 object 的 motion 数 < 加载该 object 的 env 数
        - 每个 object 的前 len(motions) 个 env 分配到各自 motion
        - 剩余 env 用同对象 motion 随机补齐
        """
        ml = self._motion_lib  # MotionLibHoi
        device = self.device

        if motion_ids is None:
            mids = ml.motion_ids
        else:
            mids = torch.as_tensor(motion_ids, device=device, dtype=torch.long)

        mids_mask = torch.zeros(int(ml._num_motions), dtype=torch.bool, device=device)
        mids_mask[mids] = True

        # 对象 -> motion 列表
        eval_buckets = {oid: bucket[mids_mask[bucket]]
                        for oid, bucket in ml._motions_by_object.items()
                        if mids_mask[bucket].any()}

        env_obj_ids = self.env_object_ids.to(device)
        single_batch = torch.empty(self.num_envs, dtype=torch.long, device=device)

        for oid, motions in eval_buckets.items():
            env_idx = torch.nonzero(env_obj_ids == oid, as_tuple=False).flatten()
            n_motions = motions.numel()
            # 前 n_motions 个 env 分配唯一 motion
            single_batch[env_idx[:n_motions]] = motions
            # 剩余 env 随机补齐
            extra_env = env_idx[n_motions:]
            ridx = torch.randint(0, n_motions, (extra_env.numel(),), device=device)
            single_batch[extra_env] = motions[ridx]

        self._eval_single_batch_ids = single_batch
        self._eval_done_once = False

    def next_eval_batch_ids(self):
        """返回一次性批次，并标记完成。"""
        if not getattr(self, "_eval_done_once", False):
            self._eval_done_once = True
            return self._eval_single_batch_ids, True
        else:
            return self._eval_single_batch_ids, True

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
    mask = x[..., -1] < 0  # 实部（w分量）为负
    x[mask] = -x[mask]
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

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
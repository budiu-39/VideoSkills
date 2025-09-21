
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import torch
from videoskills.envs.base.legged_robot_config import LeggedRobotCfg
from videoskills.envs.base.legged_robot_imi import LeggedRobotImi
from videoskills.utils.torch_utils import to_torch
import trimesh

class LeggedRobotHoi(LeggedRobotImi):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.motion_file = os.listdir(self.cfg.motion.file)
        self.object_name = [motion_example.split('_')[2] for motion_example in self.motion_file]
        self.object_density = self.cfg.object.object_density
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_target(env_id, env_ptr)
        return

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def _create_envs(self):

        self._target_handles = []
        self._load_target_asset()
        super()._create_envs()
        return

    def _load_target_asset(self):  # smplx
        asset_root = "dataset/behave/objects"
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
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]

        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]
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
        self.gym.set_actor_scale(env_ptr, target_handle, self.ball_size)

        return


    def _reset_robot(self, env_ids):
        super()._reset_robot(env_ids)
        # self._reset_target(env_ids)

    def _reset_target(self, env_ids):
        self._target_states[env_ids, :3] = self.extract_ref_component('obj_pos', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, 3:7] = self.extract_ref_component('obj_rot', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, 7:10] = self.extract_ref_component('obj_pos_vel', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, 10:13] = self.extract_ref_component('obj_rot_vel', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        return

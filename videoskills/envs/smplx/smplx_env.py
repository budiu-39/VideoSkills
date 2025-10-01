from isaacgym import gymapi
import torch
from videoskills.envs.base.legged_robot_imi import LeggedRobotImi
from videoskills.envs.base.legged_robot_hoi import LeggedRobotHoi
from videoskills.envs.base.legged_robot_imi import (
    compute_humanoid_observations_jit,
    compute_mimic_observations_jit,
    compute_amp_observations_jit,
)

SMPL24_BODIES = [
    'Pelvis',
    'L_Hip','L_Knee','L_Ankle','L_Toe',
    'R_Hip','R_Knee','R_Ankle','R_Toe',
    'Torso','Spine','Chest','Neck','Head',
    'L_Thorax','L_Shoulder','L_Elbow','L_Wrist','L_Hand',
    'R_Thorax','R_Shoulder','R_Elbow','R_Wrist','R_Hand',
]

class SMPLXRobot(LeggedRobotHoi):

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        robot_handle = self.robot_handles[env_id]
        filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        props = self.gym.get_actor_rigid_shape_properties(env_ptr, robot_handle)

        assert (len(filter_ints) == len(props))

        for p_idx in range(len(props)):
            props[p_idx].filter = filter_ints[p_idx]
        self.gym.set_actor_rigid_shape_properties(env_ptr, robot_handle, props)

        return

    def _create_envs(self):

        self._obj_handles = []
        self._load_obj_asset()
        super()._create_envs()

        SMPL_BODIES = [
            'Pelvis',
            'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
            'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
            'Torso', 'Spine', 'Chest', 'Neck', 'Head',
            'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
            'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand',
        ]

        SMPL_BODIES_NO_HAND = [
            'Pelvis',
            'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
            'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
            'Torso', 'Spine', 'Chest', 'Neck', 'Head',
            'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
            'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist',
        ]

        # sim 中名字 -> 索引
        name2sim = {n: i for i, n in enumerate(self.body_names)}

        # 选择出 sim 里的 24 个 body 索引（顺序按 SMPL24_BODIES）
        self.body_no_hand_ids = torch.as_tensor(
            [name2sim[n] for n in SMPL_BODIES_NO_HAND if n in name2sim],
            dtype=torch.long, device=self.device
        )
        self.dof_no_hand_ids = ((self.body_no_hand_ids[1:]-1).unsqueeze(1)*3 + torch.tensor([0,1,2], device=self.device)).reshape(-1)

    def compute_mimic_observations(self):
        task_obs = compute_mimic_observations_jit(self.base_pos, self.base_quat,
                                                  self.body_pos[:, self.body_no_hand_ids],
                                                  self.body_rot[:, self.body_no_hand_ids],
                                                  self.body_vel[:, self.body_no_hand_ids],
                                                  self.body_ang_vel[:, self.body_no_hand_ids],
                                                  self.ref_body_pos[:, self.body_no_hand_ids],
                                                  self.ref_body_rot[:, self.body_no_hand_ids],
                                                  self.ref_body_vel[:, self.body_no_hand_ids],
                                                  self.ref_body_ang_vel[:, self.body_no_hand_ids],
                                                  activate_quat_to_tan_norm=self.activate_quat_to_tan_norm)

        return task_obs

    # TODO: 测试一下关节对不对！  这里的代码是错误的因为会改变 self.dof_pos的内存
    # def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,
    #                    key_pos, key_rot, key_vel, key_ang_vel):
    #     self.robot_states[env_ids, 0:3] = root_pos
    #     self.robot_states[env_ids, 3:7] = root_rot
    #     self.robot_states[env_ids, 7:10] = root_vel
    #     self.robot_states[env_ids, 10:13] = root_ang_vel
    #
    #     row = self.dof_pos[env_ids].clone()
    #     row[:, self.dof_no_hand_ids_sim] = dof_pos[:, self.dof_no_hand_ids_sim]
    #     self.dof_pos[env_ids] = row
    #
    #     rowv = self.dof_vel[env_ids].clone()
    #     rowv[:, self.dof_no_hand_ids_sim] = dof_vel[:, self.dof_no_hand_ids_sim]
    #     self.dof_vel[env_ids] = rowv

        # self.dof_pos[env_ids][:, self.dof_no_hand_ids_sim] = dof_pos[:, self.dof_no_hand_ids_sim]
        # self.dof_vel[env_ids][:, self.dof_no_hand_ids_sim] = dof_vel[:, self.dof_no_hand_ids_sim]

        return







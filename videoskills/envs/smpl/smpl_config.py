from videoskills.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class SMPLRobotCfg( LeggedRobotCfg ):
    class init_state(LeggedRobotCfg.init_state):
        type = 'random'  # 'hybrid' or 'default'
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        # default_joint_angles = {  # = target angles [rad] when action = 0.0
        #     'FL_hip_joint': 0.1,  # [rad]
        #     'RL_hip_joint': 0.1,  # [rad]
        #     'FR_hip_joint': -0.1,  # [rad]
        #     'RR_hip_joint': -0.1,  # [rad]
        #
        #     'FL_thigh_joint': 0.8,  # [rad]
        #     'RL_thigh_joint': 1.,  # [rad]
        #     'FR_thigh_joint': 0.8,  # [rad]
        #     'RR_thigh_joint': 1.,  # [rad]
        #
        #     'FL_calf_joint': -1.5,  # [rad]
        #     'RL_calf_joint': -1.5,  # [rad]
        #     'FR_calf_joint': -1.5,  # [rad]
        #     'RR_calf_joint': -1.5,  # [rad]
        # }

    class motion:
        file = '{LEGGED_GYM_ROOT_DIR}/output/Humanoid_motion/smpl/amass/amass_selected_test.pkl'
        keybodys = ["R_Hand", "L_Hand", "R_Ankle", "L_Ankle"]

    class env(LeggedRobotCfg.env):

        num_observations = 24 * 15 + 1 - 3     # height + num_bodies * 15 (pos + vel + rot + ang_vel) - root_pos
        num_actions = 24
        # SMPL_MUJOCO_NAMES = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
        #                      'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow',
        #                      'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
        # filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33,
        #                 128, 0, 192, 0, 64, 0, 0, 0,
        #                 0, 0, 0, 0, 0, 0, 0]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/data/robots/smpl/smpl_0_humanoid.xml'
        name = "smpl_humanoid"
        foot_name = "Ankle"
        penalize_contacts_on = ["Hip", "Knee"]
        terminate_after_contacts_on = ["Pelvis"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0


class SMPLRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'smpl_ppo'

#
#
# class G1RoughCfg(LeggedRobotCfg):
#     class init_state(LeggedRobotCfg.init_state):
#         pos = [0.0, 0.0, 0.8]  # x,y,z [m]
#         default_joint_angles = {  # = target angles [rad] when action = 0.0
#             'left_hip_yaw_joint': 0.,
#             'left_hip_roll_joint': 0,
#             'left_hip_pitch_joint': -0.1,
#             'left_knee_joint': 0.3,
#             'left_ankle_pitch_joint': -0.2,
#             'left_ankle_roll_joint': 0,
#             'right_hip_yaw_joint': 0.,
#             'right_hip_roll_joint': 0,
#             'right_hip_pitch_joint': -0.1,
#             'right_knee_joint': 0.3,
#             'right_ankle_pitch_joint': -0.2,
#             'right_ankle_roll_joint': 0,
#             'torso_joint': 0.
#         }
#
#     class env(LeggedRobotCfg.env):
#         num_observations = 47
#         num_privileged_obs = 50
#         num_actions = 12
#
#     class domain_rand(LeggedRobotCfg.domain_rand):
#         randomize_friction = True
#         friction_range = [0.1, 1.25]
#         randomize_base_mass = True
#         added_mass_range = [-1., 3.]
#         push_robots = True
#         push_interval_s = 5
#         max_push_vel_xy = 1.5
#
#     class control(LeggedRobotCfg.control):
#         # PD Drive parameters:
#         control_type = 'P'
#         # PD Drive parameters:
#         stiffness = {'hip_yaw': 100,
#                      'hip_roll': 100,
#                      'hip_pitch': 100,
#                      'knee': 150,
#                      'ankle': 40,
#                      }  # [N*m/rad]
#         damping = {'hip_yaw': 2,
#                    'hip_roll': 2,
#                    'hip_pitch': 2,
#                    'knee': 4,
#                    'ankle': 2,
#                    }  # [N*m/rad]  # [N*m*s/rad]
#         # action scale: target angle = actionScale * action + defaultAngle
#         action_scale = 0.25
#         # decimation: Number of control action updates @ sim DT per policy DT
#         decimation = 4
#
#     class asset(LeggedRobotCfg.asset):
#         file = '{LEGGED_GYM_ROOT_DIR}/data/robots/g1_description/g1_12dof.urdf'
#         name = "g1"
#         foot_name = "ankle_roll"
#         penalize_contacts_on = ["hip", "knee"]
#         terminate_after_contacts_on = ["pelvis"]
#         self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
#         flip_visual_attachments = False
#
#     class rewards(LeggedRobotCfg.rewards):
#         soft_dof_pos_limit = 0.9
#         base_height_target = 0.78
#
#         class scales(LeggedRobotCfg.rewards.scales):
#             tracking_lin_vel = 1.0
#             tracking_ang_vel = 0.5
#             lin_vel_z = -2.0
#             ang_vel_xy = -0.05
#             orientation = -1.0
#             base_height = -10.0
#             dof_acc = -2.5e-7
#             dof_vel = -1e-3
#             feet_air_time = 0.0
#             collision = 0.0
#             action_rate = -0.01
#             dof_pos_limits = -5.0
#             alive = 0.15
#             hip_pos = -1.0
#             contact_no_vel = -0.2
#             feet_swing_height = -20.0
#             contact = 0.18
#
#
# class SMPLRoughCfgPPO(LeggedRobotCfgPPO):
#     class policy:
#         init_noise_std = 0.8
#         actor_hidden_dims = [32]
#         critic_hidden_dims = [32]
#         activation = 'selu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
#         # only for 'ActorCriticRecurrent':
#         rnn_type = 'lstm'
#         rnn_hidden_size = 64
#         rnn_num_layers = 1
#
#     class algorithm(LeggedRobotCfgPPO.algorithm):
#         entropy_coef = 0.01
#
#     class runner(LeggedRobotCfgPPO.runner):
#         policy_class_name = "ActorCriticRecurrent"
#         max_iterations = 10000
#         run_name = ''
#         experiment_name = 'g1'



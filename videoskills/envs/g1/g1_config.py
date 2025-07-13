from videoskills.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from dataclasses import dataclass
from typing import List
import numpy as np

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.33
        # actor_hidden_dims = [1024, 512, 256]
        # critic_hidden_dims =[1024, 512, 256]
        actor_hidden_dims = [2048, 1536, 1024, 1024, 512, 512]
        critic_hidden_dims = [2048, 1536, 1024, 1024, 512, 512]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate =  0.00002 #5.e-4   # 0.001    0.0005    0.00002   0.0001  0.00002
        entropy_coef = 0.01
        normalize_value = False
        normalize_obs = True
        schedule = 'fixed'

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        max_iterations = 30000
        run_name = '2e-7G1_universal'
        use_amp_runner = False
        load_run = ''
        experiment_name = 'g1_ppo'

        save_interval = 1000 # check for potential saves every this many iterations
        eval_interval = 2000

        num_steps_per_env = 32  # per iteration
        num_learning_epochs = 6
        num_mini_batches = 8


class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.89] # x,y,z [m]
        type = 'random'
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #    'left_hip_yaw_joint' : 0. ,
        #    'left_hip_roll_joint' : 0,
        #    'left_hip_pitch_joint' : -0.1,
        #    'left_knee_joint' : 0.3,
        #    'left_ankle_pitch_joint' : -0.2,
        #    'left_ankle_roll_joint' : 0,
        #    'right_hip_yaw_joint' : 0.,
        #    'right_hip_roll_joint' : 0,
        #    'right_hip_pitch_joint' : -0.1,
        #    'right_knee_joint' : 0.3,
        #    'right_ankle_pitch_joint': -0.2,
        #    'right_ankle_roll_joint' : 0,
        #    'torso_joint' : 0.
        # }   29 30 30

    class early_termination:
        enabled = True
        # distance = [0.25] * 24
        distance = [0.5] * 30
    
    class env(LeggedRobotCfg.env):
        episode_length_s = 5
        eval_mode = False
        land_event_detect = False
        num_envs = 4096
        num_actions = 29
        eval_mode = False
        # TODO: fix this
        num_observations = 1197
        norm_num_observations = 1197  # 去看一眼计算就好了
        activate_quat_to_tan_norm = True # if True, the quaternion is converted to tangent normalized quaternion

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

    class motion:
        rotate_motion = True
        # file = ('{LEGGED_GYM_ROOT_DIR}/dataset/G1_motion/AMASS_split_mid')
        file = ('{LEGGED_GYM_ROOT_DIR}/dataset/G1_motion/AMASS_train')

        bodies = ['pelvis','left_hip_pitch_link','left_hip_roll_link','left_hip_yaw_link','left_knee_link',
              'left_ankle_pitch_link','left_ankle_roll_link','right_hip_pitch_link','right_hip_roll_link',
              'right_hip_yaw_link','right_knee_link','right_ankle_pitch_link','right_ankle_roll_link',
              'waist_yaw_link','waist_roll_link','torso_link','left_shoulder_pitch_link','left_shoulder_roll_link',
              'left_shoulder_yaw_link','left_elbow_link','left_wrist_roll_link','left_wrist_pitch_link',
              'left_wrist_yaw_link','right_shoulder_pitch_link','right_shoulder_roll_link',
              'right_shoulder_yaw_link','right_elbow_link','right_wrist_roll_link','right_wrist_pitch_link',
              'right_wrist_yaw_link']

    class control:
        # PD Drive parameters:
        control_type = 'P'
        # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        pd_scale = 2
        stiffness = 12 * [100.] + 3 * [60.] + 14 * [40.]
        damping = 12 * [10.] + 3 * [6.] + 14 * [4.]

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1


    class asset( LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/data/robots/g1_description/g1_29dof.xml'
        file_urdf = '{LEGGED_GYM_ROOT_DIR}/data/robots/g1_description/g1_29dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 1
        # TODO：查看一下这是在干啥
        flip_visual_attachments = False
  
    class rewards:
        # soft_dof_pos_limit = 0.9
        only_positive_rewards = True
        class task_w:
            k_ang_vel = 0.1
            k_pos = 100
            k_rot = 10
            k_vel = 0.1
            w_ang_vel = 0.1
            w_pos = 0.5
            w_rot = 0.5
            w_vel = 0.1
        class scales:
            imitation = 1.0
            # torques = -0.00000002
            torques = -0.0000002

    class amp:
        activate = False
        num_amp_obs_steps = 10
        num_amp_obs = 232

    # config for retarget

    class retarget:
        fitting_iterations = 500
        output_dir = ('{LEGGED_GYM_ROOT_DIR}/dataset/G1_motion')
        amass_root = '/mnt/lustre/work/ponsmoll/pba936/AMASS'   # replace it with amass root in your workspace
        process_split = 'train'
        # amass_root = '/home/miku/Documents/AMASS_test'
        # process_split = 'test'
        humanoid_type = 'g1'
        num_jobs = 1
        vis = True
        joint_matches = [
            ["pelvis", "Pelvis"],
            ["left_hip_pitch_link", "L_Hip"],
            ["left_knee_link", "L_Knee"],
            ["left_ankle_roll_link", "L_Ankle"],
            ["right_hip_pitch_link", "R_Hip"],
            ["right_knee_link", "R_Knee"],
            ["right_ankle_roll_link", "R_Ankle"],
            ["left_shoulder_roll_link", "L_Shoulder"],
            ["left_elbow_link", "L_Elbow"],
            ["left_wrist_yaw_link", "L_Hand"],
            ["right_shoulder_roll_link", "R_Shoulder"],
            ["right_elbow_link", "R_Elbow"],
            ["right_wrist_yaw_link", "R_Hand"],
            ["head_link", "Head"],
            ["left_toe_link", "L_Toe"],
            ["right_toe_link", "R_Toe"],
            ]

        extend_config = [
            {
                "joint_name": "head_link",
                "parent_name": "pelvis",
                "pos": [0.0, 0.0, 0.4],
                "rot": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "joint_name": "left_toe_link",
                "parent_name": "left_ankle_roll_link",
                "pos": [0.1, 0.0, -0.032],
                "rot": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "joint_name": "right_toe_link",
                "parent_name": "right_ankle_roll_link",
                "pos": [0.1, 0.0, -0.032],
                "rot": [1.0, 0.0, 0.0, 0.0],
            },
        ]

        smpl_pose_modifier = {
            'Pelvis':[np.pi/2, 0, np.pi/2],
           'L_Shoulder':[0, 0, -np.pi/2],
           'R_Shoulder':[0, 0, np.pi/2],
           'L_Elbow':[0, -np.pi/2, 0],
           'R_Elbow':[0, np.pi/2, 0]
        }


  

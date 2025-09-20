from videoskills.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class SMPLRoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.055
        fixed_std = True
        # init_noise_std = 0.15
        # actor_hidden_dims = [1024, 512, 256]
        # critic_hidden_dims =[1024, 512, 256]
        actor_hidden_dims = [2048, 1536, 1024, 1024, 512, 512]
        critic_hidden_dims = [2048, 1536, 1024, 1024, 512, 512]
        activation = 'silu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 0.00002  # 5.e-4   # 0.001    0.0005    0.00002   0.0001  0.00002
        entropy_coef = 0.01
        normalize_value = False
        normalize_obs = True

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'phc_flexer_ET'
        experiment_name = 'smpl_ppo'
        use_amp_runner = False # 可以联动！和 amp
        max_iterations = 38000  # number of policy updates
        # load_run = 'SOTA_smpl_universal'
        # checkpoint = 10000
        # load_run = 'obs_norm'
        load_run = 'phc_universal'
        # load_run = 'SOTA_2e-8torque_norm_obs'

        # checkpoint = '6000'
        save_interval = 2000  # check for potential saves every this many iterations
        eval_interval = 2000

        num_steps_per_env = 32  # per iteration
        num_learning_epochs = 6
        num_mini_batches = 8

        # num_steps_per_env = 24 # per iteration
        # num_learning_epochs = 4
        # num_mini_batches = 5

    class refine:

        success_rate = 0.98
        convergence_threshold = 0.03
        convergence_criteria = 'reward'  # 'reward' or 'mpjpe'



    class amp_config:
        disc_batch = 512
        disc_updates = 1
        reward_coef = 1
        state_dim = 2320
        hidden_dims = [1024, 512]
        normalize_input = True
        lr = 3e-4
        grad_penalty_coef = 1.0
        logit_l2_coef = 1e-5
        weight_decay = 0.0001

        class dataset_cfg:
            replay_buffer_size = 200000
            demo_buffer_size = 200000

class SMPLRobotCfg( LeggedRobotCfg ):
    class init_state(LeggedRobotCfg.init_state):
        type = 'random'
        pos = [0.0, 0.0, 0.89]  # x,y,z [m]   1003 - 69 = 934

    class early_termination:
        enabled = True
        # distance = [0.25] * 24
        distance = [0.25] * 24

        reset_body = ['Pelvis', 'L_Hip', 'L_Knee', 'R_Hip', 'R_Knee',
                     'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow',  # 8
                     'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']    # 7

    class motion:
        rotate_motion = True
        # file = ('{LEGGED_GYM_ROOT_DIR}/dataset/smpl_motion/AMASS_train_fixed_height')
        # file = ('{LEGGED_GYM_ROOT_DIR}/dataset/smpl_motion/AMASS_test')
        file = ('{LEGGED_GYM_ROOT_DIR}/BEHAVE_processed')
        # file = ('{LEGGED_GYM_ROOT_DIR}/dataset/smpl_motion/GVHMR_tennis')
        # file = ('{LEGGED_GYM_ROOT_DIR}/dataset/smpl_motion/Crawling_push_ups_1_clip1')
        # file = ('{LEGGED_GYM_ROOT_DIR}/dataset/smpl_motion/Bent_opening_and_closing_leg_lifts_1_clip1')
        # file = ('{LEGGED_GYM_ROOT_DIR}/dataset/smpl_motion/In_situ_jump_rope_1_clip1')
        # file = ('{LEGGED_GYM_ROOT_DIR}/dataset/smpl_motion/test_data_136')
        # file = ('{LEGGED_GYM_ROOT_DIR}/dataset/smpl_motion/test_data_8')

        # bodies = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',    # 9
        #                      'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow',  # 8
        #                      'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']    # 7

        key_bodies = ["R_Ankle", "L_Ankle", "R_Wrist",  "L_Wrist"]

    class domain_rand:
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

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

    class env(LeggedRobotCfg.env):
        episode_length_s = 10  # 5 秒应该有 60 hz
        eval_mode = False
        land_event_detect = False
        num_envs = 4096
        num_actions = 69
        # TODO: now is the simplified edition
        # num_observations =  task_obs + humanoid_obs + 69 # 69 + 138 + 10 + 74 =
        num_observations = 859
        activate_quat_to_tan_norm = True
        norm_num_observations = 358 + 576 + 69

    class control:
        # PD Drive parameters:
        control_type = 'P'# [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 3.14
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2
        pd_scale = 1.0
        limit = (6 * [800] + 3 * [300] + 3 * [200] + 6 * [800] +  3 * [300] + 3 * [200]
                    + 9 * [200] + 15 * [200] + 6 * [50] + 9 * [200] + 6 * [50])
        velocity_limit = [50] * 69
        stiffness = [
            170, 150, 30, 150, 150, 30,     # 'L_Hip', 'L_Knee',
            10, 10, 10, 5, 5, 5,  # 'L_Ankle', 'L_Toe'
            170, 150, 30,  150, 150, 30,
            10, 10, 10,  5, 5, 5,
            60, 60, 60, 110, 40, 70,   # 'Torso', 'Spine',
            80, 30, 50, 5, 5, 5,  # 'Chest', 'Neck'
            5, 5, 5, 130, 20, 75,  # Head, 'L_Thorax',
            85, 20, 50, 30, 30, 30,   # 'L_Shoulder', 'L_Elbow'
            5, 5, 5, 5, 5, 5,  # 'L_Wrist', 'L_Hand'
            130, 20, 75, 85, 20, 50,
            30, 30, 30, 5, 5, 5,
            5, 5, 5,
        ]

        # 太大 的 damping 会
        damping = [
            15, 12, 3, 12, 12, 3,
            1.5, 1.5, 1.5, 1, 1, 1,
            15, 12, 3, 12, 12, 3,
            1.5, 1.5, 1.5, 1, 1, 1,
            6, 6, 6, 9, 4, 7,
            8, 3, 5, 1, 1, 1,
            1, 1, 1, 13, 2, 7,
            8, 2, 5, 3, 3, 3,
            1, 1, 1, 1, 1, 1,
            13, 2, 7, 8, 2, 5,
            3, 3, 3, 1, 1, 1,
            1, 1, 1,
        ]

        # ver 6
        # stiffness = [
        #     170, 150, 30, 30, 30, 10,     # 'L_Hip', 'L_Knee',
        #     10, 10, 10, 5, 5, 5,  # 'L_Ankle', 'L_Toe'
        #     170, 150, 30, 30, 30, 10,
        #     10, 10, 10,  5, 5, 5,
        #     60, 60, 60, 110, 40, 70,   # 'Torso', 'Spine',
        #     80, 30, 50, 5, 5, 5,  # 'Chest', 'Neck'
        #     5, 5, 5, 130, 20, 75,  # Head, 'L_Thorax',
        #     85, 20, 50, 15, 5, 12,   # 'L_Shoulder', 'L_Elbow'
        #     5, 5, 5, 5, 5, 5,  # 'L_Wrist', 'L_Hand'
        #     130, 20, 75, 85, 20, 50,
        #     15, 5, 12, 5, 5, 5,
        #     5, 5, 5,
        # ]
        # # 太大 的 damping 会
        # damping = [
        #     15, 12, 3, 3, 3, 1.5,
        #     1.5, 1.5, 1.5, 1, 1, 1,
        #     15, 12, 3, 3, 3, 1.5,
        #     1.5, 1.5, 1.5, 1, 1, 1,
        #     6, 6, 6, 9, 4, 7,
        #     8, 3, 5, 1, 1, 1,
        #     1, 1, 1, 13, 2, 7,
        #     8, 2, 5, 2, 1, 2,
        #     1, 1, 1, 1, 1, 1,
        #     13, 2, 7, 8, 2, 5,
        #     2, 1, 2, 1, 1, 1,
        #     1, 1, 1,
        # ]
        # pd_scale = 0.333
        # pd_scale = 0.2

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/data/robots/smpl/smpl_humanoid.xml'
        name = "smpl_humanoid"
        foot_name = "Ankle"
        penalize_contacts_on = ["Hip", "Knee"]
        terminate_after_contacts_on = ["Pelvis"]
        self_collisions = 1
        default_dof_drive_mode = 1


    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 10.

    class rewards:
        # soft_dof_pos_limit = 0.9
        only_positive_rewards = True
        class task_w:
            k_ang_vel = 0.1
            k_pos = 100
            k_rot = 10
            k_vel = 0.1
            w_ang_vel = 0.1
            w_pos = 0.3
            w_rot = 0.5
            w_vel = 0.1
            # k_ang_vel = 0.1
            # k_pos = 100
            # k_rot = 10
            # k_vel = 0.1
            # w_ang_vel = 0.5
            # w_pos = 0.5
            # w_rot = 0.5
            # w_vel = 0.5
        class scales:
            imitation = 1.0
            # torques = -0.000001
            dof_force = -0.0005
            # action_rate = - 0.02

    class sim(LeggedRobotCfg.sim):
        dt =  0.0166667        # 1/200 * 4 = 1/50    1/60 * 2 = 1/30
        # dt = 0.005

    class amp:
        activate = False
        num_amp_obs_steps = 10
        num_amp_obs = 232

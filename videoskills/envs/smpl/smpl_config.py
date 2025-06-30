from videoskills.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class SMPLRobotCfg( LeggedRobotCfg ):
    class init_state(LeggedRobotCfg.init_state):
        type = 'random'  # 'hybrid' or 'default'
        pos = [0.0, 0.0, 0.89]  # x,y,z [m]

    class marker:
        file = ('{LEGGED_GYM_ROOT_DIR}/data/marker/')

    class early_termination:
        enabled = True
        # distance = [0.25] * 24
        distance = [0.5] * 24

    class motion:
        # file = ('{LEGGED_GYM_ROOT_DIR}/dataset/AMASS_split_small')
        # file = ('{LEGGED_GYM_ROOT_DIR}/dataset/AMASS_test_fixed_height')
        # file = ('{LEGGED_GYM_ROOT_DIR}/AMASS_test_fixed_height')
        # file = ('{LEGGED_GYM_ROOT_DIR}/AMASS_split_mid')
        # file = ('{LEGGED_GYM_ROOT_DIR}/AMASS_split')
        # file = ('{LEGGED_GYM_ROOT_DIR}/dataset/AMASS_valid')
        file = ('{LEGGED_GYM_ROOT_DIR}/AMASS_fixed_height')
        # file = ('{LEGGED_GYM_ROOT_DIR}/AMASS_processed')
        # file = ('{LEGGED_GYM_ROOT_DIR}/AMASS_fixed_height')

        # file = ('{LEGGED_GYM_ROOT_DIR}/output/SMPL_Robot_motion/cxk')
        # file = ('{LEGGED_GYM_ROOT_DIR}/output/SMPL_Robot_motion/turn')

        # keybodys = ["R_Hand", "L_Hand", "R_Ankle", "L_Ankle"]
        bodies = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',    # 9
                             'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow',  # 8
                             'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']    # 7

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
        eval_mode = False
        land_event_detect = False
        num_envs = 4096
        num_actions = 69
        humanoid_obs = 1 + 23 * 3 + 24 * 12 #
        task_obs = 24 * 24  # (6 + 3 + 3 + 6 + 3 + 3)
        # TODO: now is the simplified edition
        # num_observations =  task_obs + humanoid_obs + 69 # 69 + 138 + 10 + 74 =
        num_observations = 859
        # base_pos 1 + base_lin_vel 3 + base_ang_vel 3 + projected_gravity 3 + dof_pos 23 * 3
        # + dof_vel 23 * 3 + actions 69

        # filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33,
        #                 128, 0, 192, 0, 64, 0, 0, 0,
        #                 0, 0, 0, 0, 0, 0, 0]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'# [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 3.14
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/data/robots/smpl/smpl_humanoid.xml'
        name = "smpl_humanoid"
        foot_name = "Ankle"
        penalize_contacts_on = ["Hip", "Knee"]
        terminate_after_contacts_on = ["Pelvis"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        pd_scale = 0.333
        # pd_scale = 0.1

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
            imitation = 10.0
            torques = -0.000001

    class sim(LeggedRobotCfg.sim):
        # dt =  0.01667        # 1/200 * 4 = 1/50    1/60 * 2 = 1/30
        dt = 0.005

    class amp:
        activate = True
        num_amp_obs_steps = 10
        num_amp_obs = 232



class SMPLRoughCfgPPO(LeggedRobotCfgPPO):

    class policy:
        init_noise_std = 0.33
        # actor_hidden_dims = [1024, 512, 256]
        # critic_hidden_dims =[1024, 512, 256]
        actor_hidden_dims = [2048, 1536, 1024, 1024, 512, 512]
        critic_hidden_dims = [2048, 1536, 1024, 1024, 512, 512]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1


    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate =  0.00002 #5.e-4   # 0.001    0.0005    0.00002   0.0001  0.00002
        entropy_coef = 0.002

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'SOTA_w_0002amp' # 'smpl_ppo'
        experiment_name = 'smpl_ppo'

        # load_run = "universal_old_toruqe_new_lr"
        # load_run = "old_stuff"
        # load_run = "universal_00001_torque_1000_imi_rotate"
        # load_run = "01pd_strictET_100imi"
        # load_run = "universal_old_torque_small_lr_noise_rotate_Jun22_10-17-04"
        # load_run = "universal_small_lr_noise_rotate_imi_Jun22_11-28-07"
        # load_run = "fixed_obs_00001_torque_100_imi_0001_lr_strict_RT_Jun25_03-29-39"
        # load_run = "universal_00001_torque_1000_imi_rotate"
        # load_run = "01pd"
        # load_run = 'universal_smpl_ref_out_Jun21_01-33-52' # -1 = last run
        # checkpoint = '6000'
        eval_interval = 2000

    amp_cfg = {
        "disc_batch": 512,
        "disc_updates": 1,
        "reward_coef": 0.5,

        "state_dim": 2320,             # 请替换为实际 amp_obs 维度
        "hidden_dims": [1024, 512],
        "normalize_input": True,
        "lr": 3e-4,
        "grad_penalty_coef": 1.0,
        "logit_l2_coef": 1e-5,
        "weight_decay": 0.0001,

        "dataset_cfg": {
            "replay_buffer_size": 200000,
            "demo_buffer_size": 200000
        }
    }


# disc_coef: 5
# disc_logit_reg: 0.01
# disc_grad_penalty: 5
# disc_reward_scale: 2
# disc_weight_decay: 0.0001
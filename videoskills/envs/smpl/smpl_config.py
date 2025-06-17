from videoskills.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class SMPLRobotCfg( LeggedRobotCfg ):
    class init_state(LeggedRobotCfg.init_state):
        type = 'random'  # 'hybrid' or 'default'
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]

    class marker:
        file = ('{LEGGED_GYM_ROOT_DIR}/data/marker/')

    class motion:
        file = ('{LEGGED_GYM_ROOT_DIR}/output/Humanoid_motion/smpl/turn')
        # keybodys = ["R_Hand", "L_Hand", "R_Ankle", "L_Ankle"]
        keybodys = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
                             'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow',
                             'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

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

        num_envs = 16
        num_actions = 69
        humanoid_obs = 1 + 23 * 3 + 24 * 10 #
        task_obs = 24 * 20
        # TODO: now is the simplified edition
        num_observations =  790 + 69 # 69 + 138 + 10 + 74 =
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

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class task_w:
            k_pos = 100
            k_rot = 10
            k_vel = 0.1
            k_ang_vel = 0.1
            w_pos = 0.5
            w_rot = 0.3
            w_vel = 0.1
            w_ang_vel = 0.1
        class scales(LeggedRobotCfg.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            imitation = 1000.0
            lin_vel_z = 0.0
            ang_vel_xy = 0
            orientation = -0.
            torques = -0.000001
            dof_vel = -0.
            dof_acc = -0.
            base_height = -0.
            feet_air_time =  0.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.00
            stand_still = -0.
            dof_pos_limits = -10.0



class SMPLRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'deepmimic_test'
        experiment_name = 'smpl_ppo'
        load_run = 'obs_max_early_termination' # -1 = last run




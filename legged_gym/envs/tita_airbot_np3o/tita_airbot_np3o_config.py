from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class TitaAirbotNP3OCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096 #8192
        num_actions = 8 + 6

        n_scan = 0 #187
        n_priv_latent =  4 + 1 + 1 + 14 + 14 +14
        n_proprio = 71

        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        history_encoding = True
        # num_privileged_obs = 71#51
        # num_propriceptive_obs = 71#51 # plus 2 dof vel(wheels) and 2 actions(wheels)
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights_critic = False
        measure_heights_actor = False
        measure_heights = False
        include_act_obs_pair_buf = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 3
        heading_command = False
        resampling_time = 10.
        global_reference = False
        class ranges:
            lin_vel_x = [-1., 1.]  # min max [m/s]
            heading = [-0.5, 0.5]
            lin_vel_y = [0, 0]
            ang_vel_yaw = [-3.14/2, 3.14/2]
    
    class init_state(LeggedRobotCfg.init_state):
        # pos = [0.0, 0.0, 0.8] # origin
        pos = [0.0, 0.0, 0.34]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "joint_left_leg_1": -0.0,
            "joint_right_leg_1": 0.0,
            "joint_left_leg_2": 0.8,
            "joint_right_leg_2": 0.8,
            "joint_left_leg_3": -1.5,
            "joint_right_leg_3": -1.5,
            "joint_left_leg_4": 0.0,
            "joint_right_leg_4": 0.0,

            "joint_arm_1": 0.0,
            "joint_arm_2": 0.0,
            "joint_arm_3": 0.0,
            "joint_arm_4": 0.0,
            "joint_arm_5": 0.0,
            "joint_arm_6": 0.0,
        }   
    
    class control(LeggedRobotCfg.control):
        control_type = "P_AND_V" # P: position, V: velocity, T: torques. 
                                 # P_AND_V: some joints use position control 
                                 # and others use vecocity control.
        # PD Drive parameters:
        stiffness = {
            "joint_left_leg_1": 30,
            "joint_left_leg_2": 30,
            "joint_left_leg_3": 30,
            "joint_right_leg_1": 30,
            "joint_right_leg_2": 30,
            "joint_right_leg_3": 30,
            "joint_left_leg_4": 0.0,
            "joint_right_leg_4": 0.0,
            
            "joint_arm_1": 15.0,
            "joint_arm_2": 15.0,
            "joint_arm_3": 15.0,
            "joint_arm_4": 15.0,
            "joint_arm_5": 15.0,
            "joint_arm_6": 15.0,
        }  # [N*m/rad]
        damping = {
            "joint_left_leg_1": 0.8,#0.8,
            "joint_left_leg_2": 0.8,#0.8,
            "joint_left_leg_3": 0.8,#0.8,
            "joint_right_leg_1": 0.8,#0.8,
            "joint_right_leg_2": 0.8,#0.8,
            "joint_right_leg_3": 0.8,#0.8,
            "joint_left_leg_4": 0.5,
            "joint_right_leg_4": 0.5,
            
            "joint_arm_1": 50.,
            "joint_arm_2": 50.,
            "joint_arm_3": 50.,
            "joint_arm_4": 50.,
            "joint_arm_5": 50.,
            "joint_arm_6": 0.1,
        }  # [N*m*s/rad]
        # action scale: target angle = actionscale * action + defaultangle
        # action_scale_pos is the action scale of joints that use position control
        # action_scale_vel is the action scale of joints that use velocity control
        action_scale_pos = 0.25
        action_scale_vel = 8
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4       

    class noise(LeggedRobotCfg.noise):
        add_noise = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tita_airbot/urdf/robot.urdf'
        foot_name = "_leg_4"
        foot_radius = 0.095
        penalize_contacts_on = ["base_link", "_leg_3"]
        terminate_after_contacts_on = ["base_link", "_leg_3"]
        arm_joint_name = ["joint_arm"]
        end_effector_name = "link_G2_base" # link6      
        replace_cylinder_with_capsule = False       
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
    
    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 2.75]
        randomize_restitution = True
        restitution_range = [0.0,1.0]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        randomize_base_com = True
        added_com_range = [-0.1, 0.1]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]

        randomize_kpkd = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]

        disturbance = False
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

        randomize_lag_timesteps = True
        lag_timesteps = 3

        added_mass_range = [-0.5, 2]
    
    class costs:
        class scales:
            dof_acc = -2.5e-07#-1.e-05
            dof_pos_limits = -2 #-2.0
            orientation = -10.#-1.0
            base_height = -10.#-50.0
            stand_still = -1.0
            torques = -2.5e-05
            torque_limits = -0.1 # cost
            action_rate = -0.01 #2.
            collision = -10.0
            feet_distance = -100 # -80
            same_foot_x_position = -10. #-1
            leg_oscillation = -1. # -20.
            dof_acc_arm = -2.5e-07
            tracking_ee_cart_l2 = -10#10.
            tracking_ee_orn_l2 = -10.
        class d_values:
            dof_acc = 0.
            dof_pos_limits = 0.
            orientation = 0.
            base_height = 0.
            stand_still = 0.
            torques = 0.
            torque_limits = 0.
            action_rate = 0.
            collision = 0.
            feet_distance = 0.
            same_foot_x_position = 0.
            leg_oscillation = 0.
            dof_acc_arm = 0.
            tracking_ee_cart_l2 = 0.
            tracking_ee_orn_l2 = 0.
            
    class cost:
        num_costs = 15
        
    class rewards:
        class scales:
            # tita scales
            tracking_lin_vel = 2.5#8. * 2#2.5
            tracking_ang_vel = 1.#2. * 2#1.

            # dof_acc = -1.e-05#-1.e-05
            # dof_pos_limits = -2 #-2.0

            # orientation = -10.#-1.0
            # base_height = -50.#-40.0
            # stand_still = -1.0
            survival = 0.9 #0.3

            # torques = -2.5e-05
            # torque_limits = -0.1 # cost
            # action_rate = -0.01 #2.
            # collision = -10.0

            no_fly = 0#1.0
            #feet_distance = -100 # -80
            wheel_adjustment = 1.0 # 1.0 off
            leg_symmetry = 2.0 #1.
            #same_foot_x_position = -10. #-1
            #leg_oscillation = -20. # -5.

            # airbot scales
            #dof_acc_arm = -2.5e-07
            tracking_ee_cart = 20.#20.
            #tracking_ee_cart_l2 = -10#10.
            tracking_ee_orn = 80.#80. 
            #tracking_ee_orn_l2 = -10.

        only_positive_rewards = False

        # tita parameter
        base_height_target = 0.39
        soft_torque_limit = 0.8
        soft_dof_pos_limit = 0.95  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        min_feet_distance = 0.57
        max_feet_distance = 0.60
        tracking_sigma = 1. #0.1 # tracking reward = exp(-error^2/sigma) lin_vel
        leg_symmetry_tracking_sigma = 0.001
        foot_x_position_sigma = 0.001

        dof_vel_history_length = 20
        oscillation_sign_thresh = 4 # number of sign changes in the history steps to consider oscillation
        oscillation_vel = 3
        
        # airbot parameter
        tracking_ee_cart_sigma = 0.1#1
        tracking_ee_orn_sigma = 1.
        active_orn_reward_threshold = 0.1 #0.15 # unit: m target ee and ee distance
        active_cartAndOrn_reward_threshold_horizontal = 0.3 #0.2
        active_cartAndOrn_reward_threshold_spin = 4.0 # the error between current yaw angular velocity and target yaw angular velocity

    class goal_ee:
        traj_time = [2, 3] 
        #hold_time = [2, 4]
        hold_time = [1, 2]
        collision_upper_limits = [0.3, 0.15, 0.05 - 0.165]
        collision_lower_limits = [-0.2, -0.15, -0.35 - 0.165]
        underground_limit = -0.57
        num_collision_check_samples = 10
        command_mode = 'sphere'
        stop_update_goal = False

        class sphere_center:
            x_offset = 0 #0.3 # Relative to base
            y_offset = 0 # Relative to base
            z_invariant_offset = 0.57 # Relative to terrain

        class ranges:
            init_pos_start = [0.35, np.pi/8, 0]
            init_pos_end = [0.4, 0, 0]
            nit_pos_l = [0.4, 0.5]
            init_pos_p = [-0.1 * np.pi / 6, 1 * np.pi / 3]
            init_pos_y = [-1 * np.pi / 4, 1 * np.pi / 4]

            pos_l = [0.4, 0.5]#pos_l = [0.35, 0.45]
            pos_p = [-1 * np.pi / 6, 1 * np.pi / 3]
            pos_y = [-0.4, 0.4]
            #pos_y = [-1 * np.pi / 4, 1 * np.pi / 4]

            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]
            final_tracking_ee_reward = 0.55

        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]#[2 / np.pi, 2 / np.pi, 2 / np.pi]

class TitaAirbotNP3OCfgPPO(LeggedRobotCfgPPO):
    class policy( LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        #priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = TitaAirbotNP3OCfg.cost.num_costs

        teacher_act = True
        imi_flag = True
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

    class runner( LeggedRobotCfgPPO.runner ):
        save_interval = 200
        run_name = ''
        experiment_name = 'tita_airbot_np3o'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        max_iterations = 4000
        num_steps_per_env = 24
        resume = False
        resume_path = ''


from legged_gym.envs import TitaAirbotRoughCfg, TitaAirbotRoughCfgPPO
import numpy as np

class TitaAirbotFlatCfg(TitaAirbotRoughCfg):
    class env(TitaAirbotRoughCfg.env):
        num_propriceptive_obs = 71 #- 8#51 # plus 2 dof vel(wheels) and 2 actions(wheels)
        num_privileged_obs = num_propriceptive_obs
        num_actions = 8 + 6
        num_envs = 4096 #8192

    class terrain(TitaAirbotRoughCfg.terrain):
        mesh_type = "plane"
        measure_heights_critic = False

    class commands(TitaAirbotRoughCfg.commands):
        num_commands = 3
        heading_command = False
        resampling_time = 5.

        class ranges(TitaAirbotRoughCfg.commands.ranges):
            lin_vel_x = [-1., 1.]  # min max [m/s]
            heading = [-0.5, 0.5]
            lin_vel_y = [0, 0]
            ang_vel_yaw = [-3.14/2, 3.14/2]
    
    class init_state(TitaAirbotRoughCfg.init_state):
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
    
    class control(TitaAirbotRoughCfg.control):
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

    class noise(TitaAirbotRoughCfg.noise):
        add_noise = False

    class asset(TitaAirbotRoughCfg.asset):
        foot_name = "_leg_4"
        foot_radius = 0.095
        penalize_contacts_on = ["base_link", "_leg_3"]
        terminate_after_contacts_on = ["base_link", "_leg_3"]
        arm_joint_name = ["joint_arm"]
        end_effector_name = "link_G2_base" # link6      
        replace_cylinder_with_capsule = False       
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    
    class domain_rand(TitaAirbotRoughCfg.domain_rand):
        friction_range = [0.2, 1.6]
        added_mass_range = [-0.5, 2]
        
    class rewards(TitaAirbotRoughCfg.rewards):
        class scales(TitaAirbotRoughCfg.rewards.scales):
            # tita scales
            tracking_lin_vel = 100. #* 2#2.5
            tracking_ang_vel = 100. #* 4 #* 2#1.

            dof_acc = -1.e-05#-1.e-05
            dof_pos_limits = -2.0

            orientation = -100.#-1.0
            base_height = -1000.#-50.0
            stand_still = -1.0
            survival = 0.9 #0.3

            torques = -2.5e-05
            torque_limits = -0.1
            action_rate = -0.01 #2.
            collision = -10.0

            no_fly = 1.0
            feet_distance = -100 # -80
            wheel_adjustment = 1.0 # 1.0 off
            leg_symmetry = 2.0 #1.
            same_foot_x_position = -10. #-1
            #leg_oscillation = -20. # -5.

            # airbot scales
            dof_acc_arm = -2.5e-07
            arm_dof_pos_limits = -1000.
            com_pos = 1. # center of mass

            tracking_ee_cart = 200.#10.
            tracking_ee_cart_l2 = -10#10.
            tracking_ee_orn = 200.#80.#80. 
            tracking_ee_orn_l2 = -10.
            
        # tita parameter
        base_height_target = 0.39
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
        tracking_ee_cart_sigma = 0.2#0.1#1
        tracking_ee_orn_sigma = 1.
        active_orn_reward_threshold = 0.1 # unit: m target ee and ee distance
        active_cartAndOrn_reward_threshold_horizontal = 0.3 #0.2
        active_cartAndOrn_reward_threshold_spin = 4.0 # the error between current yaw angular velocity and target yaw angular velocity
        com_pos_sigma = 20. #0.1

    class goal_ee:
        #traj_time = [0.6, 1.2] 
        traj_time = [2, 3] 
        #hold_time = [0.2, 0.4]
        hold_time = [1, 2]
        collision_upper_limits = [0.3, 0.15, 0.05 - 0.165]
        collision_lower_limits = [-0.2, -0.15, -0.35 - 0.165]
        underground_limit = -0.57
        num_collision_check_samples = 10
        command_mode = 'sphere'
        stop_update_goal = False
        drive_to_goal_mode = False

        class sphere_center:
            x_offset = 0 #0.3 # Relative to base
            y_offset = 0 # Relative to base
            z_invariant_offset = 0.57 # Relative to terrain

        class ranges:
            init_pos_start = [0.35, np.pi/8, 0]
            init_pos_end = [0.4, 0, 0]

            pos_l = [0.3, 0.5]#pos_l = [0.4, 0.5]
            #pos_p = [-1 * np.pi / 6, 1 * np.pi / 3]
            pos_p = [-1 * np.pi / 6, 3 * np.pi / 8]
            pos_y = [-0.4, 0.4]
            #pos_y = [-1 * np.pi / 4, 1 * np.pi / 4]

            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]

            # drive_to_goal mode
            init_pos_cart_world = [0, 0, 0.6]
            tita_start_pos = [-5., 0., 0.34]

        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]#[2 / np.pi, 2 / np.pi, 2 / np.pi]
        curriculum_length = 3000 # number of iterations

class TitaAirbotFlatCfgPPO(TitaAirbotRoughCfgPPO):
    class policy(TitaAirbotRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]

    class runner(TitaAirbotRoughCfgPPO.runner):
        experiment_name = 'tita_airbot_flat'
        max_iterations = 4000

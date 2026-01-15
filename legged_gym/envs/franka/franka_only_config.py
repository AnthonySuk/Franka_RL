from legged_gym.envs.franka.franka_rough_only_config import FrankaRoughOnlyCfgPPO, FrankaRoughOnlyCfg
import numpy as np

class FrankaOnlyCfg(FrankaRoughOnlyCfg):
    class env(FrankaRoughOnlyCfg.env):
        num_envs = 4096
        num_privileged_obs = 7 + 7 + 7 + 3 + 3 + 3 + 4 + 4
        num_propriceptive_obs = 7 + 7 + 7 + 3 + 3 + 3 + 4 + 4# plus 2 dof vel(wheels) and 2 actions(wheels)
        num_actions = 7

    class terrain(FrankaRoughOnlyCfg.terrain):
        mesh_type = "plane"
        measure_heights_critic = False

    class commands(FrankaRoughOnlyCfg.commands):
        num_commands = 3
        heading_command = False
        resampling_time = 5.

        class ranges(FrankaRoughOnlyCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            heading = [-1.0, 1.0]
            lin_vel_y = [0, 0]
            ang_vel_yaw = [-3.14, 3.14]

    class init_state(FrankaRoughOnlyCfg.init_state):
        # pos = [0.0, 0.0, 0.8] # origin
        pos = [0.0, 0.0, 0.3]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "panda_joint1": 0.0,
            "panda_joint2": -0.785,
            "panda_joint3": 0.0,
            "panda_joint4": -2.356,
            "panda_joint5": 0.0,
            "panda_joint6": 1.57,
            "panda_joint7": 0.785,
        }

    class control(FrankaRoughOnlyCfg.control):
        control_type = "P_AND_V" # P: position, V: velocity, T: torques. 
                                 # P_AND_V: some joints use position control 
                                 # and others use vecocity control.
        # PD Drive parameters:
        stiffness = {
            "panda_joint1": 15,#40.0,
            "panda_joint2": 15,#80.0,
            "panda_joint3": 15,#40.0,
            "panda_joint4": 15.0,
            "panda_joint5": 15.0,
            "panda_joint6": 20.0,
            "panda_joint7": 25.0,
        }  # [N*m/rad]
        damping = {
            "panda_joint1": 2.,#1.,
            "panda_joint2": 2.,
            "panda_joint3": 2.,
            "panda_joint4": 2.,
            "panda_joint5": 2.,
            "panda_joint6": 2.,
            "panda_joint7": 2.,
        }  # [N*m*s/rad]
        # action scale: target angle = actionscale * action + defaultangle
        # action_scale_pos is the action scale of joints that use position control
        # action_scale_vel is the action scale of joints that use velocity control
        action_scale_arm = [1., 1., 1., 0.6, 0.6, 0.6, 0.6]
        action_scale_vel = 8
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4       

        # arm_stiffness = 20
        # arm_damping = 0.2

    class asset(FrankaRoughOnlyCfg.asset):
        foot_radius = 0.095
        penalize_contacts_on = ["panda_link1","panda_link2","panda_link3","panda_link4","panda_link5","panda_link6","panda_link7","panda_hand"]
        terminate_after_contacts_on = ["panda_link1","panda_link2","panda_link3","panda_link4","panda_link5","panda_link6","panda_link7","panda_hand"]     
        arm_joint_name = ["panda_joint"]
        end_effector_name = "panda_hand_sc"
        replace_cylinder_with_capsule = False       
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

    class domain_rand(FrankaRoughOnlyCfg.domain_rand):
        friction_range = [0.2, 1.6]
        added_mass_range = [-0.5, 2]

    class rewards(FrankaRoughOnlyCfg.rewards):
        class scales(FrankaRoughOnlyCfg.rewards.scales):
            action_rate = -0.01#1
            arm_energy_abs_sum = -0.004#-0.004
            pushing_wrong_direction = 0. #-100
            tracking_ee_cart = 2.#1.
            tracking_ee_cart_l2 = -10.
            tracking_ee_orn = 20.#50. 
            tracking_ee_orn_l2 = -10.
            dof_acc = -2.5e-06#6
            dof_pos_limits = -1.
            torque_limits = -100.0
            survival = 0.9 #0.3
            
        soft_dof_pos_limit = 0.95  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        tracking_ee_cart_sigma = 0.1#1
        tracking_ee_orn_sigma = 1. # 1.
        active_orn_reward_threshold = 0.2 # target ee and ee distance
        soft_torque_limit = 0.95
    
    class goal_ee:
        traj_time = [2, 5] 
        hold_time = [2, 4]
        #hold_time = [1, 2]
        collision_upper_limits = [0.3, 0.15, 0.05 - 0.165]
        collision_lower_limits = [-0.2, -0.15, -0.35 - 0.165]
        underground_limit = -0.57
        num_collision_check_samples = 10
        command_mode = 'sphere'
        stop_update_goal = False

        class sphere_center:
            x_offset = 0 #0.3 # Relative to base
            y_offset = 0 # Relative to base
            z_invariant_offset = 0.6 # Relative to terrain

        class ranges:
            init_pos_start = [0.35, np.pi/8, 0]
            init_pos_end = [0.4, 0, 0]
            nit_pos_l = [0.4, 0.55]
            init_pos_p = [-0.1 * np.pi / 6, 1 * np.pi / 3]
            init_pos_y = [-1 * np.pi / 4, 1 * np.pi / 4]

            pos_l = [0.35, 0.45]
            #pos_p = [-1 * np.pi / 2.5, 1 * np.pi / 3]
            pos_p = [0, 1 * np.pi / 3]
            #pos_y = [0.1, 0.1]
            pos_y = [-1 * np.pi / 4, 1 * np.pi / 4]

            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]
            final_tracking_ee_reward = 0.55

        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]#[2 / np.pi, 2 / np.pi, 2 / np.pi]

class FrankaOnlyCfgPPO(FrankaRoughOnlyCfgPPO):
    class policy(FrankaRoughOnlyCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]

    class runner(FrankaRoughOnlyCfgPPO.runner):
        experiment_name = 'franka_only'
        max_iterations = 2000

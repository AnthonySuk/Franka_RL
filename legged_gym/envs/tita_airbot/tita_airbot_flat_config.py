from legged_gym.envs import TitaAirbotRoughCfg, TitaAirbotRoughCfgPPO


class TitaAirbotFlatCfg(TitaAirbotRoughCfg):
    class env(TitaAirbotRoughCfg.env):
        num_privileged_obs = 27 + 2 + 2 + 12
        num_propriceptive_obs = 27 + 2 + 2 + 12 # plus 2 dof vel(wheels) and 2 actions(wheels)
        num_actions = 8 + 6

    class terrain(TitaAirbotRoughCfg.terrain):
        mesh_type = "plane"
        measure_heights_critic = False

    class commands(TitaAirbotRoughCfg.commands):
        num_commands = 3
        heading_command = False
        resampling_time = 5.

        class ranges(TitaAirbotRoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            heading = [-1.0, 1.0]
            lin_vel_y = [0, 0]
            ang_vel_yaw = [-3.14, 3.14]
    
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

            "arm_joint1": 0.0,
            "arm_joint2": 0.0,
            "arm_joint3": 0.0,
            "arm_joint4": 0.0,
            "arm_joint5": 0.0,
            "arm_joint6": 0.0,
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
            
            "arm_joint1": 20.0,
            "arm_joint2": 20.0,
            "arm_joint3": 20.0,
            "arm_joint4": 20.0,
            "arm_joint5": 20.0,
            "arm_joint6": 20.0,
        }  # [N*m/rad]
        damping = {
            "joint_left_leg_1": 0.5,
            "joint_left_leg_2": 0.5,
            "joint_left_leg_3": 0.5,
            "joint_right_leg_1": 0.5,
            "joint_right_leg_2": 0.5,
            "joint_right_leg_3": 0.5,
            "joint_left_leg_4": 0.5,
            "joint_right_leg_4": 0.5,
            
            "arm_joint1": 0.2,
            "arm_joint2": 0.2,
            "arm_joint3": 0.2,
            "arm_joint4": 0.2,
            "arm_joint5": 0.2,
            "arm_joint6": 0.2,
        }  # [N*m*s/rad]
        # action scale: target angle = actionscale * action + defaultangle
        # action_scale_pos is the action scale of joints that use position control
        # action_scale_vel is the action scale of joints that use velocity control
        action_scale_pos = 0.25
        action_scale_vel = 8
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4       

    class asset(TitaAirbotRoughCfg.asset):
        foot_name = "_leg_4"
        foot_radius = 0.095
        penalize_contacts_on = ["base_link", "_leg_3"]
        terminate_after_contacts_on = ["base_link", "_leg_3"]       
        replace_cylinder_with_capsule = False       
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    
    class domain_rand(TitaAirbotRoughCfg.domain_rand):
        friction_range = [0.2, 1.6]
        added_mass_range = [-0.5, 2]
        
    class rewards(TitaAirbotRoughCfg.rewards):
        class scales(TitaAirbotRoughCfg.rewards.scales):
            # base class
            lin_vel_z = 0.0 # off
            ang_vel_xy = 0.0 # off
            orientation = -5.0 # 很重要，不加的话会导致存活时间下降
            base_height = -20.0
            torques = -2.5e-05
            dof_vel = 0.0 # off
            dof_acc = -2.5e-07
            action_rate = -0.01
            collision = -10.0
            termination = 0.0 # off
            dof_pos_limits = -2.0
            torque_limits = 0.0 # off
            tracking_lin_vel = 10.0
            tracking_ang_vel = 5.0 # off
            feet_air_time = 0.0 # off
            no_fly = 1.0
            unbalance_feet_air_time = 0.0 # off
            unbalance_feet_height = 0.0 # off
            feet_stumble = 0.0 # off
            stand_still = -1.0
            feet_contact_forces = 0.0 # off
            feet_distance = -100 # -100
            survival = 0.1
            # new added
            wheel_adjustment = 1.0 # 1.0 off
            inclination = 0.0 # off
            leg_symmetry = 10.0

        base_height_target = 0.4
        soft_dof_pos_limit = 0.95  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        min_feet_distance = 0.57
        max_feet_distance = 0.60
        tracking_sigma = 0.1 # tracking reward = exp(-error^2/sigma)
        nominal_foot_position_tracking_sigma = 0.005
        nominal_foot_position_tracking_sigma_wrt_v = 0.5
        # base_height_target = 0.65 + 0.1664
        leg_symmetry_tracking_sigma = 0.001
        foot_x_position_sigma = 0.001

class TitaAirbotFlatCfgPPO(TitaAirbotRoughCfgPPO):
    class policy(TitaAirbotRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]

    class runner(TitaAirbotRoughCfgPPO.runner):
        experiment_name = 'tita_airbot_flat'
        max_iterations = 2000

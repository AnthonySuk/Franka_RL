from legged_gym.envs.base.base_config import BaseConfig
import numpy as np

class TitaAirbotFollowCfg(BaseConfig):
    class env:
        num_envs = 4096 #8192 
        num_propriceptive_obs = 60
        num_privileged_obs = 181 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 8 + 6
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        num_gripper_joints = 1
        teleop_mode = False
        stop_update_goal = False
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            
            heading = [-3.14, 3.14]
    class terrain:
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 0.4
        dynamic_friction = 0.6
        restitution = 0.8
        # rough terrain only:
        measure_heights_actor = False
        measure_heights_critic = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4,
                             0.5]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0,1, 0, 0, 0]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
    class init_state:
        pos = [0.0, 0.0, 0.36]  # x,y,z [m] # z = 0.34
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

    class goal_ee:
        num_commands = 3
        traj_time = [1, 3] 
        hold_time = [0.5, 2]
        collision_upper_limits = [0.3, 0.15, 0.05 - 0.165]
        collision_lower_limits = [-0.2, -0.15, -0.35 - 0.165]
        underground_limit = -0.57
        local_axis_z_offset = 0.55 #0.3
        num_collision_check_samples = 10
        command_mode = 'sphere'
        arm_induced_pitch = 0 # 0.38 Added to -pos_p (negative goal pitch) to get default eef orn_p

        class sphere_center:
            x_offset = 0 #0.3 # Relative to base
            y_offset = 0 # Relative to base
            z_invariant_offset = 0.62 # Relative to terrain

        class ranges:
            init_pos_start = [0.5, np.pi/8, 0]
            init_pos_end = [0.7, 0, 0]
            nit_pos_l = [0.4, 0.55]
            init_pos_p = [-0.1 * np.pi / 6, 1 * np.pi / 3]
            init_pos_y = [-1 * np.pi / 4, 1 * np.pi / 4]

            pos_l = [0.4, 0.55]
            pos_p = [-1 * np.pi / 2.5, 1 * np.pi / 3]
            pos_y = [-1.2, 1.2]

            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]
            final_tracking_ee_reward = 0.55

        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]#[2 / np.pi, 2 / np.pi, 2 / np.pi]

    class arm:
        init_target_ee_base = [0.2, 0.0, 0.2]
        grasp_offset = 0.08
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)

    class rewards:
        class scales:
            # base class
            termination = 0.0 # off
            tracking_lin_vel = 1.5 # 2.5
            tracking_ang_vel = 0.5 # 1
            lin_vel_z = 0.0 # off
            ang_vel_xy = 0.0 # off
            orientation = -10.0 # 很重要，不加的话会导致存活时间下降
            torques = -2.5e-05
            dof_vel = 0.0 # off
            dof_acc = -2.5e-08 #e-08
            base_height = -20 # -30
            feet_air_time = 0.0 # off
            collision = -10.0 #-10
            feet_stumble = 0.0 # off
            action_rate = -0.01
            stand_still = -1. # -1
            dof_pos_limits = -100.0 # -10
            # don't exist on paper
            torque_limits = -1 # off
            no_fly = 1.0
            unbalance_feet_air_time = 0.0 # off
            unbalance_feet_height = 0.0 # off
            feet_contact_forces = 0.0 # off
            feet_distance = -100 # -100
            survival = 0.1 # 0.1
            # new added
            wheel_adjustment = 1.0 # 1.0 off
            inclination = 0.0 # off
            leg_symmetry = 10.0 # 10.0
            same_foot_x_position = -1.
            # arm related
            tracking_ee_sphere = 0.8
            tracking_ee_world = 0.8 # 0.8
            tracking_ee_cart = 0.8
            tracking_ee_orn = 0.8
            tracking_ee_orn_ry = 0.8

            arm_termination = 0.
            arm_orientation = 0.
            arm_energy_abs_sum = 0.
           

        base_height_target = 0.4

        soft_dof_pos_limit = 0.95  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        
        min_feet_distance = 0.57 #0.57
        max_feet_distance = 0.60
        min_feet_air_time = 0.25
        max_feet_air_time = 0.65
        
        nominal_foot_position_tracking_sigma = 0.005
        nominal_foot_position_tracking_sigma_wrt_v = 0.5

        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)
        tracking_ee_sigma = 1

        leg_symmetry_tracking_sigma = 0.001 # 0.001
        foot_x_position_sigma = 0.001 # 0.001

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        max_contact_force = 200.  # forces above this value are penalized
    class control:
        control_type = 'P_AND_V'  # P: position, V: velocity, T: torques
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
            "joint_arm_1": 40,
            "joint_arm_2": 40,
            "joint_arm_3": 40,
            "joint_arm_4": 20,
            "joint_arm_5": 20,
            "joint_arm_6": 20
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
            "joint_arm_1": 0.5,
            "joint_arm_2": 0.5,
            "joint_arm_3": 0.5,
            "joint_arm_4": 0.5,
            "joint_arm_5": 0.5,
            "joint_arm_6": 0.5
        }  # [N*m*s/rad]

        action_scale = 0.5 # use in play.py

        action_scale_pos = 0.25
        action_scale_vel = 8

        action_scale_arm = [2.1, 0.6, 0.6, 0.6, 0.6, 0.6]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    class domain_rand:
        randomize_friction = False
        friction_range = [0.0, 1.6]
        randomize_base_mass = False
        added_mass_range = [-1., 2.]
        push_robots = False
        push_interval_s = 7
        max_push_vel_xy = 1.

        randomize_base_com = False
        rand_com_vec = [0.03, 0.02, 0.03]
    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tita_airbot/urdf/robot.urdf'
        name = 'tita_airbot'
        foot_name = '_leg_4'
        terminate_after_contacts_on = ["link_G2_base", "link4", "link3"]
        penalize_contacts_on = ["base_link", "_leg_3", "link_G2_base"]
        arm_joint_name = ["joint_arm"]
        end_effector_name = "link_G2_base" # link6
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            gripper_track = 1.0

        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

    class physx:
        num_threads = 10
        solver_type = 1  # 0: pgs, 1: tgs
        num_position_iterations = 4
        num_velocity_iterations = 0
        contact_offset = 0.01  # [m]
        rest_offset = 0.0  # [m]
        bounce_threshold_velocity = 0.5  # 0.5 [m/s]
        max_depenetration_velocity = 1.0
        max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
        default_buffer_size_multiplier = 5
        contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class TitaAirbotFollowCfgPPO(BaseConfig):
    seed = 1 
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [128, 64, 32] # 128 64 32
        critic_hidden_dims = [128, 64, 32] # 128 64 32
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 2000  # number of policy updates

        # logging
        save_interval = 200  # check for potential saves every this many iterations
        experiment_name = 'tita_airbot_follow'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
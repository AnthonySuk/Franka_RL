import os
from typing import Dict
import sys

import torch
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, euler_from_quat
from legged_gym.utils.terrain import Terrain

from collections import deque
import numpy as np


class TitaAirbotRobot:
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg()
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_propriceptive_obs
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.proprioceptive_obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self._include_feet_height_rewards = self._check_if_include_feet_height_rewards()
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def get_observations(self):
        return self.proprioceptive_obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self):
        """ Reset all robots"""
        self.update_curr_ee_goal()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.actions[:,8:] = torch.clamp(self.actions[:,8:], self.dof_pos_limits[8:, 0], self.dof_pos_limits[8:, 1])
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.refresh_ee_goal_variable()

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.proprioceptive_obs_buf = torch.clip(self.proprioceptive_obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.proprioceptive_obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    # 该函数计算奖励函数，观测值，检查单个机器人训练是否终止等
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        # 更新机器人状态以及接触力
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        # add base position 计算机器人坐标系下的基本状态
        self.base_position = self.root_states[:, :3]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.dof_pos[:,[3, 7]]  = 0 

        thigh_vel = self.dof_vel[:,[1,5]]
        masked_vel = (torch.abs(thigh_vel) > self.cfg.rewards.oscillation_vel).float()
        self.dof_vel_history.append(thigh_vel.clone() * masked_vel)

        if self.cfg.terrain.measure_heights_actor or self.cfg.terrain.measure_heights_critic:
            self.measured_heights = self._get_heights()
        self._compute_feet_states()

        # 该函数主要对给训练机器人的指令进行设置
        self._post_physics_step_callback()

        self.update_curr_ee_goal()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        # add foot position and base position
        self.last_foot_positions[:] = self.foot_positions[:]
        self.last_base_position[:] = self.base_position[:]
        self.last_foot_velocities[:] = self.foot_velocities[:]

        if self.viewer:
            self.gym.clear_lines(self.viewer)
            self._draw_ee_goal_curr()
            self._draw_ee_goal_traj()

    def _check_if_include_feet_height_rewards(self):
        members = [attr for attr in dir(self.cfg.rewards.scales) if not attr.startswith("__")]
        for scale in members:
            if "feet_height" in scale:
                return True
        return False

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        # add orientation check
        # on_orientation = torch.abs(torch.norm(self.projected_gravity[:, :2], dim=-1) / self.projected_gravity[:, -1]) > 1
        # self.reset_buf |= on_orientation

        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample(env_ids)
        self._resample_ee_goal(env_ids=env_ids,is_init=True)

        self._reset_buffers(env_ids)
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reset_buffers(self, env_ids):
        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.last_feet_air_time[env_ids] = 0.
        self.current_max_feet_height[env_ids] = 0.
        self.last_max_feet_height[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # add foot positions and base position
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        for h in self.dof_vel_history:
            h[env_ids] = 0.0

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        # 初始化总奖励值为0
        self.rew_buf[:] = 0.
        # 循环在_prepare_reward_function初始化的奖励函数字典
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            # 运行对应函数并乘上权重因子
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # 之后进行加和
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        self.compute_proprioceptive_observations()
        self.compute_privileged_observations()

        self._add_noise_to_obs()

    def _add_noise_to_obs(self):
        # add noise if needed
        if self.add_noise:
            obs_noise_vec, privileged_extra_obs_noise_vec = self.noise_scale_vec
            obs_noise_buf = (2 * torch.rand_like(self.proprioceptive_obs_buf) - 1) * obs_noise_vec
            self.proprioceptive_obs_buf += obs_noise_buf
            if self.num_privileged_obs is not None:
                privileged_extra_obs_buf = (2 * torch.rand_like(
                    self.privileged_obs_buf[:, len(self.noise_scale_vec[0]):]) - 1) * privileged_extra_obs_noise_vec
                self.privileged_obs_buf += torch.cat((obs_noise_buf, privileged_extra_obs_buf), dim=1)

    def compute_privileged_observations(self):
        if self.num_privileged_obs is not None:
            self._compose_privileged_obs_buf_no_height_measure()
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights_critic:
                self.privileged_obs_buf = self._add_height_measure_to_buf(self.privileged_obs_buf)
            if self.privileged_obs_buf.shape[1] != self.num_privileged_obs:
                raise RuntimeError(
                    f"privileged_obs_buf size ({self.privileged_obs_buf.shape[1]}) does not match num_privileged_obs ({self.num_privileged_obs})")

    def _compose_privileged_obs_buf_no_height_measure(self):
        #Wheel pos maybe unbounded when training,
        #since wheels are in velocity control mode, we don't wheel pos obs.

        ## torch.cat将多个张量沿指定维度拼接在一起。
	    ## 观测量是：
	    ## base_lin_vel机器人本体坐标系下的线性速度（x，y,z）
	    ## commands机器人前三项命令，机器人坐标系x方向，y方向上的线速度，机器人z轴角速度
	    ## 各关节速度
	    ## 动作(各个关节的角度，角速度，力矩，与选择的控制模式有关)

        self.dof_err = self.dof_pos - self.default_dof_pos
        #self.dof_err[:,self.feet_indices] = 0

        self.privileged_obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, # base_ang_vel机器人本体坐标系下的角速度（w_x,w_y,w_z）
                                             self.base_lin_vel * self.obs_scales.lin_vel,
                                             self.projected_gravity, # projected_gravity机器人坐标系下的重力分量（g_x, g_y, g_z）
                                             self.dof_err * self.obs_scales.dof_pos, # 各关节位置
                                             self.dof_vel * self.obs_scales.dof_vel, # 各关节速度,轮足相比点足扩展了2维，从6到8
                                             self.actions, # 动作(各个关节的角度，角速度，力矩，与选择的控制模式有关),轮足相比点足扩展了2维，从6到8
                                             self.commands[:, :3] * self.commands_scale, # commands机器人前三项命令，机器人坐标系x方向，y方向上的线速度，机器人z轴角速度

                                             self.curr_ee_goal_local,
                                             self.ee_pos_local,
                                            #  self.ee_orn,
                                            #  self.ee_goal_orn_quat,
                                             self.ee_pos_local - self.curr_ee_goal_local
                                             ), dim=-1)

    def compute_proprioceptive_observations(self):
        self._compose_proprioceptive_obs_buf_no_height_measure()
        if self.cfg.terrain.measure_heights_actor:
            self.proprioceptive_obs_buf = self._add_height_measure_to_buf(self.proprioceptive_obs_buf)
        if self.proprioceptive_obs_buf.shape[1] != self.num_obs:
            raise RuntimeError(
                f"obs_buf size ({self.proprioceptive_obs_buf.shape[1]}) does not match num_obs ({self.num_obs})")

    def _add_height_measure_to_buf(self, buf):
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                             1.) * self.obs_scales.height_measurements
        buf = torch.cat(
            (buf, heights), dim=-1
        )
        return buf

    def _compose_proprioceptive_obs_buf_no_height_measure(self):
        #Wheel pos maybe unbounded when training,
        #since wheels are in velocity control mode, we don't wheel pos obs.
        self.dof_err = self.dof_pos - self.default_dof_pos
        #self.dof_err[:,self.feet_indices] = 0

        self.proprioceptive_obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                                 self.base_lin_vel * self.obs_scales.lin_vel,
                                                 self.projected_gravity,
                                                 self.dof_err * self.obs_scales.dof_pos,
                                                 self.dof_vel * self.obs_scales.dof_vel,
                                                 self.actions,
                                                 self.commands[:, :3] * self.commands_scale,

                                                 self.curr_ee_goal_local,
                                                 self.ee_pos_local,
                                                #  self.ee_orn,
                                                #  self.ee_goal_orn_quat,
                                                 self.ee_pos_local - self.curr_ee_goal_local
                                                 ), dim=-1)

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        obs_noise_vec = torch.zeros(self.cfg.env.num_propriceptive_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        obs_noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel

        obs_noise_vec[3:6] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel

        obs_noise_vec[3:6] = noise_scales.gravity * noise_level

        dof_pos_end_idx = 6 + self.num_dof
        obs_noise_vec[6:dof_pos_end_idx] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos

        dof_vel_end_idx = dof_pos_end_idx + self.num_dof
        obs_noise_vec[dof_pos_end_idx:dof_vel_end_idx] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel

        last_action_end_idx = dof_vel_end_idx + self.num_actions
        obs_noise_vec[dof_vel_end_idx:last_action_end_idx] = 0.  # previous actions

        command_end_idx = last_action_end_idx + self.cfg.commands.num_commands
        obs_noise_vec[last_action_end_idx:command_end_idx] = 0.  # commands
        
        if self.cfg.env.num_privileged_obs is not None:
            privileged_extra_obs_noise_vec = torch.zeros(
                self.cfg.env.num_privileged_obs - self.cfg.env.num_propriceptive_obs, device=self.device)
        else:
            privileged_extra_obs_noise_vec = None

        if self.cfg.terrain.measure_heights_actor:
            measure_heights_end_idx = last_action_end_idx + len(self.cfg.terrain.measured_points_x) * len(
                self.cfg.terrain.measured_points_y)
            obs_noise_vec[
            last_action_end_idx:measure_heights_end_idx] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        if self.cfg.terrain.measure_heights_critic:
            if self.cfg.env.num_privileged_obs is not None:
                privileged_extra_obs_noise_vec[
                :len(self.cfg.terrain.measured_points_x) * len(
                    self.cfg.terrain.measured_points_y)] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        return obs_noise_vec, privileged_extra_obs_noise_vec


    def create_sim(self):
        """ Creates simulation, terrain and environments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
            self.base_mass[env_id] = props[0].mass
        if self.cfg.domain_rand.randomize_base_com:
            com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
            props[0].com.x += np.random.uniform(-com_x, 0) # props[0].com.x += np.random.uniform(-com_x, com_x)
            props[0].com.y += np.random.uniform(-com_y, com_y)
            props[0].com.z += np.random.uniform(-com_z, com_z)
        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 当前的训练周期处以设置的重采样周期，选出能够整除的机器人序号
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        
        # 给训练机器人随机选择一些指令
        self._resample(env_ids)
        # 如果是给航向角命令，也是通过期望航向角和当前航向角算出期望z轴角速度
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
        # 随机给机器人施加推力
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample(self, env_ids):
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # command顺序在机器人坐标系x方向，y方向上的线速度，机器人z轴角速度，期望航向角四个命令
        # 在命令的上下限中随机生成一个值
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        # 在机器人坐标系x方向，y方向上的线速度命令过小则置为0，目前的比较值是0.2
        # set small commands to zero
        self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.4).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        action_scale_pos = self.cfg.control.action_scale_pos
        action_scale_vel = self.cfg.control.action_scale_vel
        control_type = self.cfg.control.control_type
        if control_type == "P_AND_V":
            torques = self.p_gains * (
                    action_scale_pos* actions + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
            # Choose joints to V control.
            # We only care about wheels velocity, so velocity control mode is used.
            V_list = [3,7]
            torques[:,V_list] = self.d_gains[V_list]*(action_scale_vel* actions[:,V_list] - self.dof_vel[:,V_list])
            # aribot torque
            torques[:,8:] = self.p_gains[8:] * (actions[:,8:] - self.dof_pos[:,8:]) - self.d_gains[8:] * self.dof_vel[:,8:]
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environment ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environment ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """Random pushes the robots."""
        max_push_force = (
                self.base_mass.mean().item()
                * self.cfg.domain_rand.max_push_vel_xy
                / self.sim_params.dt
        )
        self.rigid_body_external_forces[:] = 0
        rigid_body_external_forces = torch_rand_float(
            -max_push_force, max_push_force, (self.num_envs, 3), device=self.device
        )
        self.rigid_body_external_forces[:, 0, 0:3] = quat_rotate(
            self.base_quat, rigid_body_external_forces
        )
        self.rigid_body_external_forces[:, 0, 2] *= 0.5

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques),
            gymapi.ENV_SPACE,
        )

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
                                                          -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                                                          self.cfg.commands.max_curriculum)

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel_history = deque(maxlen=self.cfg.rewards.dof_vel_history_length)
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        base_yaw = get_euler_xyz(self.base_quat)[2]
        self.base_yaw_eular = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, -1
        )
        
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        # add foot positions and base positions
        self.base_position = self.root_states[:, :3]
        self.last_base_position = torch.zeros_like(self.base_position)
        self.foot_positions = self.rigid_body_states.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]
        self.last_foot_positions = torch.zeros_like(self.foot_positions)
        self.foot_heights = torch.zeros_like(self.foot_positions)
        self.foot_velocities = torch.zeros_like(self.foot_positions)
        self.last_foot_velocities = torch.zeros_like(self.foot_velocities)
        self.foot_velocities_f = torch.zeros_like(self.foot_positions)
        self.foot_relative_velocities = torch.zeros_like(self.foot_velocities)
        
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                              device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
                                        device=self.device, requires_grad=False)
        self.first_contact = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                       device=self.device, requires_grad=False)
        self.last_max_feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                                device=self.device, requires_grad=False)
        self.current_max_feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                                   device=self.device, requires_grad=False)
        self.rigid_body_external_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.rigid_body_external_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights_actor or self.cfg.terrain.measure_heights_critic:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.init_ee_goal_variale()

        self.last_leg_oscillation_reward = torch.zeros(self.num_envs, self.feet_indices.shape[0],device=self.device)
        self.com_tita = torch.zeros(self.num_envs, 3, device = self.device)
        self.com_arm = torch.zeros(self.num_envs, 3, device = self.device)

        rigid_body_mass = []
        for env_handle, actor_handle in zip(self.envs, self.actor_handles):
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            masses = torch.tensor([p.mass for p in body_props], dtype=torch.float32)  # [num_bodies]
            rigid_body_mass.append(masses)
        rigid_body_mass = torch.stack(rigid_body_mass, dim=0).unsqueeze(-1)

        self.rigid_body_mass_tita = rigid_body_mass[:,:9]
        self.rigid_body_mass_tita = self.rigid_body_mass_tita.to(self.device)
        self.rigid_body_mass_arm = rigid_body_mass[:,9:]
        self.rigid_body_mass_arm = self.rigid_body_mass_arm.to(self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
        
        # self.episode_sums字典将包含与self.reward_scales字典相同数量的键
        # 但每个键都将映射到一个大小为 (self.num_envs) 的全零浮点数张量
        # 张量将用于在训练过程中累积每个环境的奖励。
        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        print("asset_path: ",asset_path)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        # # 添加打印关节名称的代码
        # self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # print("加载的关节名称:")
        # for i, name in enumerate(self.dof_names):  # <--- 新增索引打印
        #     print(f"索引 {i}: {name}") 
        

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.base_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.gripper_idx = self.body_names_to_idx[self.cfg.asset.end_effector_name]

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.goal_ee_ranges = class_to_dict(self.cfg.goal_ee.ranges)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights_actor and not self.terrain.cfg.measure_heights_critic:
            return
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]).unsqueeze(1)

        heights = self._get_terrain_heights_from_points(points)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_heights_below_foot(self):
        """ Samples heights of the terrain at required points around each foot.

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.feet_state[:, :, :2]

        heights = self._get_terrain_heights_from_points(points)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_terrain_heights_from_points(self, points):
        points = points + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        return heights

    def _compute_feet_states(self):
        # add foot positions
        self.foot_positions = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        # add foot velocities
        self.foot_velocities = (self.foot_positions - self.last_foot_positions) / self.dt
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        self.last_feet_air_time = self.feet_air_time * self.first_contact + self.last_feet_air_time * ~self.first_contact
        self.feet_air_time *= ~self.contact_filt
        if self._include_feet_height_rewards:
            self.last_max_feet_height = self.current_max_feet_height * self.first_contact + self.last_max_feet_height * ~self.first_contact
            self.current_max_feet_height *= ~self.contact_filt
            self.feet_height = self.feet_state[:, :, 2] - self._get_heights_below_foot()
            self.current_max_feet_height = torch.max(self.current_max_feet_height,
                                                     self.feet_height)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt

    # ------------ ee goal related ------------
    def init_ee_goal_variale(self):
        # ee info
        self.ee_pos = self.rigid_body_states[:, self.gripper_idx, :3]
        self.ee_orn = self.rigid_body_states[:, self.gripper_idx, 3:7]
        self.ee_default_orn = self.ee_orn.clone()
        self.ee_vel = self.rigid_body_states[:, self.gripper_idx, 7:]

        self.z_invariant_offset = torch.tensor([self.cfg.goal_ee.sphere_center.z_invariant_offset], device=self.device).repeat(self.num_envs, 1)
        self.ee_pos_local = quat_rotate_inverse(self.base_yaw_quat, self.ee_pos - torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1))
      
        # time setting
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.traj_timesteps = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), 
                                               device=self.device).squeeze(1) / self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], 
                                                                           (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        
        # target_ee info
        self.init_start_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_start, device=self.device).unsqueeze(0)
        self.init_end_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_end, device=self.device).unsqueeze(0)

        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler[:, 0] = 0#np.pi / 2
        self.ee_goal_orn_euler[:, 1] = np.pi / 2
        self.ee_goal_orn_euler[:, 2] = np.pi / 2
        self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_euler[:, 0], self.ee_goal_orn_euler[:, 1], self.ee_goal_orn_euler[:, 2])
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_center_offset = torch.tensor([self.cfg.goal_ee.sphere_center.x_offset, 
                                                   self.cfg.goal_ee.sphere_center.y_offset, 
                                                   self.cfg.goal_ee.sphere_center.z_invariant_offset], 
                                                   device=self.device).repeat(self.num_envs, 1)
        
        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_cart_world = self._get_ee_goal_spherical_center() + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)
        self.curr_ee_goal_local = quat_rotate_inverse(self.base_yaw_quat, self.curr_ee_goal_cart_world - torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1))

        if self.cfg.goal_ee.command_mode == 'cart':
            self.curr_ee_goal = self.curr_ee_goal_cart
        else:
            self.curr_ee_goal = self.curr_ee_goal_sphere

        self.arm_u = torch.zeros((self.num_envs,self.num_dofs),dtype=torch.float,device=self.device)
        
        # collision setting
        self.collision_lower_limits = torch.tensor(self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float)
        self.collision_upper_limits = torch.tensor(self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float)
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        assert(self.cfg.goal_ee.command_mode in ['cart', 'sphere'])

        # error scale
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)

        # curriculum setting
        self.max_target_l = self.goal_ee_ranges["pos_l"][1]
        self.max_target_p = self.goal_ee_ranges["pos_p"][1] if abs(self.goal_ee_ranges["pos_p"][1]) > abs(self.goal_ee_ranges["pos_p"][0]) else self.goal_ee_ranges["pos_p"][0]
        self.max_target_y = self.goal_ee_ranges["pos_y"][1]

        height_tensor = torch.tensor([self.cfg.rewards.base_height_target], device=self.device)
        ee_pos_local = torch.cat([self.ee_pos[0,:2] - self.base_position[0,:2],
                                  height_tensor], 
                                  dim=0)
        init_ee_l_sphere = torch.norm(ee_pos_local - self.ee_goal_center_offset[0])
        self.mean_l = init_ee_l_sphere
        self.mean_p = 0. #(self.goal_ee_ranges["pos_p"][1] + self.goal_ee_ranges["pos_p"][0]) / 2.
        self.mean_y = 0.

        self.curriculum_progress = self.common_step_counter / (23. * self.cfg.goal_ee.curriculum_length)

        self._resample_ee_goal(torch.arange(self.num_envs, device=self.device), is_init=True)
        
    def _reset_ee_goal_start(self):
        if self.curriculum_progress < 1.:
            self.curriculum_progress = self.common_step_counter / (23. * self.cfg.goal_ee.curriculum_length)

        variance_l = (1. +  self.max_target_l**2) * self.curriculum_progress
        variance_p = (1. +  self.max_target_p**2) * self.curriculum_progress 
        variance_y = (1. +  self.max_target_y**2) * self.curriculum_progress

        l = torch.normal(mean=self.mean_l, std=variance_l, size=())
        p = torch.normal(mean=self.mean_p, std=variance_p, size=())
        y = torch.normal(mean=self.mean_y, std=variance_y, size=())

        if l > self.goal_ee_ranges["pos_l"][1]:
            l = self.goal_ee_ranges["pos_l"][1]
        if l < self.mean_l:
            l = self.mean_l
        
        if p > self.goal_ee_ranges["pos_p"][1]:
            p = self.goal_ee_ranges["pos_p"][1]
        if p < self.goal_ee_ranges["pos_p"][0]:
            p = self.goal_ee_ranges["pos_p"][0]

        if y > self.goal_ee_ranges["pos_y"][1]:
            y = self.goal_ee_ranges["pos_y"][1]
        if y < self.goal_ee_ranges["pos_y"][0]:
            y = self.goal_ee_ranges["pos_y"][0]
        
        return torch.tensor([l,p,y], device=self.device)

    def _get_ee_goal_spherical_center(self):
        center = torch.cat([self.root_states[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        center = center + quat_apply(self.base_yaw_quat, self.ee_goal_center_offset)
        return center
    
    def cart2sphere(self, cart):
        sphere = torch.zeros_like(cart)
        radius = torch.norm(cart, dim=-1)
        radius = torch.clamp(radius, min=1e-6)  # prevent division by 0
        sphere[:, 0] = radius
        sphere[:, 1] = torch.atan2(cart[:, 2], cart[:, 0])  # yaw
        y_ratio = torch.clamp(cart[:, 1] / radius, -1.0, 1.0)  # clamp to asin domain
        sphere[:, 2] = torch.asin(y_ratio)  # pitch
        return sphere

    def sphere2cart(self,sphere):
        if sphere.ndim == 1:
            sphere = sphere.view(1, -1)  # or use unsqueeze(0)
        cart = torch.zeros_like(sphere)
        cart[:, 0] = sphere[:, 0] * torch.cos(sphere[:, 2]) * torch.cos(sphere[:, 1])
        cart[:, 1] = sphere[:, 0] * torch.sin(sphere[:, 2])
        cart[:, 2] = sphere[:, 0] * torch.cos(sphere[:, 2]) * torch.sin(sphere[:, 1])
        return cart
    
    def refresh_ee_goal_variable(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.base_quat = self.root_states[:, 3:7]
        self.ee_pos = self.rigid_body_states[:, self.gripper_idx, :3]
        self.ee_orn = self.rigid_body_states[:, self.gripper_idx, 3:7]

        base_yaw = get_euler_xyz(self.base_quat)[2]
        self.base_yaw_fixed = wrap_to_pi(base_yaw).view(self.num_envs,1)
        self.base_yaw_quat[:] = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        self.base_yaw_eular = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)

        self.ee_pos_local = quat_rotate_inverse(self.base_yaw_quat, self.ee_pos - torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1))
      
    def update_curr_ee_goal(self):
        self.refresh_ee_goal_variable()
        t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
        self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])

        self.curr_ee_goal_cart[:] = self.sphere2cart(self.curr_ee_goal_sphere)
        ee_goal_cart_yaw_global = quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)
        self.curr_ee_goal_cart_world = self._get_ee_goal_spherical_center() + ee_goal_cart_yaw_global
        self.curr_ee_goal_local = quat_rotate_inverse(self.base_yaw_quat, self.curr_ee_goal_cart_world - torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1))
        
        default_yaw = torch.atan2(ee_goal_cart_yaw_global[:, 1], ee_goal_cart_yaw_global[:, 0])
        #self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_delta_rpy[:, 0] + np.pi / 2, self.ee_goal_orn_delta_rpy[:, 1], self.ee_goal_orn_delta_rpy[:, 2] + default_yaw)
        self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_delta_rpy[:, 0] , self.ee_goal_orn_delta_rpy[:, 1], self.ee_goal_orn_delta_rpy[:, 2] + default_yaw)

        self.goal_timer += 1

        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()

        if len(resample_id) > 0 and self.cfg.goal_ee.stop_update_goal:
            # set these env commands as 0
            self.commands[resample_id, 0] = 0
            self.commands[resample_id, 2] = 0

        self._resample_ee_goal(resample_id)
    
    def _resample_ee_goal(self, env_ids, is_init=False):
        if len(env_ids) > 0:
            init_env_ids = env_ids.clone()
            
            if is_init:
                self.ee_goal_orn_delta_rpy[env_ids, :] = 0
                self.ee_start_sphere[env_ids] = self._reset_ee_goal_start()
            else:
                #self._resample_ee_goal_orn_once(env_ids)
                self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()

            for i in range(10):
                    self._resample_ee_goal_sphere_once(env_ids)
                    collision_mask = self.collision_check(env_ids)
                    env_ids = env_ids[collision_mask]
                    if len(env_ids) == 0:
                        break

            self.ee_goal_cart[init_env_ids, :] = self.sphere2cart(self.ee_goal_sphere[init_env_ids, :])
            self.goal_timer[init_env_ids] = 0.0

    def _resample_ee_goal_sphere_once(self, env_ids):
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(self.goal_ee_ranges["pos_l"][0], self.goal_ee_ranges["pos_l"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(self.goal_ee_ranges["pos_p"][0], self.goal_ee_ranges["pos_p"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(self.goal_ee_ranges["pos_y"][0], self.goal_ee_ranges["pos_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    def _resample_ee_goal_orn_once(self, env_ids):
        ee_goal_delta_orn_r = torch_rand_float(self.goal_ee_ranges["delta_orn_r"][0], self.goal_ee_ranges["delta_orn_r"][1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_p = torch_rand_float(self.goal_ee_ranges["delta_orn_p"][0], self.goal_ee_ranges["delta_orn_p"][1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_y = torch_rand_float(self.goal_ee_ranges["delta_orn_y"][0], self.goal_ee_ranges["delta_orn_y"][1], (len(env_ids), 1), device=self.device)
        self.ee_goal_orn_delta_rpy[env_ids, :] = torch.cat([ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1)

    def collision_check(self, env_ids):
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ...,  None], self.collision_check_t).squeeze(-1)
        ee_target_cart = self.sphere2cart(torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(torch.logical_and(torch.all(ee_target_cart < self.collision_upper_limits, dim=-1), torch.all(ee_target_cart > self.collision_lower_limits, dim=-1)), dim=0)
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask
    
    def _draw_ee_goal_traj(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 8, 8, None, color=(1, 0, 0))
        sphere_geom_yellow = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 1, 0))

        t = torch.linspace(0, 1, 10, device=self.device)[None, None, None, :]
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[..., None], self.ee_goal_sphere[..., None], t).squeeze(0)
        ee_target_all_cart_world = torch.zeros_like(ee_target_all_sphere)
        for i in range(10):
            ee_target_cart = self.sphere2cart(ee_target_all_sphere[..., i])
            ee_target_all_cart_world[..., i] = quat_apply(self.base_yaw_quat, ee_target_cart)
        ee_target_all_cart_world += self._get_ee_goal_spherical_center()[:, :, None]
        for i in range(self.num_envs):
            for j in range(10):
                pose = gymapi.Transform(gymapi.Vec3(ee_target_all_cart_world[i, 0, j], ee_target_all_cart_world[i, 1, j], ee_target_all_cart_world[i, 2, j]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def _draw_ee_goal_curr(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))

        sphere_geom_3 = gymutil.WireframeSphereGeometry(0.05, 16, 16, None, color=(0, 1, 1))
        upper_arm_pose = self._get_ee_goal_spherical_center()

        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))
        ee_pose = self.rigid_body_states[:, self.gripper_idx, :3]
        
       
        sphere_geom_origin = gymutil.WireframeSphereGeometry(0.1, 8, 8, None, color=(0, 1, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0), r=None)
        gymutil.draw_lines(sphere_geom_origin, self.gym, self.viewer, self.envs[0], sphere_pose)  # Green ball

        axes_geom = gymutil.AxesGeometry(scale=0.2)

        for i in range(self.num_envs):
            sphere_pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
            
            sphere_pose_2 = gymapi.Transform(gymapi.Vec3(ee_pose[i, 0], ee_pose[i, 1], ee_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], sphere_pose_2) 

            sphere_pose_3 = gymapi.Transform(gymapi.Vec3(upper_arm_pose[i, 0], upper_arm_pose[i, 1], upper_arm_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_3, self.gym, self.viewer, self.envs[i], sphere_pose_3) 

            pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]), 
                                    r=gymapi.Quat(self.ee_goal_orn_quat[i, 0], self.ee_goal_orn_quat[i, 1], self.ee_goal_orn_quat[i, 2], self.ee_goal_orn_quat[i, 3]))
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)

    # ------------ tita reward functions----------------
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.norm(self.projected_gravity[:, :2], dim=1) > 0.1

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # return torch.square(base_height - self.cfg.rewards.base_height_target)
        return torch.abs(torch.clip(base_height - self.cfg.rewards.base_height_target, -1, 0))

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        vel_diff = self.last_dof_vel - self.dof_vel
        vel_diff[:,8:] = 0
        return torch.sum(torch.square(vel_diff / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        # return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                        #  dim=1)
        # change to reward not penalize
        return torch.sum((torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
      
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        # return -lin_vel_error

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1. * contacts, dim=1) == 1
        return 1. * single_contact
 
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.abs(self.commands[:, 0]) < 0.05)
  
    def _reward_feet_distance(self):
        feet_distance = torch.abs(torch.norm(self.feet_state[:, 0, :2] - self.feet_state[:, 1, :2], dim=-1))
        # reward = torch.abs(feet_distance - self.cfg.rewards.min_feet_distance)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        return reward

    def _reward_survival(self):
        # return (~self.reset_buf).float() * self.dt
        return (self.episode_length_buf * self.dt) > 10
       
    def _reward_leg_symmetry(self):
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        leg_symmetry_err = (abs(foot_positions_base[:,0,1])-abs(foot_positions_base[:,1,1]))
        return torch.exp(-(leg_symmetry_err ** 2)/ self.cfg.rewards.leg_symmetry_tracking_sigma)
    
    def _reward_wheel_adjustment(self):
        # 鼓励使用轮子的滑动克服前后的倾斜，奖励轮速和倾斜方向一致的情况，并要求轮速方向也一致
        incline_x = self.projected_gravity[:, 0]
        # mean velocity
        wheel_x_mean = (self.foot_velocities[:, 0, 0] + self.foot_velocities[:, 1, 0]) / 2
        # 两边轮速不一致的情况，不给奖励
        wheel_x_invalid = (self.foot_velocities[:, 0, 0] * self.foot_velocities[:, 1, 0]) < 0
        wheel_x_mean[wheel_x_invalid] = 0.0
        wheel_x_mean = wheel_x_mean.reshape(-1)
        reward = incline_x * wheel_x_mean > 0
        return reward
    
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    # 惩罚机器人两只脚在 X 方向上不对齐
    def _reward_same_foot_x_position(self):
        foot_positions_base = self.foot_positions - \
            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_x_position_err = foot_positions_base[:,0,0] - foot_positions_base[:,1,0] 
        penalty = torch.abs(foot_x_position_err)
        return penalty
    
    def _reward_leg_oscillation(self):
        # shape: (num_envs,)
        should_update = (self.episode_length_buf % self.cfg.rewards.dof_vel_history_length == 0)
        is_oscillating = self.check_oscillation(self.dof_vel_history)  # shape: (num_envs, 2)

        env_oscillate_left_leg = is_oscillating[:,0].float()
        env_oscillate_right_leg = is_oscillating[:,1].float()

        # if should_update:
        #     if env_oscillate_left_leg.any():
        #         print("Left leg oscillation")
        #     if env_oscillate_right_leg.any():
        #         print("Right leg oscillation")

        self.last_leg_oscillation_reward[should_update,0] = env_oscillate_left_leg[should_update]
        self.last_leg_oscillation_reward[should_update,1] = env_oscillate_right_leg[should_update]

        total_oscillation_reward = torch.sum(self.last_leg_oscillation_reward, dim=1)

        return total_oscillation_reward

    def check_oscillation(self, vel_history):
        hist = torch.stack(list(vel_history), dim=0)  # shape: (T, num_envs, 2)
        sign_changes = (torch.sign(hist[1:]) - torch.sign(hist[:-1])) != 0
        osc_score = sign_changes.sum(dim=0)           # shape: (num_envs, 2)
        
        zero_count = (hist == 0).sum(dim=0)  # (num_envs, 2)
        zero_thresh = self.cfg.rewards.dof_vel_history_length / 2.
        no_osc_mask = zero_count > zero_thresh
       
        osc_result = (osc_score > self.cfg.rewards.oscillation_sign_thresh) & (~no_osc_mask)
        return osc_result

    # ------------ airbot reward functions----------------

    def _reward_dof_acc_arm (self):
        # Penalize dof accelerations
        vel_diff = self.last_dof_vel - self.dof_vel
        vel_diff[:,:8] = 0
        return torch.sum(torch.square(vel_diff / self.dt), dim=1)

    def _reward_tracking_ee_cart(self):
        ee_pos_error = torch.sum(torch.square(self.ee_pos_local - self.curr_ee_goal_local) * self.sphere_error_scale, dim=1)

        horizontal_threshold = self.cfg.rewards.active_cartAndOrn_reward_threshold_horizontal  
        horizontal_mask = (torch.norm(self.projected_gravity[:, :2], dim=1) < horizontal_threshold).float()

        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        spin_threshold = self.cfg.rewards.active_cartAndOrn_reward_threshold_spin  
        spin_mask = (ang_vel_error < spin_threshold).float()

        return torch.exp(-ee_pos_error/ self.cfg.rewards.tracking_ee_cart_sigma) * horizontal_mask * spin_mask

    def _reward_tracking_ee_cart_l2(self):
        ee_pos_error = torch.sum(torch.square(self.ee_pos_local - self.curr_ee_goal_local) * self.sphere_error_scale, dim=1)
        return ee_pos_error
    
    def _reward_tracking_ee_orn(self):
        # Compute distance to target
        pos_err = torch.norm(self.ee_pos_local - self.curr_ee_goal_local, dim=1)
        pos_threshold = self.cfg.rewards.active_orn_reward_threshold  # meters, tune as needed
        close_mask = (pos_err < pos_threshold).float()

        horizontal_threshold = self.cfg.rewards.active_cartAndOrn_reward_threshold_horizontal
        horizontal_mask = (torch.norm(self.projected_gravity[:, :2], dim=1) < horizontal_threshold).float()

        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        spin_threshold = self.cfg.rewards.active_cartAndOrn_reward_threshold_spin  
        spin_mask = (ang_vel_error < spin_threshold).float()

        ee_orn_euler = euler_from_quat(self.ee_orn)
        ee_goal_orn_euler = euler_from_quat(self.ee_goal_orn_quat)
        orn_err = torch.sum(torch.abs((ee_goal_orn_euler - ee_orn_euler)) * self.orn_error_scale, dim=1)
        return torch.exp(-orn_err/self.cfg.rewards.tracking_ee_orn_sigma) * horizontal_mask * spin_mask#* close_mask 

    def _reward_tracking_ee_orn_l2(self):
        # Compute distance to target
        pos_err = torch.norm(self.ee_pos_local - self.curr_ee_goal_local, dim=1)
        pos_threshold = self.cfg.rewards.active_orn_reward_threshold  # meters, tune as needed
        close_mask = (pos_err < pos_threshold).float()

        ee_orn_euler = euler_from_quat(self.ee_orn)
        ee_goal_orn_euler = euler_from_quat(self.ee_goal_orn_quat)
        orn_err = torch.sum(torch.abs((ee_goal_orn_euler - ee_orn_euler)) * self.orn_error_scale, dim=1)
        return orn_err #* close_mask
    
    def _reward_com_pos(self):
        rigid_body_pos_tita = self.rigid_body_states[:,:9,:3]
        rigid_body_pos_arm = self.rigid_body_states[:,9:,:3]
        # -----------------------------
        # 底盘部分 COM
        total_mass_tita = torch.sum(self.rigid_body_mass_tita, dim=1)  # [num_envs, 1]
        weighted_chassis = torch.sum(rigid_body_pos_tita * self.rigid_body_mass_tita, dim=1)  # [num_envs, 3]
        self.com_tita = weighted_chassis / total_mass_tita              # [num_envs, 3]
        # -----------------------------
        # 机械臂部分 COM
        total_mass_arm = torch.sum(self.rigid_body_mass_arm, dim=1)  # [num_envs, 1]
        weighted_arm = torch.sum(rigid_body_pos_arm * self.rigid_body_mass_arm, dim=1)  # [num_envs, 3]
        self.com_arm = weighted_arm / total_mass_arm              # [num_envs, 3]
        
        com_err = torch.sum(torch.square((self.com_tita[:,:2] - self.com_arm[:,:2])), dim=1)
        return torch.exp(-com_err / self.cfg.rewards.com_pos_sigma)
    
    def _reward_arm_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = torch.square((self.dof_pos[:,8:] - self.dof_pos_limits[8:, 0]).clip(max=0.))  # lower limit
        out_of_limits += torch.square((self.dof_pos[:,8:] - self.dof_pos_limits[8:, 1]).clip(min=0.))
        return torch.sum(out_of_limits, dim=1)
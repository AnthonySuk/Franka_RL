import os
from typing import Dict, Tuple
import sys
import random

import torch
from torch import Tensor
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, euler_from_quat
from legged_gym.utils.terrain import Terrain

class TAFVisualWholebodyRobot:
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
        self.stand_by = self.cfg.env.stand_by

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
        #self.num_obs = cfg.env.num_proprio
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.proprioceptive_obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.arm_rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
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
        self.test_count = 0
        self.test_flag = 1

    def orientation_error(self,desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions[:, 8:] = 0. # mask out the arm action
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()

        # Action delay check
        if self.action_delay != -1:
            self.action_history_buf = torch.cat([self.action_history_buf[:, 1:], actions[:, None, :]], dim=1)
            # actions = self.action_history_buf[:, -self.action_delay - 1] # delay for 1/50=20ms
            if self.global_steps < 10000 * 24:
                actions = self.action_history_buf[:, -1]
            else:
                actions = self.action_history_buf[:, -2]

            self.actions = actions.clone()

        # Calculate arm target position using inverse kinematics
        dpos = self.curr_ee_goal_cart_world - self.ee_pos
        drot = self.orientation_error(self.ee_goal_orn_quat, self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1))
        dpose = torch.cat([dpos, drot], -1).unsqueeze(-1)
        arm_pos_targets = self._control_ik(dpose) + self.dof_pos[:, -(6 + self.cfg.env.num_gripper_joints):-self.cfg.env.num_gripper_joints]
        all_pos_targets = torch.zeros_like(self.dof_pos)
        #all_pos_targets[:, -(6 + self.cfg.env.num_gripper_joints):-self.cfg.env.num_gripper_joints] = arm_pos_targets

        for t in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions)
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(all_pos_targets))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        # clip_obs = self.cfg.normalization.clip_observations
        # self.proprioceptive_obs_buf = torch.clip(self.proprioceptive_obs_buf, -clip_obs, clip_obs)
        # if self.privileged_obs_buf is not None:
        #     self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # return self.proprioceptive_obs_buf, self.privileged_obs_buf, self.rew_buf, self.arm_rew_buf, self.reset_buf, self.extras

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.global_steps += 1

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.arm_rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.proprioceptive_obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self):
        """ Reset all robots"""
        self._update_curr_ee_goal()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
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

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        # add base position
        self.base_position = self.root_states[:, :3]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        base_yaw = euler_from_quat(self.base_quat)[:,2]
        self.base_yaw_euler[:] = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat[:] = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.dof_pos[:,[3, 7]]  = 0 

        if self.cfg.terrain.measure_heights_actor or self.cfg.terrain.measure_heights_critic:
            self.measured_heights = self._get_heights()
        self._compute_feet_states()

        self._post_physics_step_callback()

        # update ee goal
        self._update_curr_ee_goal()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_torques[:] = self.torques[:]
        # add foot position and base position
        self.last_foot_positions[:] = self.foot_positions[:]
        self.last_base_position[:] = self.base_position[:]
        
        if self.viewer :
            self.gym.clear_lines(self.viewer)
            self._draw_ee_goal_curr()
            self._draw_ee_goal_traj()
            self._draw_collision_bbox()

    def _check_if_include_feet_height_rewards(self):
        members = [attr for attr in dir(self.cfg.rewards.scales) if not attr.startswith("__")]
        for scale in members:
            if "feet_height" in scale:
                return True
        return False

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,dim=1)

        base_euler = euler_from_quat(self.base_quat)
        r = base_euler[:,0]
        p = base_euler[:,1]

        z = self.root_states[:, 2]

        r_term = torch.abs(r) > 0.8
        p_term = torch.abs(p) > 0.8
        z_term = z < 0.1
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        
        self.reset_buf |= self.time_out_buf
        # self.reset_buf |= r_term
        # self.reset_buf |= p_term
        # self.reset_buf |= z_term

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
        self._resample_ee_goal(env_ids, is_init=True)

        self._reset_buffers(env_ids)
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
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
        self.last_torques[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.last_feet_air_time[env_ids] = 0.
        self.current_max_feet_height[env_ids] = 0.
        self.last_max_feet_height[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.goal_timer[env_ids] = 0.
        # add foot positions and base position
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        #self.rew_buf /= 100

        self.arm_rew_buf[:] = 0.
        for i in range(len(self.arm_reward_functions)):
            name = self.arm_reward_names[i]
            rew = self.arm_reward_functions[i]()
            rew = rew * self.arm_reward_scales[name]
            self.arm_rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.arm_rew_buf[:] = torch.clip(self.arm_rew_buf[:], min=0.)
        # add termination reward after clipping
        if "arm_termination" in self.arm_reward_scales:
            rew = self._reward_termination()
            rew = rew * self.arm_reward_scales["arm_termination"]
            self.arm_rew_buf += rew
            self.episode_sums["arm_termination"] += rew

        #self.arm_rew_buf /= 100

    def compute_observations(self):
        """ Computes observations
        """
        # self.compute_proprioceptive_observations()
        # self.compute_privileged_observations()

        # self.obs_history_buf = torch.where(
        #     (self.episode_length_buf <= 1)[:, None, None], 
        #     torch.stack([self.proprioceptive_obs_buf] * self.cfg.env.history_len, dim=1),
        #     torch.cat([
        #         self.obs_history_buf[:, 1:],
        #         self.proprioceptive_obs_buf.unsqueeze(1)
        #     ], dim=1)
        # )

        # self._add_noise_to_obs()
        self.dof_err = self.dof_pos - self.default_dof_pos
        self.dof_err[:,self.feet_indices] = 0

        arm_base_pos = self.base_pos + quat_apply(self.base_yaw_quat, self.arm_base_offset)
        ee_goal_local_cart = quat_rotate_inverse(self.base_quat, self.curr_ee_goal_cart_world - arm_base_pos)

        if self.action_delay != -1:
            action_history = self.action_history_buf.squeeze(1)
        else:
            action_history = self.actions

        if self.stand_by:
            self.commands[:] = 0.

        obs_buf = torch.cat((      
                            self.base_lin_vel * self.obs_scales.lin_vel, # dim 3
                            self.base_ang_vel * self.obs_scales.ang_vel, # dim 3
                            self.projected_gravity, 
                            self.dof_err * self.obs_scales.dof_pos,
                            self.dof_vel * self.obs_scales.dof_vel,
                            self.commands[:, :3] * self.commands_scale,
                            action_history[:, :8],  # dim 8
                            ee_goal_local_cart,  # dim 3 position
                            0*self.curr_ee_goal_sphere  # dim 3 orientation
                            ), dim=-1)
        if self.cfg.env.observe_gait_commands:
            obs_buf = torch.cat((obs_buf,
                                      self.gait_indices.unsqueeze(1), self.clock_inputs), dim=-1)
            
        if self.cfg.domain_rand.observe_priv:
            priv_buf = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[:, :12] - 1,
            ), dim=-1)
            self.obs_buf = torch.cat([obs_buf, priv_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )        

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def compute_proprioceptive_observations(self):
        self._compose_proprioceptive_obs_buf_no_height_measure()
        if self.cfg.terrain.measure_heights_actor:
            self.proprioceptive_obs_buf = self._add_height_measure_to_buf(self.proprioceptive_obs_buf)
        if self.proprioceptive_obs_buf.shape[1] != self.num_obs:
            raise RuntimeError(
                f"obs_buf size ({self.proprioceptive_obs_buf.shape[1]}) does not match num_obs ({self.num_obs})")

    def _compose_proprioceptive_obs_buf_no_height_measure(self):
        #Wheel pos maybe unbounded when training,
        #since wheels are in velocity control mode, we don't wheel pos obs.
        self.dof_err = self.dof_pos - self.default_dof_pos
        self.dof_err[:,self.feet_indices] = 0

        arm_base_pos = self.base_pos + quat_apply(self.base_yaw_quat, self.arm_base_offset)
        ee_goal_local_cart = quat_rotate_inverse(self.base_quat, self.curr_ee_goal_cart_world - arm_base_pos)
        action_history = self.action_history_buf.squeeze(1)

        if self.stand_by:
            self.commands[:] = 0.

        self.proprioceptive_obs_buf = torch.cat((
                                                self.base_lin_vel * self.obs_scales.lin_vel, # dim 3
                                                self.base_ang_vel * self.obs_scales.ang_vel, # dim 3
                                                self.projected_gravity, 
                                                self.dof_err * self.obs_scales.dof_pos,
                                                self.dof_vel * self.obs_scales.dof_vel,
                                                self.commands[:, :3] * self.commands_scale,
                                                action_history[:, :8],  # dim 8
                                                ee_goal_local_cart,  # dim 3 position
                                                0*self.curr_ee_goal_sphere  # dim 3 orientation
                                                ), dim=-1)

    def _add_height_measure_to_buf(self, buf):
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                             1.) * self.obs_scales.height_measurements
        buf = torch.cat(
            (buf, heights), dim=-1
        )
        return buf
    
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
        self.dof_err = self.dof_pos - self.default_dof_pos
        self.dof_err[:,self.feet_indices] = 0

        arm_base_pos = self.base_pos + quat_apply(self.base_yaw_quat, self.arm_base_offset)
        ee_goal_local_cart = quat_rotate_inverse(self.base_quat, self.curr_ee_goal_cart_world - arm_base_pos)
        action_history = self.action_history_buf.squeeze(1)
        if self.stand_by:
            self.commands[:] = 0.

        self.privileged_obs_buf = torch.cat((
                                            self.base_lin_vel * self.obs_scales.lin_vel, # dim 3
                                            self.base_ang_vel * self.obs_scales.ang_vel, # dim 3
                                            self.projected_gravity, 
                                            self.dof_err * self.obs_scales.dof_pos,
                                            self.dof_vel * self.obs_scales.dof_vel,
                                            self.commands[:, :3] * self.commands_scale,
                                            action_history[:, :8],  # dim 8
                                            ee_goal_local_cart,  # dim 3 position
                                            0*self.curr_ee_goal_sphere  # dim 3 orientation
                                            ), dim=-1)

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

    def _parse_cfg(self):
        self.num_torques = self.cfg.env.num_torques
        self.dt = self.cfg.control.decimation * self.sim_params.dt  # dt for action update
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.arm_reward_scales = class_to_dict(self.cfg.rewards.arm_scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.goal_ee_ranges = class_to_dict(self.cfg.goal_ee.ranges)

        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.clip_actions = self.cfg.normalization.clip_actions
        self.action_delay = self.cfg.env.action_delay
        self.stop_update_goal = self.cfg.env.stop_update_goal
        self.record_video = self.cfg.env.record_video

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0 or scale == None :
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        
        for key in list(self.arm_reward_scales.keys()):
            scale = self.arm_reward_scales[key]
            if scale == 0 or scale == None :
                self.arm_reward_scales.pop(key)
            else:
                self.arm_reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # prepare list of functions
        self.arm_reward_functions = []
        self.arm_reward_names = []
        for name, scale in self.arm_reward_scales.items():
            if name=="termination":
                continue
            self.arm_reward_names.append(name)
            name = '_reward_' + name
            self.arm_reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in list(self.reward_scales.keys()) + list(self.arm_reward_scales.keys())}
        
        self.episode_metric_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                        for name in list(self.reward_scales.keys()) + list(self.arm_reward_scales.keys())}

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
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),(self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
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
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,requires_grad=False)
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

        if self.cfg.domain_rand.randomize_gripper_mass:
            gripper_rng_mass = self.cfg.domain_rand.gripper_added_mass_range
            gripper_rand_mass = np.random.uniform(gripper_rng_mass[0], gripper_rng_mass[1], size=(1, ))
            props[self.gripper_idx].mass += gripper_rand_mass
        else:
            gripper_rand_mass = np.zeros(1)

        if self.cfg.domain_rand.randomize_base_com:
            com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
            props[0].com.x += np.random.uniform(-com_x, 0) # props[0].com.x += np.random.uniform(-com_x, com_x)
            props[0].com.y += np.random.uniform(-com_y, com_y)
            props[0].com.z += np.random.uniform(-com_z, com_z)

        mass_params = np.concatenate([gripper_rand_mass])
        return props, mass_params

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self._resample(env_ids)
        self._step_contact_targets()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample(self, env_ids):
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
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
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
    
    def _resample_ee_goal_sphere_once(self, env_ids):
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(self.goal_ee_ranges["pos_l"][0], self.goal_ee_ranges["pos_l"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(self.goal_ee_ranges["pos_p"][0], self.goal_ee_ranges["pos_p"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(self.goal_ee_ranges["pos_y"][0], self.goal_ee_ranges["pos_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    
    def _resample_ee_goal_orn_once(self, env_ids):
        ee_goal_delta_orn_r = torch_rand_float(self.goal_ee_ranges["delta_orn_r"][0], self.goal_ee_ranges["delta_orn_r"][1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_p = torch_rand_float(self.goal_ee_ranges["delta_orn_p"][0], self.goal_ee_ranges["delta_orn_p"][1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_y = torch_rand_float(self.goal_ee_ranges["delta_orn_y"][0], self.goal_ee_ranges["delta_orn_y"][1], (len(env_ids), 1), device=self.device)
        self.ee_goal_orn_delta_rpy[env_ids, :] = torch.cat([ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1)

    def _resample_ee_goal(self, env_ids, is_init=False):
        if self.cfg.env.teleop_mode and is_init:
            self.curr_ee_goal_sphere[:] = self.init_start_ee_sphere[:]
            return
        elif self.cfg.env.teleop_mode:
            return

        if len(env_ids) > 0:
            init_env_ids = env_ids.clone()
            
            if is_init:
                self.ee_goal_orn_delta_rpy[env_ids, :] = 0
                self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:]
                self.ee_goal_sphere[env_ids] = self.init_end_ee_sphere[:]
            else:
                self._resample_ee_goal_orn_once(env_ids)
                self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
                for i in range(10):
                    self._resample_ee_goal_sphere_once(env_ids)
                    collision_mask = self._collision_check(env_ids)
                    env_ids = env_ids[collision_mask]
                    if len(env_ids) == 0:
                        break
            self.ee_goal_cart[init_env_ids, :] = self.sphere2cart(self.ee_goal_sphere[init_env_ids, :])
            self.goal_timer[init_env_ids] = 0.0

    def _collision_check(self, env_ids):
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ...,  None], self.collision_check_t).squeeze(-1)
        ee_target_cart = self.sphere2cart(torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(torch.logical_and(torch.all(ee_target_cart < self.collision_upper_limits, dim=-1), torch.all(ee_target_cart > self.collision_lower_limits, dim=-1)), dim=0)
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    def cart2sphere(self,cart):
        sphere = torch.zeros_like(cart)
        sphere[:, 0] = torch.norm(cart, dim=-1)
        sphere[:, 1] = torch.atan2(cart[:, 2], cart[:, 0])
        sphere[:, 2] = torch.asin(cart[:, 1] / sphere[:, 0])
        return sphere

    def sphere2cart(self,sphere):
        if sphere.ndim == 1:
            sphere = sphere.view(1, -1)  # or use unsqueeze(0)

        cart = torch.zeros_like(sphere)
        cart[:, 0] = sphere[:, 0] * torch.cos(sphere[:, 2]) * torch.cos(sphere[:, 1])
        cart[:, 1] = sphere[:, 0] * torch.sin(sphere[:, 2])
        cart[:, 2] = sphere[:, 0] * torch.cos(sphere[:, 2]) * torch.sin(sphere[:, 1])
        return cart

    def _update_curr_ee_goal(self):
        if not self.cfg.env.teleop_mode:
            t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
            self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])

        # TODO: for the teleop mode, we need to directly update self.curr_ee_goal_cart using VR controller.
        self.curr_ee_goal_cart[:] = self.sphere2cart(self.curr_ee_goal_sphere)
        ee_goal_cart_yaw_global = quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)
        self.curr_ee_goal_cart_world = self._get_ee_goal_spherical_center() + ee_goal_cart_yaw_global
        
        # TODO: for the teleop mode, we need to directly update self.ee_goal_orn_quat using VR controller.
        default_yaw = torch.atan2(ee_goal_cart_yaw_global[:, 1], ee_goal_cart_yaw_global[:, 0])
        default_pitch = -self.curr_ee_goal_sphere[:, 1] + self.cfg.goal_ee.arm_induced_pitch
        self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_delta_rpy[:, 0] + np.pi / 2, default_pitch + self.ee_goal_orn_delta_rpy[:, 1], self.ee_goal_orn_delta_rpy[:, 2] + default_yaw)
        
        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        
        if len(resample_id) > 0 and self.stop_update_goal:
            # set these env commands as 0
            self.commands[resample_id, 0] = 0
            self.commands[resample_id, 2] = 0

        self._resample_ee_goal(resample_id)
    
    def _get_ee_goal_spherical_center(self):
        center = torch.cat([self.root_states[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        center = center + quat_apply(self.base_yaw_quat, self.ee_goal_center_offset)
        return center

    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            frequencies = self.cfg.env.frequencies
            phases = 0.5
            offsets = 0
            bounds = 0
            durations = 0.5
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)
            self.gait_indices[~self._get_walking_cmd_mask()] = 0

            foot_indices = [self.gait_indices + phases + offsets + bounds,
                            self.gait_indices + offsets,
                            self.gait_indices + bounds,
                            self.gait_indices + phases]

            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations)
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations) * (
                            0.5 / (1 - durations))

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                    kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions
        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        action_scale_vel = self.cfg.control.action_scale_vel
        modify_dof_vel = self.dof_vel.clone().detach()
        modify_dof_vel[:,self.arm_indices] = 0

        # get scale action
        actions_scaled = actions * self.motor_strength * self.action_scale

        # pd controller 
        torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel

        # mask out arm torque
        torques[:, -6:] = 0

        # calculate torque for the wheel
        V_list = [3,7]
        torques[:,V_list] = self.d_gains[V_list]*(action_scale_vel* actions[:,V_list] - modify_dof_vel[:,V_list])
        #print("gripper torque", torques[:,13])
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits) # torch.clip() 用于限制 torques 的范围，防止超出 扭矩限制。

    def _compute_torques_armpose(self, actions):
        ik_u = self.control_ik(self._local_gripper_pos,self.curr_ee_goal,self.j_eef)
        self.arm_u[:,self.arm_indices] = self.dof_pos[:,self.arm_indices]  + actions[:,self.arm_indices] + ik_u
        return self.arm_u
    
    def control_ik(self,local_ee_pose,local_goal_pose,local_j_eef):
        pos_err = local_goal_pose[:,0:3] - local_ee_pose[:,0:3]
        orn = torch.tensor([0,0,0,1], device=self.device).repeat(self.num_envs, 1)
        orn_err = self.orientation_error(orn, orn)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        # solve damped least squares
        j_eef_T = torch.transpose(local_j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
        u = (j_eef_T @ torch.inverse(local_j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)
        return u

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
        self.gym.refresh_rigid_body_state_tensor(self.sim)
    
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
        # base orientation
        rand_yaw = self.cfg.init_state.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
        quat = quat_from_euler_xyz(0*rand_yaw, 0*rand_yaw, rand_yaw) 
        self.root_states[env_ids, 3:7] = quat[:, :]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.refresh_actor_root_state_tensor(self.sim)

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
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        obs_noise_vec = torch.zeros(self.cfg.env.num_observations, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        # Base linear velocity (dim 3)
        obs_noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        # Base angular velocity (dim 3)
        obs_noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # Base projected gravity (dim 3)
        obs_noise_vec[6:9] = noise_scales.gravity * noise_level
        # Dof pos error (dim 8 + 6 = 14)
        dof_pos_end_idx = 9 + self.num_dof
        obs_noise_vec[9:dof_pos_end_idx] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # Dof vel error (dim 8 + 6 = 14)
        dof_vel_end_idx = dof_pos_end_idx + self.num_dof
        obs_noise_vec[dof_pos_end_idx:dof_vel_end_idx] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # Command (dim 3)
        command_end_idx = dof_vel_end_idx + self.cfg.commands.num_commands
        obs_noise_vec[dof_vel_end_idx:command_end_idx] = 0.  # commands
        # Action history (dim 8)
        last_action_end_idx = dof_vel_end_idx + 8
        obs_noise_vec[command_end_idx:last_action_end_idx] = 0.  # previous actions
        # End-effector goal position (dim 3)
        ee_goal_cart_idx = last_action_end_idx + 3
        obs_noise_vec[last_action_end_idx:ee_goal_cart_idx] = 0
        # End-effector orientation (dim 3)
        ee_goal_sphere_idx = ee_goal_cart_idx + 3
        obs_noise_vec[ee_goal_cart_idx:ee_goal_sphere_idx] = 0

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
    
    def _get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self.base_quat)
        body_angles = torch.stack([r, p, y], dim=-1)

        if not return_yaw:
            return body_angles[:, :-1]
        else:
            return body_angles
        
    def _draw_collision_bbox(self):

        center = self.ee_goal_center_offset
        bbox0 = center + self.collision_upper_limits
        bbox1 = center + self.collision_lower_limits
        bboxes = torch.stack([bbox0, bbox1], dim=1)
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))

        for i in range(self.num_envs):
            bbox_geom = gymutil.WireframeBBoxGeometry(bboxes[i], None, color=(1, 0, 0))
            quat = self.base_yaw_quat[i]
            r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            pose0 = gymapi.Transform(gymapi.Vec3(self.root_states[i, 0], self.root_states[i, 1], 0), r=r)
            gymutil.draw_lines(bbox_geom, self.gym, self.viewer, self.envs[i], pose=pose0) 

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
        gymutil.draw_lines(sphere_geom_origin, self.gym, self.viewer, self.envs[0], sphere_pose)

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

    def _control_ik(self, dpose):
        # solve damped least squares
        j_eef_T = torch.transpose(self.ee_j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
        A = torch.bmm(self.ee_j_eef, j_eef_T) + lmbda[None, ...]
        u = torch.bmm(j_eef_T, torch.linalg.solve(A, dpose))#.view(self.num_envs, 6)
        return u.squeeze(-1)

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.action_scale = torch.tensor(self.cfg.control.action_scale, device=self.device)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)
        dof_torque_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 2, 6)
        self.dof_torque = gymtorch.wrap_tensor(dof_torque_tensor).view(self.num_envs, self.num_dof, -1)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_pos_wo_gripper = self.dof_pos[:, :-self.cfg.env.num_gripper_joints]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_vel_wo_gripper = self.dof_vel[:, :-self.cfg.env.num_gripper_joints]
        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, :3]
        self.arm_base_offset = torch.tensor([0.3, 0., 0.09], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        base_yaw = get_euler_xyz(self.base_quat)[2]
        self.base_yaw_euler = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        
        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.action_delay + 2, self.num_actions, device=self.device, dtype=torch.float)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, -1)

        self.jacobian_whole = gymtorch.wrap_tensor(jacobian_tensor)
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
        self.foot_velocities_f = torch.zeros_like(self.foot_positions)
        self.foot_relative_velocities = torch.zeros_like(self.foot_velocities)
        
        # ee info
        self.ee_pos = self.rigid_body_states[:, self.gripper_idx, :3]
        self.ee_orn = self.rigid_body_states[:, self.gripper_idx, 3:7]
        self.ee_vel = self.rigid_body_states[:, self.gripper_idx, 7:]
        self.ee_j_eef = self.jacobian_whole[:, self.gripper_idx, :6, -(6 + self.cfg.env.num_gripper_joints):-self.cfg.env.num_gripper_joints]

        # target_ee info
        self.grasp_offset = self.cfg.arm.grasp_offset
        self.init_target_ee_base = torch.tensor(self.cfg.arm.init_target_ee_base, device=self.device).unsqueeze(0)

        self.traj_timesteps = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler[:, 0] = np.pi / 2
        self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_euler[:, 0], self.ee_goal_orn_euler[:, 1], self.ee_goal_orn_euler[:, 2])
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)

        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)

        self.init_start_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_start, device=self.device).unsqueeze(0)
        self.init_end_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_end, device=self.device).unsqueeze(0)
        
        #noise
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.add_noise = self.cfg.noise.add_noise

        self.collision_lower_limits = torch.tensor(self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float)
        self.collision_upper_limits = torch.tensor(self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float)
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        assert(self.cfg.goal_ee.command_mode in ['cart', 'sphere'])
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)
        self.ee_goal_center_offset = torch.tensor([self.cfg.goal_ee.sphere_center.x_offset, 
                                                   self.cfg.goal_ee.sphere_center.y_offset, 
                                                   self.cfg.goal_ee.sphere_center.z_invariant_offset], 
                                                   device=self.device).repeat(self.num_envs, 1)

        self.curr_ee_goal_cart_world = self._get_ee_goal_spherical_center() + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)

        print('------------------------------------------------------')
        print(f'root_states shape: {self.root_states.shape}')
        print(f'dof_state shape: {self.dof_state.shape}')
        print(f'force_sensor_tensor shape: {self.force_sensor_tensor.shape}')
        print(f'contact_forces shape: {self.contact_forces.shape}')
        print(f'rigid_body_state shape: {self.rigid_body_states.shape}')
        print(f'jacobian_whole shape: {self.jacobian_whole.shape}')
        print('------------------------------------------------------')

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_torques = torch.zeros_like(self.torques)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],device=self.device, requires_grad=False, )  # TODO change this
        
        self.gripper_torques_zero = torch.zeros(self.num_envs, self.cfg.env.num_gripper_joints, device=self.device)
        
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,device=self.device, requires_grad=False)
        self.last_feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,device=self.device, requires_grad=False)
        self.first_contact = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,requires_grad=False)
        self.feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,device=self.device, requires_grad=False)
        self.last_max_feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,device=self.device, requires_grad=False)
        self.current_max_feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.rigid_body_external_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False)
        self.rigid_body_external_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)

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
            print("###Current joint ",name," P: ",self.p_gains[i]," D: ",self.d_gains[i])
        
        #self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_wo_gripper = self.default_dof_pos[:-self.cfg.env.num_gripper_joints]
        self.global_steps = 0

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

        # Robot
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.robot_asset = robot_asset
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        dof_props_asset['driveMode'][8:].fill(gymapi.DOF_MODE_POS)  # set arm to pos control
        dof_props_asset['stiffness'][8:].fill(400.0)
        dof_props_asset['damping'][8:].fill(40.0)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_wo_gripper_names = self.dof_names[:-self.cfg.env.num_gripper_joints]
        self.dof_names_to_idx = self.gym.get_asset_dof_dict(robot_asset) 
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        
        self.sensor_indices = []
        for name in feet_names:
            foot_idx = self.body_names_to_idx[name]
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, -0.05))
            sensor_idx = self.gym.create_asset_force_sensor(robot_asset, foot_idx, sensor_pose)
            self.sensor_indices.append(sensor_idx)

        self.gripper_idx = self.body_names_to_idx[self.cfg.asset.gripper_name]

        print('------------------------------------------------------')
        print('num_actions: {}'.format(self.num_actions))
        print('num_torques: {}'.format(self.num_torques))
        print('num_dofs: {}'.format(self.num_dofs))
        print('num_bodies: {}'.format(self.num_bodies))
        print('body name: {}'.format(body_names))
        print('penalized_contact_names: {}'.format(penalized_contact_names))
        print('termination_contact_names: {}'.format(termination_contact_names))
        print('feet_names: {}'.format(feet_names))
        print(f"EE Gripper index: {self.gripper_idx}")

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        self.base_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 5, dtype=torch.float, device=self.device, requires_grad=False)
        for j in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)

            # Tita
            pos = self.env_origins[j].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, j)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, j,self.cfg.asset.self_collisions, 0)
            self.actor_handles.append(actor_handle)

            dof_props = self._process_dof_props(dof_props_asset, j)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, j)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            self.mass_params_tensor[j, :] = torch.from_numpy(mass_params).to(self.device)

        assert(np.all(np.array(self.actor_handles) == 0))
        self.robot_actor_indices = torch.arange(0, 2 * self.num_envs, 2, device=self.device)

        self.motor_strength = torch.ones(self.num_envs, self.num_torques, device=self.device)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])
        
        arm_names =[]
        for name in self.cfg.asset.arm_joint_name:
            arm_names.extend([s for s in self.dof_names if name in s])

        self.arm_indices = torch.zeros(len(arm_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(arm_names)):
            self.arm_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], arm_names[i])
            
        print('penalized_contact_indices: {}'.format(self.penalised_contact_indices))
        print('termination_contact_indices: {}'.format(self.termination_contact_indices))
        print('feet_indices: {}'.format(self.feet_indices))
        
        if self.record_video:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 720
            camera_props.height = 480
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                # root_pos = self.root_states[i, :3].cpu().numpy()
                # cam_pos = root_pos + np.array([0, 1, 0.5])
                cam_pos = np.array([0, 1, 0.5])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0*cam_pos))

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

    # ------------ reward functions----------------
    # 计算 Z 轴方向上线速度的平方，如果速度越大惩罚越狠，限制Z轴上过大的运动，避免偏离平面或飞离地面
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    # 限制机器人在 XY 平面的旋转，防止它旋转过快
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        # torch.square 计算这些角速度的 平方值，这样可以避免负数对惩罚项的影响（负角速度仍然会被惩罚）。
        # tirch.sum 在每个环境实例（batch）内 对 XY 轴的角速度平方求和，得到一个标量惩罚值。
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    # 惩罚机器人非水平的底座姿态
    def _reward_orientation(self):
        # Penalize non flat base orientation
        # self.projected_gravity 代表投影到机器人坐标系中的重力向量 [:, :2] 选取 X 和 Y 轴分量
        # torch.norm 计算 X 和 Y 方向重力分量的欧几里得范数,如果机器人底座是完全水平的 X 和 Y 方向的分量应该接近零
        # 如果计算出的范数超过 0.1，说明机器人底座不是完全水平的
        return torch.norm(self.projected_gravity[:, :2], dim=1) > 0.1

    # 惩罚机器人机体远离目标高度
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # return torch.square(base_height - self.cfg.rewards.base_height_target)
        return torch.abs(torch.clip(base_height - self.cfg.rewards.base_height_target, -1, 0))

     # 惩罚过大的执行器扭矩，鼓励智能体使用更小的控制力以提高能效
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    # 励项鼓励智能体维持较低的关节速度，惩罚过高速度变化
    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

     # 惩罚过大的加速度，鼓励智能体的运动更加平滑
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    # 惩罚过大的动作变化
    def _reward_action_rate(self):
        # Penalize changes in actions
        #print("Last action: {}".format(self.last_actions[0,:]))
        #print("Current action: {}".format(self.last_actions[0,:]))
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # 算发生碰撞的次数，并累加这些碰撞事件，使其变成一种奖励机制; 智能体将更倾向于发生碰撞
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        # return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                        #  dim=1)
        # change to reward not penalize
        return torch.sum((torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)


    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # 只惩罚y速度跟不上（在y指令为0时代表惩罚侧倾的现象）
        # lin_vel_error = torch.abs(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma) 
        # return -lin_vel_error

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward steps between proper duration
        rew_airTime_below_min = torch.sum(
            torch.min(self.feet_air_time - self.cfg.rewards.min_feet_air_time,
                      torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime_above_max = torch.sum(
            torch.min(self.cfg.rewards.max_feet_air_time - self.feet_air_time,
                      torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime = rew_airTime_below_min + rew_airTime_above_max
        return rew_airTime

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1. * contacts, dim=1) == 1
        return 1. * single_contact

    def _reward_unbalance_feet_air_time(self):
        return torch.var(self.last_feet_air_time, dim=-1)

    def _reward_unbalance_feet_height(self):
        return torch.var(self.last_max_feet_height, dim=-1)

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        dof_err = self.dof_pos - self.default_dof_pos
        #dof_err[:,self.arm_indices] = 0
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_joint_pos_rate(self):
        # Penalize motion at zero commands
        dof_err = self.last_dof_pos - self.default_dof_pos
        return torch.sum(torch.square(dof_err), dim=1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    # 计算机器人脚之间的距离并生成奖励，确保它们保持在期望范围内
    def _reward_feet_distance(self):
        feet_distance = torch.abs(torch.norm(self.feet_state[:, 0, :2] - self.feet_state[:, 1, :2], dim=-1))
        # reward = torch.abs(feet_distance - self.cfg.rewards.min_feet_distance)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        return reward

    def _reward_survival(self):
        # return (~self.reset_buf).float() * self.dt
        return (self.episode_length_buf * self.dt) > 10
    
    # 奖励智能体的足部位置保持在理想高度，确保机器人在运动过程中维持合理的步态
    def _reward_nominal_foot_position(self):
        #1. calculate foot postion wrt base in base frame  
        nominal_base_height = -(self.cfg.rewards.base_height_target- self.cfg.asset.foot_radius) # 机器人底盘应当保持的理想高度
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1) # 足部相对于底盘的位置
        reward = 0
        # 转换到机器人底盘坐标系
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] ) #  将足部位置变换到底盘参考系
            height_error = nominal_base_height - foot_positions_base[:, i, 2]
            reward += torch.exp(-(height_error ** 2)/ self.cfg.rewards.nominal_foot_position_tracking_sigma)
        # 如果机器人速度较高，则奖励减少，以鼓励其在移动过程中保持稳定
        vel_cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        return reward / len(self.feet_indices)*torch.exp(-(vel_cmd_norm ** 2)/self.cfg.rewards.nominal_foot_position_tracking_sigma_wrt_v)
    
    # 奖励机器人两腿的对称性，确保步态协调
    def _reward_leg_symmetry(self):
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        leg_symmetry_err = (abs(foot_positions_base[:,0,1])-abs(foot_positions_base[:,1,1]))
        return torch.exp(-(leg_symmetry_err ** 2)/ self.cfg.rewards.leg_symmetry_tracking_sigma)
    
    # 奖励机器人两只脚在 Z 方向上的对齐
    def _reward_same_foot_z_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_z_position_err = foot_positions_base[:,0,2] - foot_positions_base[:,1,2]
        return foot_z_position_err ** 2

    # 奖励机器人两只脚在 X 方向上的对齐
    def _reward_same_foot_x_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_x_position_err = foot_positions_base[:,0,0] - foot_positions_base[:,1,0] # 0,0 1,0
        # foot_x_position_sigma 控制误差对奖励的影响，值越小要求越严格。
        # reward = torch.exp(-(foot_x_position_err ** 2)/ self.cfg.rewards.foot_x_position_sigma) 
        # return reward
        penalty = torch.abs(foot_x_position_err)
        return penalty
    
    def _reward_feet_vel(self):
        reward = torch.norm(self.foot_velocities[:, 0, :], dim=1) + torch.norm(self.foot_velocities[:, 1, :], dim=1)
        return reward
    
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
    
    def _reward_inclination(self):
        # 惩罚pitch和roll方向的角速度，防止侧倾
        rp_error = torch.norm(self.base_ang_vel[:, :2], dim=1) # commands前两个维度是速度，和角速度无关
        return rp_error

    def _reward_delta_torques(self):
        rew = torch.sum(torch.square(self.torques - self.last_torques)[:, :8], dim=1)
        return rew

        # -------------Z1: Reward functions----------------
    def _reward_tracking_ee_sphere(self):
        ee_pos_local = quat_rotate_inverse(self.base_yaw_quat, self.ee_pos - self.get_ee_goal_spherical_center())
        ee_pos_error = torch.sum(torch.abs(cart2sphere(ee_pos_local) - self.curr_ee_goal_sphere) * self.sphere_error_scale, dim=1)
        return torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma)
    
    def _reward_tracking_ee_world(self):
        ee_pos_error = torch.sum(torch.abs(self.ee_pos - self.curr_ee_goal_cart_world), dim=1)
        rew = torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma * 2)
        return rew
    
    def _reward_tracking_ee_sphere_walking(self):
        reward, metric = self._reward_tracking_ee_sphere()
        walking_mask = self._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward

    def _reward_tracking_ee_sphere_standing(self):
        reward, metric = self._reward_tracking_ee_sphere()
        walking_mask = self._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward

    def _reward_tracking_ee_cart(self):
        target_ee = self.get_ee_goal_spherical_center() + quat_apply(self.env.base_yaw_quat, self.env.curr_ee_goal_cart)
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - target_ee), dim=1)
        return torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma), ee_pos_error
    
    def _reward_tracking_ee_orn(self):
        ee_orn_euler = torch.stack(euler_from_quat(self.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs(torch_wrap_to_pi_minuspi(self.ee_goal_orn_euler - ee_orn_euler)) * self.orn_error_scale, dim=1)
        return torch.exp(-orn_err/self.cfg.rewards.tracking_ee_sigma)

    def _reward_arm_energy_abs_sum(self):
        energy = torch.sum(torch.abs(self.torques[:, 8:-self.cfg.env.num_gripper_joints] * self.env.dof_vel[:, 8:-self.env.cfg.num_gripper_joints]), dim = 1)
        return energy

    def _reward_tracking_ee_orn_ry(self):
        ee_orn_euler = torch.stack(euler_from_quat(self.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs((torch_wrap_to_pi_minuspi(self.ee_goal_orn_euler - ee_orn_euler) * self.orn_error_scale)[:, [0, 2]]), dim=1)
        return torch.exp(-orn_err/self.cfg.rewards.tracking_ee_sigma)

    # Penalize motion at zero commands
    def _reward_joint_pos_rate(self):
        dof_err = self.last_dof_pos - self.default_dof_pos
        rew = torch.sum(torch.square(dof_err), dim=1)
        return rew
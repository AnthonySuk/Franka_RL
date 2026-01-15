import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
import torch
from scipy.spatial.transform import Rotation as R

# mujoco version == 3.1.4 very important

def quat_to_grav(q):
    q = np.asarray(q)
    v = np.array([0, 0, -1], dtype=np.float32)
    q_w = q[..., -1]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0)[..., np.newaxis]
    b = 2.0 * q_w[..., np.newaxis] * np.cross(q_vec, v)
    c = 2.0 * q_vec * np.sum(q_vec * v, axis=-1)[..., np.newaxis]
    return a - b + c

def reindex(qpos):
    return qpos[[4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 12, 13]]

def pd_control(default_dof_pos, actions, q, kp, dq, kd):
    torques = (actions * 0.25 + default_dof_pos - q) * kp - dq * kd
    torques[[3,7]] = kd[[3,7]] * ( 8.0 * actions[[3,7]] - dq[[3,7]])
    torques[8:] = kp[8:] * (actions[8:] - q[8:]) - kd[8:] * dq[8:]
    return torques

def get_obs(data, gripper_idx):
    '''Extracts an observation from the mujoco data structure
    '''
    cart = data.qpos[0:3]
    quat = data.qpos[3:7]
    q = data.qpos[7:]
    quat = [quat[1], quat[2], quat[3], quat[0]]

    vel = data.qvel[0:3]
    omega = data.qvel[3:6]
    dq = data.qvel[6:]

    proj_grav = quat_to_grav(quat)

    ee_pos = data.xpos[gripper_idx]
    ee_quat = data.xquat[gripper_idx]

    return cart, q, dq, omega, proj_grav, vel, quat,ee_pos, ee_quat
        # cart, q, dq, omega, grav, vel, quat, ee_pos, ee_quat

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

# ------------ ee goal related ------------
def _get_ee_goal_spherical_center(cart, base_yaw_quat, ee_goal_center_offset, _device):
    center = torch.cat([cart[:2], torch.zeros(1, device=_device)])
    center = center + quat_apply(base_yaw_quat, ee_goal_center_offset)
    return center

def sphere2cart(sphere):
    if sphere.ndim == 1:
        sphere = sphere.view(1, -1)  # or use unsqueeze(0)
    cart = torch.zeros_like(sphere)
    cart[:, 0] = sphere[:, 0] * torch.cos(sphere[:, 2]) * torch.cos(sphere[:, 1])
    cart[:, 1] = sphere[:, 0] * torch.sin(sphere[:, 2])
    cart[:, 2] = sphere[:, 0] * torch.cos(sphere[:, 2]) * torch.sin(sphere[:, 1])
    return cart.squeeze(0)  # 保证输出 shape 为 [3] 而不是 [1, 3]

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


def quat_from_euler_xyz(roll, pitch, yaw):
    # 输入可以是标量或张量
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([x, y, z, w])

def get_euler_xyz(quat):
    # quat: [w, x, y, z] or [x, y, z, w]
    # 这里假设输入为 [x, y, z, w]
    x, y, z, w = quat
    # roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)
    # pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)
    # yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)
    return torch.stack([roll, pitch, yaw])

def torch_rand_float(low, high, shape, device=None):
    return (high - low) * torch.rand(shape, device=device) + low

def quat_conjugate(q):
    # q: [x, y, z, w]
    x, y, z, w = q
    return torch.stack([-x, -y, -z, w])

def quat_apply(q, v):
    # q: [x, y, z, w], v: [3]
    # 结果 shape: [3]
    qvec = q[:3]
    w = q[3]
    t = 2.0 * torch.cross(qvec, v, dim=-1)
    return v + w * t + torch.cross(qvec, t, dim=-1)

def quat_rotate_inverse(q, v):
    # 用四元数的共轭实现逆旋转
    q_conj = quat_conjugate(q)
    return quat_apply(q_conj, v)

def run_mujoco(policy, cfg, device):
    policy = policy.to(device)
    policy.eval()

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)

    for i in range(model.nu):
        joint_id = model.actuator_trnid[i][0]  # actuator i → joint ID
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"ctrl[{i}] -> actuator '{actuator_name}' -> joint '{joint_name}'")

    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    default_pos = [0.0, 0.8, -1.5, 0.0, 0.0, 0.8, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = default_pos
    # In MuJoCO, the first 7 elements are reserved for the root position and orientation, 
    # the rest are for the robot's joints.
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    hist_obs    = deque([np.zeros([1,71],dtype=np.float32) for _ in range(cfg.sim_config.frame_stack)])
    obs_history = deque([np.zeros([1,71],dtype=np.float32) for _ in range(cfg.sim_config.o_h_frame_stack)])
    count_low   = 0
    actions     = np.zeros(14, dtype=np.float32)
    cmd         = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    total_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)

    gripper_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.robot_config.end_effector_name)

    num_joints = 14
    dof_pos_limits_min = model.jnt_range[:num_joints, 0]
    dof_pos_limits_max = model.jnt_range[:num_joints, 1]
    dof_pos_limits_min = np.array(dof_pos_limits_min)
    dof_pos_limits_max = np.array(dof_pos_limits_max)

    # ee goal timer setting
    goal_timer = torch.zeros(1, device=device)
    traj_timesteps = torch_rand_float(cfg.goal_ee.traj_time[0], cfg.goal_ee.traj_time[1], (), device=device) / cfg.sim_config.dt
    traj_total_timesteps = traj_timesteps + torch_rand_float(cfg.goal_ee.hold_time[0], cfg.goal_ee.hold_time[1], 
                                                                        (), device=device) / cfg.sim_config.dt
    # ee goal variables setting
    z_invariant_offset = torch.as_tensor([cfg.goal_ee.sphere_center.z_invariant_offset], device=device, dtype=torch.float32)

    curr_ee_goal_cart = torch.zeros(3, device=device)
    curr_ee_goal_sphere = torch.zeros(3, device=device)
    curr_ee_goal_cart_world = curr_ee_goal_cart
    curr_ee_goal_local = curr_ee_goal_cart

    ee_start_sphere = torch.zeros(3, device=device)
    ee_goal_cart = torch.zeros(3, device=device)
    ee_goal_sphere = torch.zeros(3, device=device)
    ee_goal_orn_delta_rpy = torch.zeros(3, device=device)

    ee_goal_center_offset = torch.tensor([cfg.goal_ee.sphere_center.x_offset, 
                                          cfg.goal_ee.sphere_center.y_offset, 
                                          cfg.goal_ee.sphere_center.z_invariant_offset], 
                                                device=device)
    
    goal_ee_ranges = class_to_dict(cfg.goal_ee.ranges)

    init_start_ee_sphere = torch.tensor(cfg.goal_ee.ranges.init_pos_start, device=device)
    init_end_ee_sphere = torch.tensor(cfg.goal_ee.ranges.init_pos_end, device=device)

    ee_goal_orn_delta_rpy[:] = 0
    ee_start_sphere = init_start_ee_sphere
    ee_goal_sphere = init_end_ee_sphere

    # collision setting
    collision_lower_limits = torch.tensor(cfg.goal_ee.collision_lower_limits, device=device, dtype=torch.float)
    collision_upper_limits = torch.tensor(cfg.goal_ee.collision_upper_limits, device=device, dtype=torch.float)
    underground_limit = cfg.goal_ee.underground_limit
    num_collision_check_samples = cfg.goal_ee.num_collision_check_samples
    collision_check_t = torch.linspace(0, 1, num_collision_check_samples, device=device)[None, None, :]

    for _ in tqdm(range(total_steps), desc="Simulating..."):
        cart, q, dq, omega, grav, vel, quat, ee_pos, ee_quat = get_obs(data, gripper_idx)
        q[[3,7]] = 0
        # ee goal related
        ee_pos = torch.as_tensor(ee_pos, device=device, dtype=torch.float32)
        q_tensor = torch.as_tensor(q, device=device, dtype=torch.float32)
        quat_tensor = torch.as_tensor(quat, device=device, dtype=torch.float32)
        vel_tensor = torch.as_tensor(vel, device=device, dtype=torch.float32)
        omega_tensor = torch.as_tensor(omega, device=device, dtype=torch.float32)
        cart_tensor = torch.as_tensor(cart, device=device, dtype=torch.float32)

        base_yaw_quat = quat_tensor #quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        ee_pos_local = quat_rotate_inverse(base_yaw_quat, ee_pos - torch.cat([q_tensor[:2], z_invariant_offset]))

        t = torch.clip(goal_timer / traj_timesteps, 0, 1)
        curr_ee_goal_sphere[:] = torch.lerp(ee_start_sphere, ee_goal_sphere, t[None])

        curr_ee_goal_cart = sphere2cart(curr_ee_goal_sphere)
        ee_goal_cart_yaw_global = quat_apply(base_yaw_quat, curr_ee_goal_cart)
        curr_ee_goal_cart_world = _get_ee_goal_spherical_center(cart_tensor,quat_tensor,ee_goal_center_offset,device) + ee_goal_cart_yaw_global
        curr_ee_goal_local = quat_rotate_inverse(base_yaw_quat, curr_ee_goal_cart_world - torch.cat([q_tensor[:2], z_invariant_offset]))
        
        default_yaw = torch.atan2(ee_goal_cart_yaw_global[1], ee_goal_cart_yaw_global[0])
        
        ee_goal_orn_quat = quat_from_euler_xyz(ee_goal_orn_delta_rpy[0] , ee_goal_orn_delta_rpy[1], ee_goal_orn_delta_rpy[2] + default_yaw)

        goal_timer += 1
        resample = goal_timer > traj_total_timesteps
        
        if resample:
            # ee_goal_delta_orn_r = torch_rand_float(self.goal_ee_ranges["delta_orn_r"][0], self.goal_ee_ranges["delta_orn_r"][1], (len(env_ids), 1), device=self.device)
            # ee_goal_delta_orn_p = torch_rand_float(self.goal_ee_ranges["delta_orn_p"][0], self.goal_ee_ranges["delta_orn_p"][1], (len(env_ids), 1), device=self.device)
            # ee_goal_delta_orn_y = torch_rand_float(self.goal_ee_ranges["delta_orn_y"][0], self.goal_ee_ranges["delta_orn_y"][1], (len(env_ids), 1), device=self.device)
            # self.ee_goal_orn_delta_rpy[env_ids, :] = torch.cat([ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1)
            ee_start_sphere = ee_goal_sphere.clone()
            for i in range(10):
                ee_goal_sphere[0] = torch_rand_float(goal_ee_ranges["pos_l"][0], goal_ee_ranges["pos_l"][1], (1, 1), device=device).squeeze(1)
                ee_goal_sphere[1] = torch_rand_float(goal_ee_ranges["pos_p"][0], goal_ee_ranges["pos_p"][1], (1, 1), device=device).squeeze(1)
                ee_goal_sphere[2] = torch_rand_float(goal_ee_ranges["pos_y"][0], goal_ee_ranges["pos_y"][1], (1, 1), device=device).squeeze(1)
                
                ee_target_all_sphere = torch.lerp(ee_start_sphere[..., None], ee_goal_sphere[...,  None], collision_check_t).squeeze(-1)
                ee_target_cart = sphere2cart(torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)).reshape(num_collision_check_samples, -1, 3)
                collision_mask = torch.any(torch.logical_and(torch.all(ee_target_cart < collision_upper_limits, dim=-1), torch.all(ee_target_cart > collision_lower_limits, dim=-1)), dim=0)
                underground_mask = torch.any(ee_target_cart[..., 2] < underground_limit, dim=0)
                collision = collision_mask | underground_mask
                if len(collision) == 0:
                    break

            ee_goal_cart[:] = sphere2cart(ee_goal_sphere[:])
            goal_timer[0] = 0.0

        # self.privileged_obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, # base_ang_vel机器人本体坐标系下的角速度（w_x,w_y,w_z）
        #                                      self.base_lin_vel * self.obs_scales.lin_vel,
        #                                      self.projected_gravity, # projected_gravity机器人坐标系下的重力分量（g_x, g_y, g_z）
        #                                      self.dof_err * self.obs_scales.dof_pos, # 各关节位置
        #                                      self.dof_vel * self.obs_scales.dof_vel, # 各关节速度,轮足相比点足扩展了2维，从6到8
        #                                      self.actions, # 动作(各个关节的角度，角速度，力矩，与选择的控制模式有关),轮足相比点足扩展了2维，从6到8
        #                                      self.commands[:, :3] * self.commands_scale, # commands机器人前三项命令，机器人坐标系x方向，y方向上的线速度，机器人z轴角速度

        #                                      self.curr_ee_goal_local,
        #                                      self.ee_pos_local,
        #                                      self.ee_orn,
        #                                      self.ee_goal_orn_quat,
        #                                      self.ee_pos_local - self.curr_ee_goal_local
        #                                      ), dim=-1)

        base_lin_vel = quat_rotate_inverse(quat_tensor, vel_tensor)
        base_ang_vel = quat_rotate_inverse(quat_tensor, omega_tensor)
        # print("base_ang_vel: ", base_ang_vel)
        # print("base_lin_vel: ", base_lin_vel)
        # print("grav: ", grav)
        # print("q: ", q - default_pos)
        # print("dq: ", dq)
        # print("actions: ", actions)
        # print("cmd: ", cmd)
        if count_low % cfg.sim_config.decimation == 0:
            obs = np.zeros([1,71],dtype=np.float32)
            obs[0,0:3]   = (base_ang_vel * 0.25).cpu().numpy() 
            obs[0,3:6]   = (base_lin_vel * 2.0).cpu().numpy()
            obs[0,6:9]   = grav
            obs[0,9:23]  = q - default_pos #reindex(q - default_pos)
            obs[0,23:37] = dq * 0.05 #reindex(dq) * 0.05
            obs[0,37:51] = actions
            obs[0,51:54] = cmd * 2

            obs[0,54:57] = curr_ee_goal_local.cpu().numpy() 
            obs[0,57:60] = ee_pos_local.cpu().numpy() 
            obs[0,60:64] = ee_quat 
            obs[0,64:68] = ee_goal_orn_quat.cpu().numpy() 
            obs[0,68:71] = (ee_pos_local - curr_ee_goal_local).cpu().numpy() 
            
            obs = np.clip(obs, -100, 100)

            print("=== Observation Debug Info ===")
            print("Base Angular Velocity (scaled, obs[0:3]):",        (base_ang_vel * 0.25).cpu().numpy())
            print("Base Linear Velocity (scaled, obs[3:6]):",         (base_lin_vel * 2.0).cpu().numpy())
            print("Gravity Vector (obs[6:9]):",                        grav)
            print("Joint Position Delta (obs[9:23]):",                 (q - default_pos))
            print("Joint Velocity (scaled, obs[23:37]):",              (dq * 0.05))
            print("Actions (obs[37:51]):",                             actions)
            print("Commanded Velocity (scaled, obs[51:54]):",          (cmd * 2))

            print("EE Goal (local frame, obs[54:57]):",                curr_ee_goal_local.cpu().numpy())
            print("EE Pos (local frame, obs[57:60]):",                 ee_pos_local.cpu().numpy())
            print("EE Quaternion (obs[60:64]):",                       ee_quat)
            print("EE Goal Quaternion (obs[64:68]):",                  ee_goal_orn_quat.cpu().numpy())
            print("EE Goal Error (pos diff, obs[68:71]):",             (ee_pos_local - curr_ee_goal_local).cpu().numpy())

            hist_obs.append(obs);    hist_obs.popleft()
            obs_history.append(obs); obs_history.popleft()
            pi   = np.concatenate([h[0] for h in hist_obs], axis=0)[None,:]
            phi  = np.stack(obs_history, axis=1) 
            t_pi  = torch.from_numpy(pi).float().to(device)
            t_phi = torch.from_numpy(phi).float().to(device)

            with torch.no_grad():
                t_act = policy(t_pi, t_phi)

            actions = t_act.cpu().numpy().reshape(-1)
            actions = np.clip(actions, -100, 100)
            actions[8:] = np.clip(actions[8:], dof_pos_limits_min[8:], dof_pos_limits_max[8:])

        tau = pd_control(default_pos, actions, q, cfg.robot_config.kps, dq, cfg.robot_config.kds)
        #tau = np.array([0, 0, 0, 0, -20, 0, 0, 0, 0, 0, 0, 0, 0, 0])  
        #tau[8:] = 0.0
        #print(tau[:8])
        data.ctrl = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)

        mujoco.mj_step(model, data)
        
        viewer.add_marker(
            pos=ee_pos.cpu().numpy(), size=[0.05, 0.05, 0.05], rgba=[1, 0, 0, 0.5], type=mujoco.mjtGeom.mjGEOM_SPHERE, label="end_effector")
        viewer.add_marker(
            pos=cart, size=[0.05, 0.05, 0.05], rgba=[1, 0, 0, 0.5], type=mujoco.mjtGeom.mjGEOM_SPHERE, label="robot_base")
        viewer.add_marker(
            pos=curr_ee_goal_cart_world.cpu().numpy(), size=[0.05, 0.05, 0.05], rgba=[1, 0, 0, 0.5], type=mujoco.mjtGeom.mjGEOM_SPHERE, label="ee_goal")
        upper_arm_pose = _get_ee_goal_spherical_center(cart_tensor, base_yaw_quat, ee_goal_center_offset, device)
        viewer.add_marker(
            pos=upper_arm_pose.cpu().numpy(), size=[0.05, 0.05, 0.05], rgba=[1, 0, 0, 0.5], type=mujoco.mjtGeom.mjGEOM_SPHERE, label="upper_arm_center")
        
        viewer.render()
        count_low += 1

    viewer.close()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class Sim2simCfg():
        class sim_config:
            frame_stack       = 1
            o_h_frame_stack   = 10
            mujoco_model_path = '/home/jks_n/桌面/DDT_VE/pointfoot-legged-gym/resources/robots/tita_airbot_mujuco/urdf/robot.xml'
            action_scale      = 0.25 #0.1 # 0.5
            sim_duration      = 60.0
            dt                = 0.005
            decimation        = 4 #10

        class robot_config:
            kps = np.array([30, 30, 30, 0, 30, 30, 30, 0, 
                   15, 15, 15, 15, 15, 15], dtype=np.double)
            kds = np.array([0.8, 0.8, 0.8, 0.5, 0.8, 0.8, 0.8, 0.5,
                   50.0, 50.0, 50.0, 50.0, 50.0, 0.1], dtype=np.double)
            
            tau_limit = np.array([50, 50, 50, 50, 50, 50, 50, 50,
                                  20, 20, 20, 3, 3, 3], dtype=np.double)
            end_effector_name = "link_G2_base"

        class goal_ee:
            z_invariant_offset = 0.57
            traj_time = [2, 3] 
            hold_time = [1, 2]
            collision_upper_limits = [0.3, 0.15, 0.05 - 0.165]
            collision_lower_limits = [-0.2, -0.15, -0.35 - 0.165]
            underground_limit = -0.57
            num_collision_check_samples = 10
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

    #model_path = '/home/jks_n/桌面/DDT_VE/pointfoot-legged-gym/logs/tita_airbot_flat/exported/policies/policy_1.pt'
    model_path = 'model_tita_airbot.pt'
    policy = torch.jit.load(model_path, map_location=device)
    print(policy.code)

    policy = policy.to(device).eval()
    run_mujoco(policy, Sim2simCfg(), device)
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
import torch


def quat_to_grav(q):
    q = np.asarray(q)
    v = np.array([0, 0, -1], dtype=np.float32)
    q_w = q[..., -1]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0)[..., np.newaxis]
    b = 2.0 * q_w[..., np.newaxis] * np.cross(q_vec, v)
    c = 2.0 * q_vec * np.sum(q_vec * v, axis=-1)[..., np.newaxis]
    return a - b + c

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos[7:]
    dq = data.qvel[6:]
    omega = data.qvel[3:6]
    quat = data.qpos[3:7]
    quat = [quat[1], quat[2], quat[3], quat[0]]
    proj_grav = quat_to_grav(quat)
    return q, dq, omega, proj_grav

def reindex(qpos):
    return qpos[[4, 5, 6, 7, 0, 1, 2, 3]]

def pd_control(default_dof_pos, target_q, q, kp, dq, kd):
    q_error = default_dof_pos - q
    torques = (target_q + q_error) * kp - dq * kd
    # torques[[3,7]] = 10 * (target_q[[3,7]] + q_error[[3,7]]) * kp[[3,7]] - 0.5 * dq[[3,7]] * kd[[3,7]]
    return torques

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
    default_pos = [0.0, 0.8, -1.5, 0.0, 0.0, 0.8, -1.5, 0.0]
    data.qpos[7:] = default_pos
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    hist_obs    = deque([np.zeros([1,33],dtype=np.float32) for _ in range(cfg.sim_config.frame_stack)])
    obs_history = deque([np.zeros([1,33],dtype=np.float32) for _ in range(cfg.sim_config.o_h_frame_stack)])
    count_low   = 0
    actions     = np.zeros(8, dtype=np.float32)
    cmd         = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    total_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
    for _ in tqdm(range(total_steps), desc="Simulating..."):
        q, dq, omega, grav = get_obs(data)
        q[[3,7]] = 0
        if count_low % cfg.sim_config.decimation == 0:
            obs = np.zeros([1,33],dtype=np.float32)
            obs[0,0:3]   = omega * 0.25
            obs[0,3:6]   = grav
            obs[0,6:9]   = cmd * 2
            obs[0,9:17]  = reindex(q - default_pos)
            obs[0,17:25] = reindex(dq * 0.05)
            obs[0,25:33] = actions
            obs = np.clip(obs, -100, 100)
            hist_obs.append(obs);    hist_obs.popleft()
            obs_history.append(obs); obs_history.popleft()
            pi   = np.concatenate([h[0] for h in hist_obs], axis=0)[None,:]
            phi  = np.stack(obs_history, axis=1)  # shape [1, frame, 33]
            t_pi  = torch.from_numpy(pi).float().to(device)
            t_phi = torch.from_numpy(phi).float().to(device)

            with torch.no_grad():
                t_act = policy(t_pi, t_phi)
            actions = t_act.cpu().numpy().reshape(-1)
            actions = np.clip(actions, -100, 100)
            target_q = actions * cfg.sim_config.action_scale
            target_q[[0,4]] *= 0.5
            target_q = reindex(target_q)

        tau = pd_control(default_pos, target_q, q, cfg.robot_config.kps, dq, cfg.robot_config.kds)
        #tau = [20,0,0,0,0,0,0,0]
        data.ctrl = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        print(data.ctrl)

        mujoco.mj_step(model, data)
        viewer.render()
        count_low += 1

    viewer.close()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class Sim2simCfg():
        class sim_config:
            frame_stack       = 1
            o_h_frame_stack   = 10
            mujoco_model_path = '/home/jks_n/桌面/DDT_VE/pointfoot-legged-gym/resources/robots/tita/xml/tita.xml'
            action_scale      = 0.5
            sim_duration      = 60.0
            dt                = 0.001
            decimation        = 10
        class robot_config:
            kps = 40. * np.ones(8, dtype=np.double)
            kds = 1. * np.ones(8, dtype=np.double)
            tau_limit = np.array([60, 60, 60, 15, 60, 60, 60, 15], dtype=np.double)

    model_path = 'model.pt'
    policy = torch.jit.load(model_path, map_location=device)
    print(policy.code)
#     def forward(self, obs: Tensor,
#     obs_hist: Tensor) -> Tensor: actor = self.actor
#       obs_encoder = self.obs_encoder
#       _1 = getattr(obs_encoder, "1")
#       mlp_encoder = self.mlp_encoder
#       _0 = torch.slice(obs_hist, 0, 0, 9223372036854775807)
#       _2 = torch.slice(_0, 1, 1, 9223372036854775807)
#      _3 = torch.slice(_2, 2, 0, 9223372036854775807)
#       obs_hist_full = torch.cat([_3, torch.unsqueeze(obs, 1)], 1)
#       b = ops.prim.NumToTensor(torch.size(obs_hist_full, 0))
#       _4 = int(b)
#       _5 = torch.slice(obs_hist_full, 0, 0, 9223372036854775807)
#       _6 = torch.slice(_5, 1, 0, 9223372036854775807)
#       _7 = torch.slice(_6, 2, 0, 9223372036854775807)
#       input = torch.view(_7, [_4, -1])
#       _8 = [(mlp_encoder).forward(_1, input, ), obs]
#       input0 = torch.cat(_8, -1)
#   return (actor).forward(_1, input0, )

    policy = policy.to(device).eval()
    run_mujoco(policy, Sim2simCfg(), device)
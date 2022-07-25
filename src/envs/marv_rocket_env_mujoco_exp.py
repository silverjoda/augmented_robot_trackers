import os
import time
import mujoco_py
import numpy as np
import quaternion
from gym import spaces


class MARVRocketBalMujocoEnv:
    metadata = {
        'render.modes': ['human'],
        "video.frames_per_second": int(np.round(1.0 / 0.01))
    }
    project_root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/marv.xml")

    def __init__(self, config):
        self.config = config

        self.model = mujoco_py.load_model_from_path(MARVRocketBalMujocoEnv.MODELPATH)
        self.sim = mujoco_py.MjSim(self.model)
        self.sim.nsubsteps = 5

        self.geom_name_dict = {self.model.geom_names[i] : i for i in range(len(self.model.geom_names))}
        self.joint_name_dict = {self.model.joint_names[i] : i for i in range(len(self.model.joint_names))}

        self.fw_bw_wheel_ratio = 0.1165 / 0.08

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.qpos_indeces = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 16])
        self.qvel_indeces = np.array([0, 1, 2, 3, 4, 5, 6, 8, 12, 15])

        # Environment variables
        self.state_list = None
        self.state_history_len = 2
        self.act_list = None
        self.act_history_len = 2
        self.act_dim = 2
        self.obs_dim = 2 + len(self.qpos_indeces) + len(self.qvel_indeces)
        self.obs_dim_available = 2 + 7 * self.state_history_len + self.act_dim * self.act_history_len

        self.observation_space = spaces.Box(low=-4, high=4, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(self.act_dim,), dtype=np.float32)
        self.reward_range = (-3, 3)

        self.episode_ctr = 0
        self.param_target = -1

    def get_available_obs(self):
        time_feature = (float(self.step_ctr) / self.config["max_steps"]) * 2 - 1

        # Assemble complete obs vec
        obs = np.concatenate([self.state_list, self.act_list, [time_feature, self.param_target]])

        return obs

    def tick_state_queue(self):
        qpos = np.array(self.sim.get_state().qpos.tolist())[self.qpos_indeces]

        body_mat = self.sim.data.body_xmat[self.geom_name_dict["torso"]]
        _, pitch = self.calc_verticality(body_mat)
        position = [qpos[0], qpos[2]]
        flipper_positions = list(qpos[7:11])
        state = position + [pitch] + flipper_positions

        if self.state_list is None:
            self.state_list = state * self.state_history_len  # Repeat h times
        else:
            self.state_list.extend(state)
            del self.state_list[0:len(state)]

    def tick_act_queue(self, act):
        if self.act_list is None:
            self.act_list = [0] * self.act_dim * (self.act_history_len - 1) + list(act)  # Repeat h times
        else:
            self.act_list.extend(act)
            del self.act_list[0:len(act)]

    def get_privileged_obs(self):
        qpos = np.array(self.sim.get_state().qpos.tolist())[self.qpos_indeces]
        qvel = np.array(self.sim.get_state().qvel.tolist())[self.qvel_indeces]

        time_feature = (float(self.step_ctr) / self.config["max_steps"]) * 2 - 1

        # Assemble complete obs vec
        obs = np.concatenate([qpos, qvel, [time_feature, self.param_target]])

        return obs

    def set_state(self, qpos, qvel=None):
        # continue here,  get self.q_dim first
        qvel = np.zeros(self.qvel_dim) if qvel is None else qvel
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()
        self.sim.step()

    def render(self, mode=None):
        if not hasattr(self, "viewer"):
            self.viewer = mujoco_py.MjViewer(self.sim)

        self.viewer.render()

    def calc_verticality(self, rot_mat):
        rot_mat = np.reshape(rot_mat, (3,3))
        verticality = rot_mat[2, 0]
        pitch_dir = np.sign(rot_mat[0, 0])
        pitch_normed = verticality
        if pitch_dir < 0:
            pitch_normed = + (2 - verticality)
        pitch = -pitch_normed * np.pi * 0.5
        return verticality, pitch

    def calc_target_height_and_pitch(self):
        p_sc = (self.param_target + 1) * 0.5
        #height = 0.14 + (0.26 * p_sc) + p_sc * (np.sin(p_sc) * 0.26)
        height = 0.12 + p_sc * 0.7
        pitch = -p_sc * (np.pi * 0.4)
        return height, pitch

    def step_sim_complete(self, act):
        flipper_act_front = 0# act[0]
        flipper_act_rear = act[0] + 1.5
        main_wheels_left_act = act[1]
        main_wheels_right_act = act[1]
        tip_wheels_left_act = main_wheels_left_act * self.fw_bw_wheel_ratio
        tip_wheels_right_act = main_wheels_right_act * self.fw_bw_wheel_ratio
        act_complete = flipper_act_front,\
                       flipper_act_front,\
                       flipper_act_rear,\
                       flipper_act_rear,\
                       main_wheels_left_act, \
                       main_wheels_right_act, \
                       main_wheels_left_act, \
                       main_wheels_right_act, \
                       tip_wheels_left_act, \
                       tip_wheels_right_act, \
                       tip_wheels_left_act, \
                       tip_wheels_right_act
        self.sim.data.ctrl[:] = act_complete
        self.sim.forward()
        self.sim.step()

    def step(self, act):
        self.step_sim_complete(act)

        # Tick the observation queue
        self.tick_act_queue(act)
        self.tick_state_queue()

        priveleged_obs = self.get_privileged_obs()
        available_obs = self.get_available_obs()

        verticality, pitch = self.calc_verticality(self.sim.data.geom_xmat[self.geom_name_dict["torso"]])

        tail_height = self.sim.data.geom_xpos[self.geom_name_dict["tail"]][2]
        p_sc = (self.param_target + 1) * 0.5
        target_tail_height = 0.174 + p_sc * 0.38
        rew = 1 / (1 + 100 * np.square(tail_height - target_tail_height)) + 1 / (1 + 10 * np.square(priveleged_obs[0])) - np.maximum(pitch, 0) * 1

        toppled = pitch > 2
        out_of_bnds = abs(priveleged_obs[0]) > 1.5

        done = self.step_ctr > self.config["max_steps"] or toppled or out_of_bnds

        self.step_ctr += 1

        return priveleged_obs, rew, done, {"privileged_obs" : priveleged_obs, "available_obs" : available_obs}

    def reset(self):
        self.step_ctr = 0
        self.episode_ctr += 1
        self.param_target = np.random.rand() * 2 - 1
        self.state_list = None
        self.act_list = None

        # Set environment state
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[2] = 0.2
        #init_q[3:7] = [0.737, 0, -0.676, 0]
        init_q[7] = 0
        init_q[10] = 0

        init_qvel = np.zeros(self.qvel_dim, dtype=np.float32)
        self.set_state(init_q, init_qvel)

        obs, _, _, _ = self.step(np.zeros(self.act_dim))

        return obs

    def demo(self):
        self.reset()
        acts = [[1,3,1,1], [-1,-3,-0,0], [0,0,-1,-1]]
        while True:
            for a in acts:
                print(a)
                for i in range(self.config["max_steps"]):
                    self.step(np.random.randn(4))
                    self.render()
            self.reset()

    def kill(self):
        pass

    def close(self):
        pass

if __name__=="__main__":
    import yaml
    with open("configs/marv_rocket_mujoco_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env = MARVRocketBalMujocoEnv(config)
    env.demo()


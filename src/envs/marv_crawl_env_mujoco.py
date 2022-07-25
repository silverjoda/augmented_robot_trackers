import os
import time
import mujoco_py
import numpy as np
import quaternion
from gym import spaces

class MARVCrawlMujocoEnv:
    metadata = {
        'render.modes': ['human'],
        "video.frames_per_second": int(np.round(1.0 / 0.01))
    }
    project_root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/marv_crawl.xml")

    def __init__(self, config):
        self.config = config

        self.model = mujoco_py.load_model_from_path(MARVCrawlMujocoEnv.MODELPATH)
        self.sim = mujoco_py.MjSim(self.model)
        self.sim.nsubsteps = 5

        self.geom_name_dict = {self.model.geom_names[i] : i for i in range(len(self.model.geom_names))}
        self.joint_name_dict = {self.model.joint_names[i] : i for i in range(len(self.model.joint_names))}

        self.fw_bw_wheel_ratio = 0.1165 / 0.08

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.qpos_indeces = np.array([1, 2, 3, 4, 5, 6, 7, 10, 13, 16])
        self.qvel_indeces = np.array([0, 1, 2, 3, 4, 5, 6, 8, 12, 15])

        # Environment variables
        self.act_dim = 4
        self.obs_dim = 2 + len(self.qpos_indeces) + len(self.qvel_indeces)

        self.observation_space = spaces.Box(low=-7, high=7, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(self.act_dim,), dtype=np.float32)
        self.reward_range = (-3, 3)

        self.episode_ctr = 0
        self.param_target = -1

    def get_obs(self):
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
        #rot_mat = quaternion.as_rotation_matrix(quaternion.quaternion(*quat))
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
        self.sim.data.ctrl[:] = act
        self.sim.forward()
        self.sim.step()

    def step(self, act):
        # TODO: MAYBE CHANGE TO INCREMENTAL ACTIONS

        self.step_sim_complete(act)

        obs = self.get_obs()
        #verticality, pitch = self.calc_verticality(self.sim.data.geom_xmat[self.geom_name_dict["torso"]])

        vel_x, vel_y, vel_z = self.sim.data.body_xvelp[self.geom_name_dict["torso"]]

        rew = -np.square(vel_x - 0.3) - np.square(obs[0]) * 0.5

        done = self.step_ctr > self.config["max_steps"]

        self.step_ctr += 1

        return obs, rew, done, {}

    def reset(self):
        self.step_ctr = 0
        self.episode_ctr += 1
        self.param_target = np.random.rand() * 2 - 1

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
        acts = [[-1,-1,0,0], [1,1,-1,-1], [0,0,0,0]]
        while True:
            for a in acts:
                print(a)
                for i in range(self.config["max_steps"]):
                    self.step(a)
                    self.render()
            self.reset()

    def kill(self):
        pass

    def close(self):
        pass

if __name__=="__main__":
    import yaml
    with open("configs/marv_crawl_mujoco_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env = MARVCrawlMujocoEnv(config)
    env.demo()


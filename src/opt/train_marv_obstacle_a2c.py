import argparse
import os
import random
import time
from pprint import pprint

import numpy as np
import torch as T
import yaml
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

import src.utilities as utilities
from src.envs import marv_obstacle_env_gym_wrapper

class SBMarvObstacleTrainer:
    def __init__(self):
        self.args = self.parse_args()
        self.config = self.read_configs()
        self.env = marv_obstacle_env_gym_wrapper.MARVObstacleEnvGymWrapper(self.config)
        self.model, self.checkpoint_callback, self.stats_path = self.setup_train()

        if self.config["train"]:
            t1 = time.time()
            self.model.learn(total_timesteps=self.config["iters"], callback=self.checkpoint_callback, log_interval=1)
            t2 = time.time()

            print("Training time: {}".format(t2 - t1))
            pprint(self.config)

            self.model.save("agents/{}_SB_policy".format(self.config["session_ID"]))
            self.env.close()

        if self.config["test"]:
            self.model = A2C.load("agents/{}".format(self.args["test_agent_path"]))
            N_test = 100
            total_rew = self.test_agent(deterministic=True, N=N_test)
            print(f"Total test rew: {total_rew / N_test}")

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Pass in parameters. ')
        parser.add_argument('--train', action='store_true', required=False,
                            help='Flag indicating whether the training process is to be run. ')
        parser.add_argument('--test', action='store_true', required=False,
                            help='Flag indicating whether the testing process is to be run. ')
        parser.add_argument('--animate', action='store_true', required=False,
                            help='Flag indicating whether the environment will be rendered. ')
        parser.add_argument('--test_agent_path', type=str, default=".", required=False,
                            help='Path of test agent. ')
        parser.add_argument('--iters', type=int, required=False, default=800000, help='Number of training steps. ')

        args = parser.parse_args()
        return args.__dict__

    def read_configs(self):
        with open(os.path.join(os.path.dirname(__file__), "configs/train_marv_rocket_config.yaml"), 'r') as f:
            algo_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(os.path.dirname(__file__), "configs/a2c_default_config.yaml"), 'r') as f:
            env_config = yaml.load(f, Loader=yaml.FullLoader)

        config = utilities.merge_dicts([algo_config, env_config])
        return config

    def setup_train(self, setup_dirs=True):
        T.set_num_threads(1)
        if setup_dirs:
            for s in ["agents", "agents_cp", "tb"]:
                if not os.path.exists(s):
                    os.makedirs(s)

        # Random ID of this session
        if self.config["test_agent_ID"] is None:
            self.config["session_ID"] = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))
        else:
            self.config["session_ID"] = self.config["default_session_ID"]

        stats_path = "agents/{}_vecnorm.pkl".format(self.config["session_ID"])
        checkpoint_callback = CheckpointCallback(save_freq=100000,
                                                 save_path='agents_cp/',
                                                 name_prefix=self.config["session_ID"], verbose=1)

        tb_log = None
        if self.config["tensorboard_log"]:
            tb_log = "tb/{}/".format(self.config["session_ID"])


        model = A2C(policy=self.config["policy_name"],
                    env=self.env,
                    gae_lambda=self.config["gae_lambda"],
                    use_rms_prop=self.config["use_rms_prop"],
                    normalize_advantage=self.config["normalize_advantage"],
                    gamma=self.config["gamma"],
                    n_steps=self.config["n_steps"],
                    vf_coef=self.config["vf_coef"],
                    ent_coef=self.config["ent_coef"],
                    max_grad_norm=self.config["max_grad_norm"],
                    learning_rate=eval(self.config["learning_rate"]),
                    verbose=self.config["verbose"],
                    use_sde=self.config["use_sde"],
                    tensorboard_log=tb_log,
                    device="cpu",
                    policy_kwargs=dict(net_arch=[64, 64]))

        return model, checkpoint_callback, stats_path

    def test_agent(self, deterministic=True, N=100, print_rew=True, render=True):
        total_rew = 0

        for _ in range(N):
            obs = self.env.reset()
            episode_rew = 0
            while True:
                action, _states = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.env.step(action)
                episode_rew += reward
                total_rew += reward

                if done:
                    if print_rew:
                        print(episode_rew)
                    break
        return total_rew

if __name__ == "__main__":
    trainer = SBMarvObstacleTrainer()

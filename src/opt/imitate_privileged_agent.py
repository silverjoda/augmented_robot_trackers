import argparse
import os
import random
import time
from pprint import pprint

import torch as T
import yaml
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv, VecMonitor

from src.envs import marv_rocket_env_mujoco_exp
from src.utilities import utilities

import tkinter as tk
from tkinter import ttk
from threading import Thread
import torch as T

from src.policies import policies

class PrivilegedAgentImitator:
    def __init__(self):
        self.config = self.read_configs()
        self.env_fun = marv_rocket_env_mujoco_exp.MARVRocketBalMujocoEnv

        # thread = Thread(target=self.make_control_widget)
        # thread.start()

        # Load trained agent and env
        self.model = A2C.load("agents/{}_SB_policy".format(self.config["default_session_ID"]))
        vec_env = DummyVecEnv(env_fns=[lambda: self.env_fun(self.config)] * 1)
        monitor_env = VecMonitor(vec_env)
        normed_env = VecNormalize(venv=monitor_env, training=False, norm_obs=True, norm_reward=True, clip_reward=10.0)
        self.env = VecNormalize.load("agents/{}_vecnorm.pkl".format(self.config["default_session_ID"]), normed_env)

        # Create agent
        tmp_env = self.env_fun(self.config)
        student = policies.MLP(tmp_env.obs_dim_available, tmp_env.act_dim, hid_dim=256)
        del tmp_env

        # Imitate agent
        self.imitate_agent(self.env, self.model, student)

        N_test = 10
        total_rew = self.test_agent(self.env, student, deterministic=True, N=N_test)
        print(f"Total test rew: {total_rew / N_test}")

    def make_control_widget(self):
        # root window
        root = tk.Tk()
        root.geometry('200x80')
        root.resizable(False, False)
        root.title('Slider Demo')

        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=3)

        # slider current value
        current_value = tk.DoubleVar()

        def get_current_value():
            return '{: .2f}'.format(current_value.get())

        def slider_changed(event):
            value_label.configure(text=get_current_value())
            self.env.unwrapped.envs[0].param_target = float(get_current_value())

        # label for the slider
        slider_label = ttk.Label(root, text='Slider:')
        slider_label.grid(column=0, row=0, sticky='w')

        #  slider
        slider = ttk.Scale(root, from_=-1, to=1, orient='horizontal', command=slider_changed, variable=current_value)

        slider.grid(column=1, row=0, sticky='we')

        # current value label
        current_value_label = ttk.Label(root, text='Current Value:')

        current_value_label.grid(row=1, columnspan=2, sticky='n', ipadx=10, ipady=10)

        # value label
        value_label = ttk.Label(root, text=get_current_value())
        value_label.grid(row=2, columnspan=2, sticky='n')

        root.mainloop()

    def read_configs(self):
        with open(os.path.join(os.path.dirname(__file__), "configs/train_marv_rocket_config.yaml"), 'r') as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(os.path.dirname(__file__), "configs/a2c_default_config.yaml"), 'r') as f:
            algo_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/marv_rocket_mujoco_config.yaml"), 'r') as f:
            env_config = yaml.load(f, Loader=yaml.FullLoader)

        config = utilities.merge_dicts([algo_config, train_config, env_config])
        return config

    def imitate_agent(self, env, teacher, student):
        optim = T.optim.Adam(student.parameters(), lr=0.001)
        mse_loss = T.nn.MSELoss()

        n_rollouts = 3
        n_iters = 1000

        # Train
        for i in range(n_iters):
            # Gather N batches of rollouts from imit_agent
            obs_available_T, act_labels, avg_rew = self.get_rollouts(env, teacher, student, n_rollouts, render=(i % 100 == 0) and False)

            # Calc loss and backprop
            acts_student = student(obs_available_T)
            loss = mse_loss(acts_student, act_labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 10 == 0:
                print(f"Iteration: {i}, loss: {loss.data}, avg_rew: {avg_rew}")

    def get_rollouts(self, env, teacher, student, n, render=False):
        obs_available_list = []
        acts_teacher_list = []
        cum_rew = 0
        for i in range(n):
            obs_priv = env.reset()
            obs_av = env.unwrapped.envs[0].get_available_obs()

            while True:
                teacher_act, _ = teacher.predict(obs_priv, deterministic=True)
                acts_teacher_list.append(teacher_act[0])
                obs_available_list.append(obs_av)
                obs_T = T.Tensor(obs_av).unsqueeze(0)
                act = student(obs_T)
                if i % 2 == 0:
                    obs_priv, rew, done, obs_dict = env.step(act.detach().numpy())
                else:
                    obs_priv, rew, done, obs_dict = env.step(teacher_act)
                obs_av = obs_dict[0]["available_obs"]
                cum_rew += env.get_original_reward()
                if done: break

                if render and i == 0:
                    env.render()

        avg_rew = cum_rew / float(n)

        obs_available_T = T.Tensor(obs_available_list)
        acts_teacher_T = T.Tensor(acts_teacher_list)
        return obs_available_T, acts_teacher_T, avg_rew

    def test_agent(self, env, student, deterministic=True, N=100, print_rew=True, render=True):
        total_rew = 0

        for i in range(N):
            _ = env.reset()
            obs_av = env.unwrapped.envs[0].get_available_obs()

            while True:
                obs_T = T.Tensor(obs_av).unsqueeze(0)
                act = student(obs_T)
                obs_priv, rew, done, obs_dict = env.step(act.detach().numpy())

                obs_av = obs_dict[0]["available_obs"]
                total_rew += env.get_original_reward()
                if done: break

                if render:
                    env.render()
        return total_rew / float(N)

if __name__ == "__main__":
    trainer = PrivilegedAgentImitator()

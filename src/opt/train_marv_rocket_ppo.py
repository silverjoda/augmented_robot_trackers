import argparse
import os
import random
import time
import tkinter as tk
from pprint import pprint
from tkinter import ttk

import torch as T
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv, VecMonitor

from src.envs import marv_rocket_env_mujoco
from src.utilities import utilities


class SBMarvRocketTrainer:
    def __init__(self):
        self.config = self.read_configs()
        self.env_fun = marv_rocket_env_mujoco.MARVRocketBalMujocoEnv
        self.env = self.env_fun(self.config)
        self.N_cores = 6

        self.model, self.checkpoint_callback, self.stats_path = self.setup_train()

        #thread = Thread(target=self.make_control_widget)
        #thread.start()

        if self.config["train"]:
            t1 = time.time()
            self.model.learn(total_timesteps=self.config["iters"], callback=self.checkpoint_callback, log_interval=1)
            t2 = time.time()

            print("Training time: {}".format(t2 - t1))
            pprint(self.config)

            self.env.save(self.stats_path)

            self.model.save("agents/{}_SB_policy".format(self.config["session_ID"]))
            self.env.close()

        if not self.config["train"]:
            self.model = PPO.load("agents/{}_SB_policy".format(self.config["session_ID"]))

            vec_env = DummyVecEnv(env_fns=[lambda: self.env_fun(self.config)] * 1)
            monitor_env = VecMonitor(vec_env)
            normed_env = VecNormalize(venv=monitor_env, training=False, norm_obs=True, norm_reward=True, clip_reward=10.0)
            self.env = VecNormalize.load(self.stats_path, normed_env)

            N_test = 100
            total_rew = self.test_agent(deterministic=True, N=N_test)
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
        with open(os.path.join(os.path.dirname(__file__), "configs/ppo_default_config.yaml"), 'r') as f:
            algo_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/marv_rocket_mujoco_config.yaml"), 'r') as f:
            env_config = yaml.load(f, Loader=yaml.FullLoader)

        config = utilities.merge_dicts([algo_config, train_config, env_config])
        return config

    def setup_train(self, setup_dirs=True):
        T.set_num_threads(1)
        if setup_dirs:
            for s in ["agents", "agents_cp", "tb", "logs"]:
                if not os.path.exists(s):
                    os.makedirs(s)

        # Random ID of this session
        if self.config["default_session_ID"] is None:
            self.config["session_ID"] = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))
        else:
            self.config["session_ID"] = self.config["default_session_ID"]

        #vec_env = DummyVecEnv(env_fns=[lambda : marv_rocket_bal_env_mujoco.MARVRocketBalMujocoEnv(self.config)] * self.N_cores)
        vec_env = SubprocVecEnv(env_fns=[lambda : self.env_fun(self.config) for _ in range(self.N_cores)] , start_method="fork")
        monitor_env = VecMonitor(vec_env)
        normed_env = VecNormalize(venv=monitor_env, training=True, norm_obs=True, norm_reward=True)

        stats_path = "agents/{}_vecnorm.pkl".format(self.config["session_ID"])
        checkpoint_callback = CheckpointCallback(save_freq=300000,
                                                 save_path='agents_cp/',
                                                 name_prefix=self.config["session_ID"], verbose=1)

        # eval_callback = EvalCallback(normed_env, best_model_save_path='logs/',
        #                              log_path='logs/', eval_freq=10000, n_eval_episodes=1,
        #                              deterministic=False, render=True)

        self.env = normed_env

        model = PPO(policy=self.config["policy_name"],
                    env=self.env,
                    gamma=self.config["gamma"],
                    n_steps=self.config["n_steps"],
                    vf_coef=self.config["vf_coef"],
                    ent_coef=self.config["ent_coef"],
                    max_grad_norm=self.config["max_grad_norm"],
                    learning_rate= self.config["learning_rate"], # lambda x : x * self.config["learning_rate"] # doesn't work due to picking err when saving model,
                    verbose=self.config["verbose"],
                    device="cpu",
                    policy_kwargs=dict(net_arch=[self.config["policy_hid_dim"], self.config["policy_hid_dim"]]))

        callback_list = CallbackList([checkpoint_callback])

        return model, callback_list, stats_path

    def test_agent(self, deterministic=True, N=100, print_rew=True, render=True):
        total_rew = 0

        for _ in range(N):
            obs = self.env.reset()
            episode_rew = 0
            while True:
                action, _states = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.env.step(action)
                episode_rew += self.env.get_original_reward()
                total_rew += self.env.get_original_reward()
                if render:
                    self.env.render()
                # if done:
                #     if print_rew:
                #         print(episode_rew)
                #     break
        return total_rew

if __name__ == "__main__":
    trainer = SBMarvRocketTrainer()

import argparse
import logging
import os
import random
import socket
import sys
import time

import cma
import numpy as np

import torch
import torch as T
import yaml
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
T.set_num_threads(1)

from src.envs.marv_stairs_env import MARVStairsEnv
from src.path_follower.smart_tracker import FlipperTracker
import src.utilities as utilities

def objective(w, env, policy):
    reward = 0

    for _ in range(2):
        done = False
        env.reset()

        policy.set_parameters_from_vector(w)
        policy.reset()

        while not done:
            policy.step()
            _, rew, done, _ = env.step()

            # Get reward from path_follower
            if policy.current_state != "NEUTRAL":
                reward -= 0.05

            reward += rew

        print("REW: {}".format(reward))

    return -reward

def train(env, policy, config):
    w = policy.get_vector_from_current_parameters()
    es = cma.CMAEvolutionStrategy(w, config["cma_std"])
    f = lambda x: objective(x, env, policy)

    print('N_params: {}'.format(len(w)))

    it = 0
    try:
        while not es.stop():
            it += 1
            if it > config["iters"]:
                break
            X = es.ask()
            es.tell(X, [f(x) for x in X])
            es.disp()

    except KeyboardInterrupt:
        print("User interrupted process.")

    policy.set_parameters_from_vector(es.result.xbest)
    policy.save("agents/{}".format(config["session_ID"]))
    print("Saved agent, agents/{}".format(config["session_ID"]))

    return es.result.fbest

def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def test_agent(env, policy, N=30):
    total_rew = 0
    for _ in range(N):
        obs = env.reset()
        cum_rew = 0
        while True:
            action = policy.calculate_action(obs)
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            total_rew += reward
            if done:
                print(cum_rew)
                break
    return total_rew / N

if __name__=="__main__":
    import yaml
    with open("configs/train_tracker_config.yaml") as f:
        algo_config = yaml.load(f, Loader=yaml.FullLoader)
    with open("../envs/configs/marv_stairs_config.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    with open("../control/configs/smart_tracker_config.yaml") as f:
        tracker_config = yaml.load(f, Loader=yaml.FullLoader)
    with open("../ros_robot_interface/configs/ros_robot_interface_config.yaml") as f:
        ros_if_config = yaml.load(f, Loader=yaml.FullLoader)

    config = utilities.merge_dicts([algo_config,
                                    env_config,
                                    tracker_config,
                                    ros_if_config])

    print(config)

    if not os.path.exists("agents"):
        os.makedirs("agents")

    # Random ID of this session
    config["session_ID"] = "FLI"

    env = MARVStairsEnv(config, None)
    policy = FlipperTracker(config)
    env.ros_interface = policy.ros_robot_interface

    if config["train"]:
        t1 = time.time()
        train(env, policy, config)
        t2 = time.time()

        print("Training time: {}".format(t2 - t1))
        print(config)

    if config["test"]:
        print("Loading policy")
        policy.load("agents/flippers_ascent.pickle")

        print("Testings")
        avg_rew = test_agent(env, policy)

        print("Avg test rew: {}".format(avg_rew))



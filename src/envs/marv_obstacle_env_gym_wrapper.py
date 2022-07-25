import os

import rospy
from augmented_robot_trackers.srv import StepEnv, ResetEnv
from gym import spaces
from std_msgs.msg import Float64MultiArray
import numpy as np

class MARVObstacleEnvGymWrapper:
    metadata = {
        'render.modes': ['human'],
    }
    project_root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    def __init__(self, config):
        self.config = config

        # Environment variables
        self.act_dim = 2
        self.obs_dim = 14

        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim,), dtype=np.float32)
        self.reward_range = (-3,3)

        rospy.init_node("marv_rocket_env_gym_wrapper")

        rospy.wait_for_service('step_service')
        rospy.wait_for_service('reset_service')
        self.step_env_service = rospy.ServiceProxy('step_service', StepEnv)
        self.reset_env_service = rospy.ServiceProxy('reset_service', ResetEnv)

    def step(self, act):
        resp = self.step_env_service(Float64MultiArray(data=act))
        obs = resp.obs.data
        rew = resp.rew.data
        done = resp.done.data
        return obs, rew, done, {}

    def reset(self):
        return self.reset_env_service().obs.data

    def close(self):
        pass

if __name__=="__main__":
    import yaml
    with open("configs/marv_obstacle_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env = MARVObstacleEnvGymWrapper(config)
    rospy.spin()


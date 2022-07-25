import os
import time

import numpy as np

from src.ros_robot_interface.ros_robot_interface import ROSRobotInterface
from src.utilities import merge_two_dicts

class MARVStairsEnv:
    def __init__(self, config, ros_interface):
        self.config = config

        self.initial_setup()
        self.reset_episode_variables()

        self.ros_interface = ros_interface

        # Environment variables
        self.act_dim = 6
        self.obs_dim = None

        self.teleport_idx = 0

    def initial_setup(self):
        print("Starting initial setup")

        os.system('killall -9 atop bwm-ng parameter_bridge subt_ros_relay rosout rosmaster ruby image_bridge optical_frame_publisher pose_tf_broadcaster roslaunch set_rate_relay > /dev/null 2>&1')

        # Initialize sim
        os.system('SUBT_ENDLESS_MISSION=1 SUBT_ROBOT_TEAM=marv_ta ~/cras_subt/src/subt_virtual/scripts/run_sim worldName:=simple_urban_01 enableGroundTruth:=false headless:={} > /dev/null 2>&1 &'.format(self.config["headless_simulation"]))

        # Initialize bridge
        os.system('SUBT_ROBOT_TEAM=marv_ta SUBT_USE_SINGULARITY=0 ~/cras_subt/src/subt_virtual/scripts/run_bridge_all > /dev/null 2>&1 &')
        time.sleep(20)

        # Set topic rates and teleport robot to initial location
        self.set_initial_services()

        time.sleep(10)

        # Initialize mapping
        os.system("roslaunch aloam_velodyne sys_marv_ta.launch worldName:=simple_urban_01 > /dev/null 2>&1 &")

        # Make zpr frame and dummy path
        os.system('rosrun augmented_robot_trackers bl_zpr_frame_publisher.py &')
        os.system('rosrun augmented_robot_trackers dummy_path_publisher.py &')

        time.sleep(5)

        # Initialize rds
        os.system("roslaunch rds_map_nav ta_husky_default.launch &")

    def set_initial_services(self):
        # Sleep to give it time, and then place robot on start location
        self.teleport_to(self.config["init_position"], self.config["init_rotation"])

    def set_seed(self):
        pass

    def teleport_to(self, position, orientation):
        os.system(
            "IGN_TRANSPORT_TOPIC_STATISTICS=1 ign service -s /world/simple_urban_01/set_pose --reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --timeout 2000 --req " +
            "'" + "name: \"X1\", position: " + "{{x: {}, y: {}, z: {}}}, ".format(
                *position) + "orientation: {{x: {}, y: {}, z: {}, w: {}}}".format(
                *orientation) + "' &")

    def assemble_observation(self):
        pose_dict = self.ros_interface.get_robot_pose_dict()
        prop_data_dict = self.ros_interface.get_proprioceptive_data_dict_marv()
        ext_data_dict = self.ros_interface.get_exteroceptive_data_dict_marv()

        obs = np.concatenate((pose_dict["position"],
                              pose_dict["quat"],
                              pose_dict["position"],
                              prop_data_dict["flippers_state"],
                              prop_data_dict["imu"],
                              ext_data_dict["frontal_low_feat"],
                              ext_data_dict["frontal_mid_feat"],
                              ext_data_dict["fl_flipper_feat"],
                              ext_data_dict["fr_flipper_feat"],
                              ext_data_dict["rl_flipper_feat"],
                              ext_data_dict["rr_flipper_feat"]))

        obs_dict = merge_two_dicts(merge_two_dicts(pose_dict, prop_data_dict), ext_data_dict)
        return obs, obs_dict

    def step(self):
        pose_dict = self.ros_interface.get_robot_pose_dict()
        position = pose_dict["position"]
        roll = pose_dict["euler"][1]

        # Calculate reward
        rew = (position.x - self.prev_position_x) / self.config["sim_step_period"] - np.abs(roll)
        self.prev_position_x = position.x

        # Calculate done condition
        done = False
        if self.step_ctr > self.config["max_episode_steps"] or position.x > self.config["x_goal"]:
            done = True

        self.step_ctr += 1

        return pose_dict, rew, done, {}

    def reset(self):
        # Publish zero values
        self.ros_interface.publish_track_vel(linear=0, angular=0)
        self.ros_interface.publish_flipper_pos({"front_left" : -2, "front_right" : -2,
                                               "rear_left" : 2, "rear_right" : 2})

        time.sleep(1)

        # Teleport to initial position
        position_list = ["init_position", "up_stairs_position"]
        rotation_list = ["init_rotation", "up_stairs_rotation"]
        teleport_position = position_list[self.teleport_idx]
        teleport_rotation = rotation_list[self.teleport_idx]
        self.teleport_idx = (self.teleport_idx + 1 ) % len(position_list)

        self.teleport_to(self.config[teleport_position], self.config[teleport_rotation])

        time.sleep(1)

        # Kill the mapping
        os.system('rosnode kill alaserMapping alaserOdometry ascanRegistration')

        time.sleep(1)

        # Relaunch mapping
        os.system("roslaunch aloam_velodyne aloam_ta.launch rviz:=false > /dev/null 2>&1 &")

        time.sleep(2)

        self.reset_episode_variables()

    def reset_episode_variables(self):
        self.prev_position_x = 0
        self.step_ctr = 0

    def demo(self):
        while True:
            t1 = time.time()
            while True:
                self.ros_interface.publish_track_vel(linear=1, angular=0)
                if time.time() - t1 > 8: break
                time.sleep(0.1)

            self.reset()

    def kill(self):
        pass

if __name__=="__main__":
    import yaml
    with open("configs/marv_stairs_config.yaml") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)

    with open("../control/configs/smart_tracker_config.yaml") as f:
        tracker_config = yaml.load(f, Loader=yaml.FullLoader)

    with open("../ros_robot_interface/configs/ros_robot_interface_config.yaml") as f:
        ros_if_config = yaml.load(f, Loader=yaml.FullLoader)

    config = merge_two_dicts(merge_two_dicts(tracker_config, env_config), ros_if_config)

    env = MARVStairsEnv(config, None)
    ros_interface = ROSRobotInterface(config, "env_demo_IF")
    env.ros_interface = ros_interface

    env.demo()

    while True:
        pass

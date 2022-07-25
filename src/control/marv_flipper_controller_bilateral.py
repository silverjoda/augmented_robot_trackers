#!/usr/bin/python

import os
import sys
import time

import rospy
import tf2_ros
from augmented_robot_trackers.msg import MarvPCFeats
from marv_msgs.msg import Float64MultiArray as MarvFloat64MultiArray
from std_msgs.msg import Float64, Float64MultiArray
from std_msgs.msg import String

from src.policies.policies import *
from src.utilities import utilities, ros_utilities


class MarvFlipperControllerBilateral:
    def __init__(self, config):
        self.config = config

        self.base_link_dict = {"base_link" : self.config["robot_prefix"] + "base_link_zrp",
                               "base_link_rev" : self.config["robot_prefix"] + "base_link_zrp_rev"}

        self.state_list = ["NEUTRAL",
                            "ASCENDING_FRONT",
                            "UP_STAIRS",
                            "ASCENDING_REAR",
                            "DESCENDING_FRONT",
                            "DOWN_STAIRS",
                            "DESCENDING_REAR"]

        self.enable_flippers = self.config["enable_flippers"]

        self.state_to_short_name_dict = {"NEUTRAL": "N",
                                         "ASCENDING_FRONT": "AF",
                                         "UP_STAIRS": "US",
                                         "ASCENDING_REAR": "AR",
                                         "DESCENDING_FRONT": "DF",
                                         "DOWN_STAIRS": "DS",
                                         "DESCENDING_REAR": "DR"}

        self.short_to_state_name_dict = {v: k for k, v in self.state_to_short_name_dict.items()}

        self.current_base_link_frame = self.base_link_dict["base_link"]

        # Initialize state machine policy
        self.init_control_policies()

        self.reset_time = time.time()
        self.init_ros()

    def init_control_policies(self):
        self.current_state = "NEUTRAL"

        self.policy_left = StateClassifier(feat_dim=self.config["feat_vec_dim"], state_list=self.state_list, linear=False)
        self.policy_right = StateClassifier(feat_dim=self.config["feat_vec_dim"], state_list=self.state_list, linear=False)
        param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "opt/agents/sm_full/imitation_state_classification_non_lin.p")
        self.policy_left.load_state_dict(T.load(param_path), strict=False)
        self.policy_right.load_state_dict(T.load(param_path), strict=False)

        self.last_state_change_time = time.time()

    def init_ros(self):
        rospy.init_node("art_flipper_controller")

        self.ros_rate = rospy.Rate(self.config["ros_rate"])
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.flippers_pos_publisher = rospy.Publisher("marv/flippers_position_controller/cmd_vel",
                                                      MarvFloat64MultiArray,
                                                      queue_size=1)

        self.flippers_max_torque_publisher = rospy.Publisher("marv/flippers_max_torque_controller/cmd_vel",
                                                      MarvFloat64MultiArray,
                                                      queue_size=1)

        self.flipper_positions_subscriber = ros_utilities.subscriber_factory("art/flipper_positions", Float64MultiArray)
        self.pc_feat_vec_subscriber = ros_utilities.subscriber_factory("art/pc_feat_vec", Float64MultiArray)
        self.pc_feat_msg_subscriber = ros_utilities.subscriber_factory("art/pc_feat_msg", MarvPCFeats)
        self.teleop_state_subscriber = ros_utilities.subscriber_factory("teleop/state", String)
        self.controller_state_subscriber = ros_utilities.subscriber_factory("art/marv_flipper_controller_state", String)
        self.stagnation_subscriber = ros_utilities.subscriber_factory("art/marv_progress_stagnation", Float64)

        time.sleep(0.1)

    def get_current_obs_dict(self):
        pose_dict = ros_utilities.get_robot_pose_dict(self.config["root_frame"],
                                                      "X1/base_link",
                                                      self.tf_buffer,
                                                      rospy.Time(0))
        if pose_dict is None: return None

        with self.flipper_positions_subscriber.lock:
            if self.flipper_positions_subscriber.msg is None:
                return None
            proprioceptive_data = self.flipper_positions_subscriber.msg.data
            proprioceptive_data_dict = {"front_left_flipper" : proprioceptive_data[0],
                                        "front_right_flipper" : proprioceptive_data[1],
                                        "rear_left_flipper": proprioceptive_data[2],
                                        "rear_right_flipper": proprioceptive_data[3]}
        #if proprioceptive_data_dict is None: return None

        with self.pc_feat_msg_subscriber.lock:
            if self.pc_feat_msg_subscriber.msg is None: return None
            exteroceptive_data_dict = {"frontal_low_feat" : self.pc_feat_msg_subscriber.msg.frontal_low_feat.data,
                                       "frontal_mid_feat" : self.pc_feat_msg_subscriber.msg.frontal_mid_feat.data,
                                       "rear_low_feat" : self.pc_feat_msg_subscriber.msg.rear_low_feat.data,
                                       "fl_flipper_feat" : self.pc_feat_msg_subscriber.msg.fl_flipper_feat.data,
                                       "fr_flipper_feat" : self.pc_feat_msg_subscriber.msg.fr_flipper_feat.data,
                                       "rl_flipper_feat" : self.pc_feat_msg_subscriber.msg.rl_flipper_feat.data,
                                       "rr_flipper_feat" : self.pc_feat_msg_subscriber.msg.rr_flipper_feat.data}

        with self.pc_feat_vec_subscriber.lock:
            if self.pc_feat_vec_subscriber.msg is None: return None
            exteroceptive_data_dict["pc_feat_vec"] = self.pc_feat_vec_subscriber.msg.data

        with self.stagnation_subscriber.lock:
            if self.stagnation_subscriber.msg is None: return None
            proprioceptive_data_dict["stagnation"] = self.stagnation_subscriber.msg.data

        return utilities.merge_dicts([pose_dict, proprioceptive_data_dict, exteroceptive_data_dict])

    def get_current_feature_vec(self, obs_dict, location):
        if obs_dict is None: return None

        if location=="left":
            feature_vec = [obs_dict["euler"][1]] + list(obs_dict["fl_flipper_feat"]) + list(obs_dict["rl_flipper_feat"])
        elif location=="right":
            feature_vec = [obs_dict["euler"][1]] + list(obs_dict["fr_flipper_feat"]) + list(obs_dict["rr_flipper_feat"])
        else:
            raise NotImplementedError

        return feature_vec

    def get_current_state(self):
        with self.controller_state_subscriber.lock:
            if self.controller_state_subscriber.msg is not None:
                return self.controller_state_subscriber.msg.data
        with self.teleop_state_subscriber.lock:
            if self.teleop_state_subscriber.msg is not None:
                return self.short_to_state_name_dict[self.teleop_state_subscriber.msg.data]

    def calculate_flipper_action(self, state_left, state_right, obs_dict):
        roll, pitch, yaw = obs_dict["euler"]

        fl_flipper_stab_correction = 0
        fr_flipper_stab_correction = 0
        rl_flipper_stab_correction = 0
        rr_flipper_stab_correction = 0
        if self.config["enable_flipper_stabilization"]:
            fl_flipper_stab_correction = -roll * self.config["roll_stabilization_coeff"]
            fr_flipper_stab_correction = roll * self.config["roll_stabilization_coeff"]
            rl_flipper_stab_correction = roll * self.config["roll_stabilization_coeff"]
            rr_flipper_stab_correction = -roll * self.config["roll_stabilization_coeff"]

        fl_flipper_correction = 0
        rl_flipper_correction = 0
        fr_flipper_correction = 0
        rr_flipper_correction = 0

        if state_left == "ASCENDING_FRONT":
            fl_flipper_correction = -obs_dict["stagnation"] * 0.2
            rl_flipper_correction = obs_dict["stagnation"] * 0.2

        if state_right == "ASCENDING_FRONT":
            fr_flipper_correction = -obs_dict["stagnation"] * 0.2
            rr_flipper_correction = obs_dict["stagnation"] * 0.2

        if state_left == "ASCENDING_REAR":
            fl_flipper_correction = obs_dict["stagnation"] * 0.5
            rl_flipper_correction = -obs_dict["stagnation"] * 0.7

        if state_right == "ASCENDING_REAR":
            fr_flipper_correction = obs_dict["stagnation"] * 0.5
            rr_flipper_correction = -obs_dict["stagnation"] * 0.7

        if state_left == "DESCENDING_FRONT":
            fl_flipper_correction = obs_dict["stagnation"] * 0.4
            rl_flipper_correction = - obs_dict["stagnation"] * 0.5

        if state_right == "DESCENDING_FRONT":
            fr_flipper_correction = obs_dict["stagnation"] * 0.4
            rr_flipper_correction = - obs_dict["stagnation"] * 0.5

        if state_left == "UP_STAIRS" or state_left == "DOWN_STAIRS":
            fl_flipper_correction = obs_dict["stagnation"] * 0.1
            rl_flipper_correction = - obs_dict["stagnation"] * 0.1

        if state_right == "UP_STAIRS" or state_right == "DOWN_STAIRS":
            fr_flipper_correction = obs_dict["stagnation"] * 0.1
            rr_flipper_correction = - obs_dict["stagnation"] * 0.1

        flipper_commands_dict = {}
        flipper_commands_dict["front_left"] = self.config["FLIPPERS_{}".format(state_left)][0] + fl_flipper_stab_correction + fl_flipper_correction
        flipper_commands_dict["front_right"] = self.config["FLIPPERS_{}".format(state_right)][1] + fr_flipper_stab_correction + fr_flipper_correction
        flipper_commands_dict["rear_left"] = self.config["FLIPPERS_{}".format(state_left)][2] + rl_flipper_stab_correction + rl_flipper_correction
        flipper_commands_dict["rear_right"] = self.config["FLIPPERS_{}".format(state_right)][3] + rr_flipper_stab_correction + rr_flipper_correction

        flipper_torques_dict = {}
        flipper_torques_dict["front_left"] = self.config["FLIPPERS_CURRENT_{}".format(state_left)][0]
        flipper_torques_dict["front_right"] = self.config["FLIPPERS_CURRENT_{}".format(state_right)][0]
        flipper_torques_dict["rear_left"] = self.config["FLIPPERS_CURRENT_{}".format(state_left)][1]
        flipper_torques_dict["rear_right"] = self.config["FLIPPERS_CURRENT_{}".format(state_right)][1]

        return flipper_commands_dict, flipper_torques_dict

    def publish_flipper_pos(self, flipper_dict):
        if not self.enable_flippers:
            return

        # Publish flippers vel
        flippers_pos_msg = MarvFloat64MultiArray()
        if self.current_base_link_frame == self.config["robot_prefix"] + "base_link_zrp":
            flippers_pos_msg.data = [flipper_dict["front_left"], flipper_dict["front_right"],
                                     flipper_dict["rear_left"], flipper_dict["rear_right"]]
        else:
            flippers_pos_msg.data = [flipper_dict["rear_right"], flipper_dict["rear_left"],
                                     flipper_dict["front_right"],flipper_dict["front_left"]]
        self.flippers_pos_publisher.publish(flippers_pos_msg)

    def publish_flipper_torque_limits(self, flipper_dict):
        if not self.enable_flippers:
            return

        # Publish flippers vel
        flippers_torque_msg = MarvFloat64MultiArray()
        if self.current_base_link_frame == self.config["robot_prefix"] + "base_link_zrp":
            flippers_torque_msg.data = [flipper_dict["front_left"], flipper_dict["front_right"],
                                     flipper_dict["rear_left"], flipper_dict["rear_right"]]
        else:
            flippers_torque_msg.data = [flipper_dict["rear_right"], flipper_dict["rear_left"],
                                     flipper_dict["front_right"], flipper_dict["front_left"]]
        self.flippers_max_torque_publisher.publish(flippers_torque_msg)

    def step(self, new_state_left, new_state_right, obs_dict):

        flipper_commands_dict, flipper_torques_dict = self.calculate_flipper_action(new_state_left, new_state_right, obs_dict)

        self.publish_flipper_pos(flipper_commands_dict)
        self.publish_flipper_torque_limits(flipper_torques_dict)

    def reset(self):
        self.current_state = "NEUTRAL"
        self.current_tracking_orientation = "fw"
        self.reset_time = time.time()

    def loop(self):
        rospy.loginfo("Art flipper controller starting flipper control...")

        while not rospy.is_shutdown():
            # Observations
            # Observations
            obs_dict = self.get_current_obs_dict()
            current_state = self.get_current_state()
            if obs_dict is None or current_state is None:
                rospy.loginfo_throttle(1, "Obs_dict was none, skipping iteration")
                self.ros_rate.sleep()
                continue
            obs_feat_vec_left = self.get_current_feature_vec(obs_dict, "left")
            obs_feat_vec_right = self.get_current_feature_vec(obs_dict, "right")

            # Calculate new state
            new_state_left, _ = self.policy_left.decide_next_state(obs_feat_vec_left)
            new_state_right, _ = self.policy_right.decide_next_state(obs_feat_vec_right)

            # Perform control step
            self.step(new_state_left, new_state_right, obs_dict)

            self.ros_rate.sleep()

def main():
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) == 2:
        config_name = myargv[1]
    else:
        config_name = "marv_flipper_controller_bilateral_config.yaml"

    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/{}".format(config_name)), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    controller = MarvFlipperControllerBilateral(config)
    controller.loop()

if __name__=="__main__":
    main()


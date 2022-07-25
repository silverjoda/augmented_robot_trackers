#!/usr/bin/python

import os
import sys
import time

import rospy
import tf2_ros
from augmented_robot_trackers.msg import MarvPCFeats
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Float64MultiArray
from std_msgs.msg import String

from src.policies.policies import *
from src.utilities import utilities, ros_utilities
from sensor_msgs.msg import Imu

class TradrFlipperModulator:
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

        # Initialize system variables (momentum, baselink frames, etc)
        self.current_base_link_frame = self.base_link_dict["base_link"]

        self.reset_time = time.time()
        self.init_ros()

    def init_ros(self):
        rospy.init_node("art_flipper_controller")

        self.ros_rate = rospy.Rate(self.config["ros_rate"])
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.flippers_pos_fl_publisher = rospy.Publisher("/X1/flippers_cmd_pos/front_left",
                                                         Float64,
                                                         queue_size=1)
        self.flippers_pos_fr_publisher = rospy.Publisher("/X1/flippers_cmd_pos/front_right",
                                                         Float64,
                                                         queue_size=1)
        self.flippers_pos_rl_publisher = rospy.Publisher("/X1/flippers_cmd_pos/rear_left",
                                                         Float64,
                                                         queue_size=1)
        self.flippers_pos_rr_publisher = rospy.Publisher("/X1/flippers_cmd_pos/rear_right",
                                                         Float64,
                                                         queue_size=1)

        self.flippers_torque_fl_publisher = rospy.Publisher("/X1/flippers_cmd_max_torque/front_left",
                                                            Float64,
                                                            queue_size=1)
        self.flippers_torque_fr_publisher = rospy.Publisher("/X1/flippers_cmd_max_torque/front_right",
                                                            Float64,
                                                            queue_size=1)
        self.flippers_torque_rl_publisher = rospy.Publisher("/X1/flippers_cmd_max_torque/rear_left",
                                                            Float64,
                                                            queue_size=1)
        self.flippers_torque_rr_publisher = rospy.Publisher("/X1/flippers_cmd_max_torque/rear_right",
                                                            Float64,
                                                            queue_size=1)

        self.flipper_positions_subscriber = ros_utilities.subscriber_factory("art/flipper_positions", Float64MultiArray)
        self.pc_feat_vec_subscriber = ros_utilities.subscriber_factory("art/pc_feat_vec", Float64MultiArray)
        self.pc_feat_msg_subscriber = ros_utilities.subscriber_factory("art/pc_feat_msg", MarvPCFeats)
        self.teleop_state_subscriber = ros_utilities.subscriber_factory("teleop/state", String)
        self.controller_state_subscriber = ros_utilities.subscriber_factory("art/tradr_flipper_controller_state", String)
        self.stagnation_subscriber = ros_utilities.subscriber_factory("art/tradr_progress_stagnation", Float64)
        self.imu_subscriber = ros_utilities.subscriber_factory("X1/imu/data", Imu)

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

    def get_current_state(self):
        with self.controller_state_subscriber.lock:
            if self.controller_state_subscriber.msg is not None:
                return self.controller_state_subscriber.msg.data
        with self.teleop_state_subscriber.lock:
            if self.teleop_state_subscriber.msg is not None:
                return self.short_to_state_name_dict[self.teleop_state_subscriber.msg.data]

    def calculate_flipper_action(self, current_state, obs_dict):
        roll, pitch, yaw = obs_dict["euler"]

        roll_vel = 0
        with self.imu_subscriber.lock:
            if self.imu_subscriber.msg is not None:
                roll_vel = self.imu_subscriber.msg.angular_velocity.x

        fl_flipper_stab_correction = 0
        fr_flipper_stab_correction = 0
        rl_flipper_stab_correction = 0
        rr_flipper_stab_correction = 0
        if self.config["enable_flipper_stabilization"]:
            fl_flipper_stab_correction = -roll * self.config["roll_stabilization_p"] + roll_vel * self.config["roll_stabilization_d"]
            fr_flipper_stab_correction = roll * self.config["roll_stabilization_p"] - roll_vel * self.config["roll_stabilization_d"]
            rl_flipper_stab_correction = roll * self.config["roll_stabilization_p"] - roll_vel * self.config["roll_stabilization_d"]
            rr_flipper_stab_correction = -roll * self.config["roll_stabilization_p"] + roll_vel * self.config["roll_stabilization_d"]

        front_flipper_correction = 0
        rear_flipper_correction = 0

        if current_state == "ASCENDING_FRONT":
            front_flipper_correction = -obs_dict["stagnation"] * 0.2
            rear_flipper_correction = obs_dict["stagnation"] * 0.2

        if current_state == "ASCENDING_REAR":
            front_flipper_correction = obs_dict["stagnation"] * 0.5
            rear_flipper_correction = -obs_dict["stagnation"] * 0.7

        if current_state == "DESCENDING_FRONT":
            front_flipper_correction = obs_dict["stagnation"] * 0.4
            rear_flipper_correction = - obs_dict["stagnation"] * 0.5

        if current_state == "UP_STAIRS" or current_state == "DOWN_STAIRS":
            front_flipper_correction = obs_dict["stagnation"] * 0.10
            rear_flipper_correction = - obs_dict["stagnation"] * 0.2

        flipper_commands_dict = {}
        flipper_commands_dict["front_left"] = self.config["FLIPPERS_{}".format(current_state)][0] + fl_flipper_stab_correction + front_flipper_correction
        flipper_commands_dict["front_right"] = self.config["FLIPPERS_{}".format(current_state)][1] + fr_flipper_stab_correction + front_flipper_correction
        flipper_commands_dict["rear_left"] = self.config["FLIPPERS_{}".format(current_state)][2] + rl_flipper_stab_correction + rear_flipper_correction
        flipper_commands_dict["rear_right"] = self.config["FLIPPERS_{}".format(current_state)][3] + rr_flipper_stab_correction + rear_flipper_correction

        flipper_torques_dict = {}
        flipper_torques_dict["front_left"] = self.config["FLIPPERS_CURRENT_{}".format(current_state)][0]
        flipper_torques_dict["front_right"] = self.config["FLIPPERS_CURRENT_{}".format(current_state)][0]
        flipper_torques_dict["rear_left"] = self.config["FLIPPERS_CURRENT_{}".format(current_state)][1]
        flipper_torques_dict["rear_right"] = self.config["FLIPPERS_CURRENT_{}".format(current_state)][1]

        return flipper_commands_dict, flipper_torques_dict

    def publish_flipper_pos(self, flipper_dict):
        if not self.enable_flippers:
            return

        # Publish flippers vel
        if self.current_base_link_frame == self.config["robot_prefix"] + "base_link_zrp":
            flippers_pos_cmd_data = [flipper_dict["front_left"], flipper_dict["front_right"],
                                     flipper_dict["rear_left"], flipper_dict["rear_right"]]
        else:
            flippers_pos_cmd_data = [flipper_dict["rear_right"], flipper_dict["rear_left"],
                                     flipper_dict["front_right"],flipper_dict["front_left"]]

        self.flippers_pos_fl_publisher.publish(Float64(data=flippers_pos_cmd_data[0]))
        self.flippers_pos_fr_publisher.publish(Float64(data=flippers_pos_cmd_data[1]))
        self.flippers_pos_rl_publisher.publish(Float64(data=flippers_pos_cmd_data[2]))
        self.flippers_pos_rr_publisher.publish(Float64(data=flippers_pos_cmd_data[3]))

    def publish_flipper_torque_limits(self, flipper_dict):
        if not self.enable_flippers:
            return

        # Publish flippers vel
        if self.current_base_link_frame == self.config["robot_prefix"] + "base_link_zrp":
            flippers_torque_data = [flipper_dict["front_left"], flipper_dict["front_right"],
                                     flipper_dict["rear_left"], flipper_dict["rear_right"]]
        else:
            flippers_torque_data = [flipper_dict["rear_right"], flipper_dict["rear_left"],
                                     flipper_dict["front_right"], flipper_dict["front_left"]]
        self.flippers_torque_fl_publisher.publish(Float64(data=flippers_torque_data[0]))
        self.flippers_torque_fr_publisher.publish(Float64(data=flippers_torque_data[1]))
        self.flippers_torque_rl_publisher.publish(Float64(data=flippers_torque_data[2]))
        self.flippers_torque_rr_publisher.publish(Float64(data=flippers_torque_data[3]))

    def loop(self):
        rospy.loginfo("Art flipper controller starting flipper control...")

        while not rospy.is_shutdown():
            # Observations
            obs_dict = self.get_current_obs_dict()
            current_state = self.get_current_state()
            if obs_dict is None or current_state is None:
                rospy.loginfo_throttle(1, "Obs_dict was none, skipping iteration")
                self.ros_rate.sleep()
                continue

            flipper_commands_dict, flipper_torques_dict = self.calculate_flipper_action(current_state, obs_dict)

            self.publish_flipper_pos(flipper_commands_dict)
            self.publish_flipper_torque_limits(flipper_torques_dict)

            self.ros_rate.sleep()

def main():
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) == 2:
        config_name = myargv[1]
    else:
        config_name = "tradr_flipper_modulator_config.yaml"

    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/{}".format(config_name)), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    controller = TradrFlipperModulator(config)
    controller.loop()

if __name__=="__main__":
    main()


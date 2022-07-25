#!/usr/bin/python

import os
import sys
import time

import rospy
import tf.transformations
import tf2_ros
from augmented_robot_trackers.msg import MarvPCFeats
from geometry_msgs.msg import Twist
from marv_msgs.msg import Float64MultiArray as MarvFloat64MultiArray
from std_msgs.msg import Float64, Float64MultiArray
from std_msgs.msg import String

from src.policies.policies import *
from src.utilities import utilities, ros_utilities
import threading
from sensor_msgs.msg import Imu
from std_srvs.srv import Trigger, TriggerResponse

class MarvFlipperMonitor:
    def __init__(self):
        self.state_list = ["NEUTRAL",
                            "ASCENDING_FRONT",
                            "UP_STAIRS",
                            "ASCENDING_REAR",
                            "DESCENDING_FRONT",
                            "DOWN_STAIRS",
                            "DESCENDING_REAR"]

        self.state_to_short_name_dict = {"NEUTRAL": "N",
                                         "ASCENDING_FRONT": "AF",
                                         "UP_STAIRS": "US",
                                         "ASCENDING_REAR": "AR",
                                         "DESCENDING_FRONT": "DF",
                                         "DOWN_STAIRS": "DS",
                                         "DESCENDING_REAR": "DR"}

        self.short_to_state_name_dict = {v: k for k, v in self.state_to_short_name_dict.items()}

        self.reset_all_vars()

        self.init_ros()


    def init_ros(self):
        rospy.init_node("art_flipper_monitor")
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.controller_state_subscriber = ros_utilities.subscriber_factory("art/marv_flipper_controller_state", String)

        self.state_lock = threading.Lock()
        rospy.Subscriber("art/marv_flipper_controller_state",
                         String,
                         self._ros_state_callback, queue_size=1)

        self.imu_lock = threading.Lock()
        rospy.Subscriber("X1/imu/data",
                         Imu,
                         self._ros_imu_callback, queue_size=1)

        self.reset_service = rospy.Service('/art_flipper_monitor/reset', Trigger, self.trigger_response)

    def get_current_obs_dict(self):
        pose_dict = ros_utilities.get_robot_pose_dict("map",
                                                      "X1/base_link",
                                                      self.tf_buffer,
                                                      rospy.Time(0))
        return pose_dict


    def reset_all_vars(self):
        self.cumulative_acceleration = 0
        self.cumulative_angular_pitch_vel = 0
        self.body_roll_list = []
        self.maximal_acceleration = 0
        self.maximal_angular_vel = 0
        self.n_state_changes = 0
        self.start_time = time.time()
        self.state_transition_dict = {}
        self.current_state = "NEUTRAL"

    def trigger_response(self, request):
        print("===========================================")
        print("Cumulative square acceleration", self.cumulative_acceleration)
        print("Cumulative angular pitch vel", self.cumulative_angular_pitch_vel)
        print("Maximal acceleration", self.maximal_acceleration)
        print("Maximal angular vel", self.maximal_angular_vel)
        print("Bodyroll min: {}, max: {}, mean, {}, std: {}, mean_abs_sum: {}".format(min(self.body_roll_list), max(self.body_roll_list), np.mean(self.body_roll_list), np.std(self.body_roll_list), np.mean(np.abs(self.body_roll_list))))
        print("Amount of state changes", self.n_state_changes)
        print("Time taken", time.time() - self.start_time)
        print("State transition dict", self.state_transition_dict)
        print("===========================================")
        print("\n\n\n")
        self.reset_all_vars()

        return TriggerResponse(
            success=True,
            message="-"
        )

    def _ros_state_callback(self, data):
        with self.state_lock:
            trans = (self.current_state, data.data)
            self.n_state_changes += (trans[0] != trans[1])
            if trans in self.state_transition_dict:
                self.state_transition_dict[trans] += 1
            else:
                self.state_transition_dict[trans] = 1
            self.current_state = data.data

    def _ros_imu_callback(self, data):
        with self.imu_lock:
            accel_sum = np.maximum(np.sum(np.abs(np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]))) - 11, 0)
            ang_vel = np.maximum(np.abs(np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])) - 0.1, 0)

            self.cumulative_acceleration += np.square(accel_sum)
            self.cumulative_angular_pitch_vel += np.square(ang_vel[1])

            if accel_sum > self.maximal_acceleration:
                self.maximal_acceleration = accel_sum

            if np.max(ang_vel) > self.maximal_angular_vel:
                self.maximal_angular_vel = np.max(ang_vel)

            obs_dict = self.get_current_obs_dict()
            if obs_dict is not None:
                self.body_roll_list.append(obs_dict["euler"][0])


def main():
    MarvFlipperMonitor()
    rospy.spin()

if __name__=="__main__":
    main()


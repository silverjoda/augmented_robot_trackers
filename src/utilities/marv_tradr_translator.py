#!/usr/bin/env python
import os
import threading
import time
from copy import deepcopy

import rospy
from geometry_msgs.msg import Twist
# from rds_msgs.msg import K3
# from spot_msgs.srv import SetLocomotion
# from spot_msgs.msg import MobilityParams
from marv_msgs.msg import Float64MultiArray
from nifti_robot_driver_msgs.msg import (FlippersState,
                                         FlippersVel,
                                         FlippersCurrentControllerParams)
from nifti_teleop_client.client import TeleopClient
from nifti_teleop_client.msg import Priority as MsgPriority
from nifti_teleop_client.priority import PRIORITIES_BY_VALUE

import src.utilities as utilities

class MarvTradrTranslator:
    def __init__(self, config):
        self.config = config
        self.init_ros("marv_tradr_translator_node")

    def init_ros(self, name):
        rospy.loginfo("Starting marv-tradr translator node")
        rospy.init_node(name)

        self.ros_rate = rospy.Rate(self.config["ros_rate"])

        self.tracks_vel_cmd_in_lock = threading.Lock()
        self.tracks_vel_cmd_in_data = None

        self.tracks_vel_cmd_in_joy_lock = threading.Lock()
        self.tracks_vel_cmd_in_joy_data = None

        self.flipper_vel_cmd_in_lock = threading.Lock()
        self.flipper_vel_cmd_in_data = None

        self.flipper_pos_cmd_in_lock = threading.Lock()
        self.flipper_pos_cmd_in_data = None

        self.flipper_torque_cmd_in_lock = threading.Lock()
        self.flipper_torque_cmd_in_data = None

        self.flipper_publisher = TeleopClient(PRIORITIES_BY_VALUE[MsgPriority.PRIORITY_NAV],
                                              None,
                                              'flippers_cmd',
                                              None,
                                              queue_size=1)

        self.flipper_current_limit_publisher = rospy.Publisher('flippers_current_controller_params',
                                                               FlippersCurrentControllerParams,
                                                               queue_size=1)

        # Tracks vel publisher
        self.tracks_vel_publisher = TeleopClient(PRIORITIES_BY_VALUE[MsgPriority.PRIORITY_NAV],
                                                 None,
                                                 'cmd_vel',
                                                 None,
                                                 queue_size=10)

        self.tracks_vel_publisher_joy = TeleopClient(PRIORITIES_BY_VALUE[MsgPriority.PRIORITY_LOCAL_JOY],
                                                 None,
                                                 'cmd_vel',
                                                 None,
                                                 queue_size=10)

        rospy.Subscriber(self.config["tracks_velocity_control_out"],
                         Twist,
                         self._ros_track_vel_cmd_in_callback, queue_size=1)

        rospy.Subscriber(self.config["tracks_velocity_control_out_joy"],
                         Twist,
                         self._ros_track_vel_cmd_in_joy_callback, queue_size=1)

        rospy.Subscriber(self.config["flippers_velocity_control_out"],
                         Float64MultiArray,
                         self._ros_flipper_vel_cmd_in_callback, queue_size=1)

        rospy.Subscriber(self.config["flippers_position_control_out"],
                         Float64MultiArray,
                         self._ros_flipper_pos_cmd_in_callback, queue_size=1)

        rospy.Subscriber(self.config["flippers_max_torque_control_out"],
                         Float64MultiArray,
                         self._ros_flipper_torque_cmd_in_callback, queue_size=1)

        time.sleep(0.1)

        self.flipper_publisher.acquire()
        self.tracks_vel_publisher.acquire()
        self.tracks_vel_publisher_joy.acquire()

    def _ros_track_vel_cmd_in_callback(self, data):
        with self.tracks_vel_cmd_in_lock:
            self.tracks_vel_cmd_in_data = data

    def _ros_track_vel_cmd_in_joy_callback(self, data):
        with self.tracks_vel_cmd_in_joy_lock:
            self.tracks_vel_cmd_in_joy_data = data

    def _ros_flipper_vel_cmd_in_callback(self, data):
        with self.flipper_vel_cmd_in_lock:
            self.flipper_vel_cmd_in_data = data

    def _ros_flipper_pos_cmd_in_callback(self, data):
        with self.flipper_pos_cmd_in_lock:
            self.flipper_pos_cmd_in_data = data

    def _ros_flipper_torque_cmd_in_callback(self, data):
        with self.flipper_torque_cmd_in_lock:
            self.flipper_torque_cmd_in_data = data

    def loop(self):
        while not rospy.is_shutdown():
            with self.tracks_vel_cmd_in_lock:
                tracks_vel_cmd_in_data = deepcopy(self.tracks_vel_cmd_in_data)

            with self.tracks_vel_cmd_in_joy_lock:
                tracks_vel_cmd_in_joy_data = deepcopy(self.tracks_vel_cmd_in_joy_data)

            with self.flipper_pos_cmd_in_lock:
                flipper_pos_cmd_in_data = deepcopy(self.flipper_pos_cmd_in_data)

            with self.flipper_torque_cmd_in_lock:
                flipper_torque_cmd_in_data = deepcopy(self.flipper_torque_cmd_in_data)

            if tracks_vel_cmd_in_data is not None:
                msg = Twist()
                msg.linear.x = tracks_vel_cmd_in_data.linear.x
                msg.angular.z = tracks_vel_cmd_in_data.angular.z

                try:
                    self.tracks_vel_publisher.publish(msg, True)
                except:
                    rospy.logerr("Marv-tradr translator: Tracks vel publish failure")

            if tracks_vel_cmd_in_joy_data is not None:
                msg = Twist()
                msg.linear.x = tracks_vel_cmd_in_joy_data.linear.x
                msg.angular.z = tracks_vel_cmd_in_joy_data.angular.z

                try:
                    self.tracks_vel_publisher_joy.publish(msg, True)
                except:
                    rospy.logerr("Marv-tradr translator: Tracks vel joy publish failure")

            if flipper_pos_cmd_in_data is not None:
                msg = FlippersState()
                msg.frontLeft = flipper_pos_cmd_in_data.data[0]
                msg.frontRight = flipper_pos_cmd_in_data.data[1]
                msg.rearLeft = flipper_pos_cmd_in_data.data[2]
                msg.rearRight = flipper_pos_cmd_in_data.data[3]

                try:
                    self.flipper_publisher.publish(msg, True)
                except:
                    rospy.logerr("Marv-tradr translator: Flippers publish failure")

            if flipper_torque_cmd_in_data is not None:
                msg = FlippersCurrentControllerParams()
                msg.frontLeft = flipper_torque_cmd_in_data.data[0]
                msg.frontRight = flipper_torque_cmd_in_data.data[1]
                msg.rearLeft = flipper_torque_cmd_in_data.data[2]
                msg.rearRight = flipper_torque_cmd_in_data.data[3]

                self.flipper_current_limit_publisher.publish(msg)

            self.ros_rate.sleep()


def main():
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/marv_teleop_config.yaml"), 'r') as f:
        teleop_config = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(os.path.dirname(__file__), "..", "path_follower/configs/marv_tracker_config.yaml"), 'r') as f:
        tracker_config = yaml.load(f, Loader=yaml.FullLoader)

    config = utilities.merge_two_dicts(teleop_config, tracker_config)
    tracker = MarvTradrTranslator(config)
    tracker.loop()

if __name__=="__main__":
    main()

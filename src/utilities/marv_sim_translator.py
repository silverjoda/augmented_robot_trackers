#!/usr/bin/env python
import os
import threading
import time
from copy import deepcopy

import rospy
from marv_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64

import src.utilities as utilities

class MarvSimTranslator:
    def __init__(self):

        self.init_ros("marv_sim_translator_node")

    def init_ros(self, name):
        rospy.loginfo("Starting marv translator node")
        rospy.init_node(name)

        self.ros_rate = rospy.Rate(30)

        self.flipper_pos_cmd_in_lock = threading.Lock()
        self.flipper_pos_cmd_in_data = None

        self.flipper_torque_cmd_in_lock = threading.Lock()
        self.flipper_torque_cmd_in_data = None

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

        rospy.Subscriber("marv/flippers_position_controller/cmd_vel",
                         Float64MultiArray,
                         self._ros_flipper_pos_cmd_in_callback, queue_size=1)
        rospy.Subscriber("marv/flippers_max_torque_controller/cmd_vel",
                         Float64MultiArray,
                         self._ros_flipper_torque_cmd_in_callback, queue_size=1)

        time.sleep(2)

    def _ros_flipper_pos_cmd_in_callback(self, data):
        with self.flipper_pos_cmd_in_lock:
            self.flipper_pos_cmd_in_data = data
            self.flipper_pos_cmd_in_data.header.stamp.secs = rospy.get_time()

    def _ros_flipper_torque_cmd_in_callback(self, data):
        with self.flipper_torque_cmd_in_lock:
            self.flipper_torque_cmd_in_data = data
            self.flipper_torque_cmd_in_data.header.stamp.secs = rospy.get_time()

    def is_data_stale(self, ros_message, timeout):
        #if ros_message is not None:
        #    print(rospy.get_time(), ros_message.header.stamp.to_sec())
        return ros_message is None or (rospy.get_time() - ros_message.header.stamp.to_sec()) > timeout

    def loop(self):
        while not rospy.is_shutdown():
            with self.flipper_pos_cmd_in_lock:
                flipper_pos_cmd_in_data = deepcopy(self.flipper_pos_cmd_in_data)

            with self.flipper_torque_cmd_in_lock:
                flipper_torque_cmd_in_data = deepcopy(self.flipper_torque_cmd_in_data)

            if not self.is_data_stale(flipper_pos_cmd_in_data, 0.3):
                self.flippers_pos_fl_publisher.publish(Float64(data=flipper_pos_cmd_in_data.data[0]))
                self.flippers_pos_fr_publisher.publish(Float64(data=flipper_pos_cmd_in_data.data[1]))
                self.flippers_pos_rl_publisher.publish(Float64(data=flipper_pos_cmd_in_data.data[2]))
                self.flippers_pos_rr_publisher.publish(Float64(data=flipper_pos_cmd_in_data.data[3]))

            if not self.is_data_stale(flipper_torque_cmd_in_data, 0.3):
                self.flippers_torque_fl_publisher.publish(Float64(data=flipper_torque_cmd_in_data.data[0]))
                self.flippers_torque_fr_publisher.publish(Float64(data=flipper_torque_cmd_in_data.data[1]))
                self.flippers_torque_rl_publisher.publish(Float64(data=flipper_torque_cmd_in_data.data[2]))
                self.flippers_torque_rr_publisher.publish(Float64(data=flipper_torque_cmd_in_data.data[3]))

            self.ros_rate.sleep()

def main():
    tracker = MarvSimTranslator()
    tracker.loop()

if __name__=="__main__":
    main()

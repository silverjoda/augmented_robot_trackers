#!/usr/bin/python

import os
import pickle
import sys
import threading
import time

import numpy as np
import ros_numpy
import rospy
import tf2_ros
from augmented_robot_trackers.msg import MarvPCFeats
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64, Float64MultiArray
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from tf.transformations import quaternion_matrix
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from nifti_robot_driver_msgs.msg import TracksStamped

from src.utilities import utilities, ros_utilities
from copy import deepcopy

class TradrOdomGenerator:
    def __init__(self):
        self.init_ros()

    def init_ros(self):
        rospy.init_node("tradr_tracks_to_odom")
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.current_position = [0,0,0]
        self.current_orientation = [0,0,0,1]

        self.odom_publisher = rospy.Publisher("art_odom",
                                                      Odometry,
                                                      queue_size=1)

        rospy.Subscriber("tracks_vel",
                         TracksStamped,
                         self._ros_tracks_vel_callback, queue_size=1)

    def _ros_tracks_vel_callback(self, msg):
        # Make odom message and publish
        pass

def main():
    TradrOdomGenerator()
    rospy.spin()

if __name__=="__main__":
    main()


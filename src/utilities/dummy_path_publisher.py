#!/usr/bin/env python
import os
import sys
import threading
import time
from copy import deepcopy

import numpy as np
import ros_numpy
import rospy
import tf2_ros
from geometry_msgs.msg import Twist, PoseStamped
from marv_msgs.msg import Float64MultiArray
from nav_msgs.msg import Path
from sensor_msgs.msg import Joy
from std_msgs.msg import Empty

import src.utilities as utilities

class DummyPath:
    def __init__(self, config):
        self.config = config

        self.init_ros("dummy_path_publisher")
        rospy.loginfo("Starting dummy path publisher")

    def init_ros(self, name):
        rospy.init_node(name)

        self.ros_rate = rospy.Rate(30)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.config["tf_buffer_size"]))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.joy_plan_publisher = rospy.Publisher(self.config["joy_plan_out"],
                                                    Path,
                                                    queue_size=1)

        rospy.Subscriber("time_reset",
                         Empty,
                         self._ros_time_reset_cb, queue_size=1)

        time.sleep(2)

    def _ros_time_reset_cb(self):
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.config["tf_buffer_size"]))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def loop(self):
        while not rospy.is_shutdown():
            self.publish_plan()
            try:
                self.ros_rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def publish_plan(self):
        # Get most recent transform to root frame
        try:
            trans = self.tf_buffer.lookup_transform(self.config["root_frame"],
                                                    self.config["robot_prefix"] + "base_link",
                                                    rospy.Time(0),
                                                    rospy.Duration(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn("Dummy path: process waypoints: TRANSFORMATION OLD, err: {}".format(err))
            return [], 0

        trans_np = ros_numpy.numpify(trans.transform)

        processed_path = []
        for i in range(self.config["joy_plan_n_waypoints"]):
            wp_x = i * self.config["joy_plan_waypoints_gap"]
            wp_y = 0
            wp_glob = np.matmul(trans_np, np.array([wp_x, wp_y, 0, 1]))[:3]
            processed_path.append(wp_glob)

        joy_plan_msg = Path()
        joy_plan_msg.header.frame_id = self.config["root_frame"]
        for p in processed_path:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self.config["root_frame"]
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = p
            joy_plan_msg.poses.append(pose_msg)
        self.joy_plan_publisher.publish(joy_plan_msg)

def main():
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) == 2:
        config_name = myargv[1]
    else:
        config_name = "marv_teleop_config.yaml"

    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/{}".format(config_name)), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    obj = DummyPath(config)
    obj.loop()

if __name__=="__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
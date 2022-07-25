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

class StaticPathGen:
    def __init__(self, config):
        self.config = config
        self.make_dense_waypoints_from_config()

        self.init_ros()

        rospy.loginfo("Starting static path generator")

    def init_ros(self):
        rospy.init_node("static_path_generator")

        self.ros_rate = rospy.Rate(5)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.config["tf_buffer_size"]))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.joy_plan_publisher = rospy.Publisher("static_path_out",
                                                    Path,
                                                    queue_size=1)

        rospy.Subscriber("reset",
                         Empty,
                         self._ros_reset_cb, queue_size=1)

        time.sleep(2)

    def make_dense_waypoints_from_config(self):
        self.dense_waypoint_list = []

        # Go through all points and fill in
        # for i in range(len(self.config["rough_waypoint_list"]) - 1):
        #     dwp_c = self.config["rough_waypoint_list"][i]
        #     dwp_n = self.config["rough_waypoint_list"][i + 1]
        #     self.dense_waypoint_list.append(dwp_c)
        #     dist_wp = self.xy_dist_between_waypoints(dwp_c, dwp_n)
        #     if dist_wp > self.config["waypoint_gap"]:
        #         n_extra_wps = int(float(dist_wp) / self.config["waypoint_gap"]) - 1
        #         gap = float(dist_wp) / float(n_extra_wps + 1)
        #         new_wps_raw = np.linspace(dwp_c, dwp_n)
        #         for k in range(1, len(new_wps_raw) - 1):
        #             self.dense_waypoint_list.append(new_wps_raw[k])
        #
        ## Add last point
        #self.dense_waypoint_list.append(self.config["rough_waypoint_list"][-1])

        for wp in self.config["rough_waypoint_list_{}".format(self.config["course"])]:
            self.dense_waypoint_list.append(wp)

        self.reset()

    def xy_dist_between_waypoints(self, wp1, wp2):
        return np.sqrt((wp1[0] - wp2[0]) ** 2 + (wp1[1] - wp2[1]) ** 2)

    def _ros_reset_cb(self):
        self.reset()

    def reset(self):
        self.current_target_idx = 0

    def update_plan(self):
        # Get pose of robot
        # Get most recent transform to root frame
        try:
            trans = self.tf_buffer.lookup_transform(self.config["root_frame"],
                                                    self.config["robot_prefix"] + "base_link",
                                                    rospy.Time(0),
                                                    rospy.Duration(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn("Static path generator: fun=update_plan msg=TRANSFORMATION OLD, err: {}".format(err))
            return

        pos_list = [trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z]

        # If robot has visited current target, update index
        if self.xy_dist_between_waypoints(pos_list, self.dense_waypoint_list[self.current_target_idx]) < self.config["waypoint_reach_dist"]:
            self.current_target_idx = (self.current_target_idx + 1) % len(self.dense_waypoint_list)


    def publish_plan(self):
        static_path_msg = Path()
        static_path_msg.header.frame_id = self.config["root_frame"]
        for i in range(self.current_target_idx, len(self.dense_waypoint_list)):
            p = self.dense_waypoint_list[i]
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self.config["root_frame"]
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = p
            static_path_msg.poses.append(pose_msg)
        self.joy_plan_publisher.publish(static_path_msg)

    def loop(self):
        while not rospy.is_shutdown():
            self.update_plan()
            self.publish_plan()
            self.ros_rate.sleep()

def main():
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) == 2:
        config_path = myargv[1]
    else:
        config_path = "static_path_config.yaml"

    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/{}".format(config_path)), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    obj = StaticPathGen(config)
    obj.loop()

if __name__=="__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/python

import os
import sys
import threading
import time
from copy import deepcopy
import numpy as np
import rospy
import tf2_ros
from augmented_robot_trackers.srv import GetTrackerParams, SetTrackerParams
from geometry_msgs.msg import Twist, PointStamped, Pose, PoseStamped
from marv_msgs.msg import Float64MultiArray as MarvFloat64MultiArray
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from std_srvs.srv import Trigger, SetBool
from tf.transformations import euler_from_quaternion, quaternion_matrix
from visualization_msgs.msg import Marker

from src.utilities import utilities

class PathProcessor:
    def __init__(self, config):
        self.config = config
        self.init_ros()

    def init_ros(self):
        rospy.init_node("art_path_processor")

        self.ros_rate = rospy.Rate(self.config["ros_rate"])
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.current_path = []
        self.current_target = None

        self.path_lock = threading.Lock()
        self.pose_lock = threading.Lock()

        self.current_target_publisher = rospy.Publisher("current_tracker_target",
                                                        PointStamped,
                                                        queue_size=1)

        self.current_path_publisher = rospy.Publisher("processed_path",
                                                        Path,
                                                        queue_size=1)

        rospy.Subscriber("path_in",
                         Path,
                         self._ros_path_callback, queue_size=1)

    def _ros_path_callback(self, data):
        with self.path_lock:
            self.path_data = data
            self.new_path = True

    def _ros_pose_callback(self, data):
        with self.pose_lock:
            self.pose_data = data

    def get_current_target_and_path(self, pose_dict):
        with self.path_lock:
            # If empty path or we didn't receive path messages yet
            if self.path_data is None or len(self.path_data.poses) == 0 or pose_dict is None:
                return self.current_target, self.current_path

            if not self.new_path:
                return self.current_target, self.current_path

            if len(self.path_data.poses) == 1:
                self.current_path = self.path_data
            else:
                if self.config["enable_plan_filtering"]:
                    # Find closest point in path, discard and start from next
                    self.current_path = self.filter_initial_path(self.path_data, self.current_target, pose_dict)
                else:
                    self.current_path = self.path_data

            while True:
                if self.current_target is not None and utilities.dist_between_pose_and_position(self.current_target, pose_dict["position"]) >= self.config[
                    "waypoint_reach_distance"]:
                    break
                if len(self.current_path.poses) > 0:
                    self.current_target = self.current_path.poses[0]
                    del self.current_path.poses[0]
                if len(self.current_path.poses) == 0 or self.current_target is not None:
                    break

        return self.current_target, self.current_path

    def filter_initial_path(self, path, current_target, pose_dict):
        # Initial dist
        smallest_dist = utilities.dist_between_pose_and_position(path.poses[0], pose_dict["position"])
        ctr = 1
        for i in range(1, len(path.poses)):
            dist = utilities.dist_between_poses(path.poses[i], current_target)
            if dist > smallest_dist:
                break
            smallest_dist = dist
            ctr = i

        # Delete past waypoints
        del path.poses[:ctr]

        self.new_path = False

        return path

    def set_current_target(self, current_target):
        self.current_target = current_target

    def delete_pose_waypoint(self, idx):
        del self.current_path.poses[idx]

    def get_robot_pose_dict(self):
        # Get pose using TF
        try:
            trans = self.tf_buffer.lookup_transform(self.config["root_frame"],
                                                    "X1/base_link",
                                                    rospy.Time(0),
                                                    rospy.Duration(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn_throttle(1, "Get_robot_pose_dict: TRANSFORMATION ERROR, err: {}".format(err))
            return None

        # Translation
        pos = trans.transform.translation

        # Orientation
        quat = trans.transform.rotation
        roll, pitch, yaw = euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))
        rot_mat = quaternion_matrix((quat.x, quat.y, quat.z, quat.w))

        # Directional vectors
        x1, y1 = [np.cos(yaw), np.sin(yaw)]

        pose_dict = {"position": pos,
                     "quat": quat,
                     "matrix": rot_mat,
                     "euler": (roll, pitch, yaw),
                     "dir_vec": (x1, y1)}

        # Give results in quaterion, euler and vector form
        return pose_dict

    def dist_to_target(self, robot_pose, target):
        pos_delta = np.sqrt(np.square(robot_pose.position.x - target.pose.position.x)
                            + np.square(robot_pose.position.y - target.pose.position.y)
                            + np.square(robot_pose.position.z - target.pose.position.z))
        return pos_delta

    def publish_current_plan(self):
        path_msg = Path()
        path_msg.header.frame_id = self.config["root_frame"]
        for p in self.current_path:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self.config["root_frame"]
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = p
            path_msg.poses.append(pose_msg)
        self.current_path_publisher.publish(path_msg)

    def loop(self):
        while not rospy.is_shutdown():
            robot_pose = self.get_robot_pose_dict()

            # If new plan, find closest relevant target
            with self.path_lock:
                if hasattr(self, "path_data"):
                    path = deepcopy(self.path_data)
                    if self.new_path and self.config["enable_path_filtering"]:
                        self.current_path = self.filter_initial_path()
                    else:
                        self.current_path = path

                    # If old plan, See if we have reached target
                    pos_delta = self.dist_to_target(robot_pose, self.current_path.poses[0])
                    if len(self.path_data.poses) > 1:
                        if np.abs(pos_delta) < self.config["waypoint_reach_distance"]:
                            del self.path_data.poses[0]

                # Update and publish plan
                self.publish_current_plan()


def main():
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) == 2:
        config_name = myargv[1]
    else:
        config_name = "path_processor_config.yaml"

    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/{}".format(config_name)), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    path_proc = PathProcessor(config)
    path_proc.loop()

if __name__=="__main__":
    main()


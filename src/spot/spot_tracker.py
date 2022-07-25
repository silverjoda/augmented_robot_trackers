#!/usr/bin/env python

import src.utilities as utilities
import pickle
import os
import rospy
import numpy as np
import time
import tf
import tf2_ros
from nav_msgs.msg import Path
import threading
from tf.transformations import euler_from_quaternion, quaternion_matrix
from geometry_msgs.msg import Twist, Vector3, Pose, PoseArray, PointStamped

class SpotTracker:
    def __init__(self, config):
        self.config = config

        # Variables
        self.current_tracking_orientation = "fw" # fw / bw

        self.init_ros(self.config["node_name"])

    def init_ros(self, name):
        rospy.init_node(name)

        self.ros_rate = rospy.Rate(self.config["ros_rate"])
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.config["tf_buffer_size"]))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.current_path = []
        self.current_target = None

        self.path_data = None
        self.path_lock = threading.Lock()

        self.force_dir_data = Vector3()
        self.force_dir_data.x, self.force_dir_data.y, self.force_dir_data.z = 0, 0, -1
        self.force_dir_lock = threading.Lock()

        rospy.Subscriber(self.config["planner_path_topic"],
                         Path,
                         self._ros_path_callback, queue_size=1)

        rospy.Subscriber(self.config["force_dir_topic"],
                         Vector3,
                         self._ros_force_dir_callback, queue_size=1)

        self.cmd_vel_publisher = rospy.Publisher(self.config["cmd_vel_publish_topic"],
                                                 Twist,
                                                 queue_size=1)

        self.current_target_publisher = rospy.Publisher(self.config["current_target_topic"],
                                                 PointStamped,
                                                 queue_size=1)

        time.sleep(1)

    def _ros_path_callback(self, data):
        with self.path_lock:
            self.path_data = data
            self.new_path = True

    def _ros_force_dir_callback(self, data):
        with self.force_dir_lock:
            self.force_dir_data = data

    def get_current_target_and_path(self):
        pose_dict = self.get_robot_pose_dict()
        with self.path_lock:
            # If empty path
            if len(self.path_data.poses) == 0:
                return self.current_target, self.current_path

            if not self.new_path:
                return self.current_target, self.current_path

            if len(self.path_data.poses) == 1:
                self.current_path = self.path_data
            else:
                # Find closest point in path, discard and start from next
                self.current_path = self.filter_initial_path(self.path_data, self.current_target, pose_dict)

            while True:
                if len(self.current_path.poses) == 0:
                    self.current_target = None
                    break
                self.current_target = self.current_path.poses[0]
                del self.current_path.poses[0]
                if utilities.dist_between_pose_and_position(self.current_target, pose_dict["position"]) >= self.config["waypoint_reach_distance"]:
                    break
        return self.current_target, self.current_path

    def filter_initial_path(self, path, current_target, pose_dict):
        if current_target is None:
            return path

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
                                                    self.config["spot_root_frame"],
                                                    rospy.Time(0),
                                                    rospy.Duration(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn("Get_robot_pose_dict: TRANSFORMATION OLD, err: {}".format(err))
            return

        # Translation
        pos = trans.transform.translation

        # Orientation
        quat = trans.transform.rotation
        roll, pitch, yaw = euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))
        rot_mat = quaternion_matrix((quat.x, quat.y, quat.z, quat.w))

        # Directional vectors
        x1, y1 = [np.cos(yaw), np.sin(yaw)]

        pose_dict = {"position" : pos,
                     "quat" : quat,
                     "matrix" : rot_mat,
                     "euler" : (roll, pitch, yaw),
                     "dir_vec" : (x1,  y1)}

        # Give results in quaterion, euler and vector form
        return pose_dict

    def calculate_target_dev_spot(self, position, yaw, current_target):
        # Check if we currently have a force dir command
        with self.force_dir_lock: force_dir = (self.force_dir_data.z == 1)

        # Distance between robot pose and target
        pos_delta = np.sqrt(np.square(position.x - current_target.pose.position.x)
                            + np.square(position.y - current_target.pose.position.y)
                            + np.square(position.z - current_target.pose.position.z))

        # Vector in x,y in which robot is facing
        x_rob, y_rob = [np.cos(yaw), np.sin(yaw)]

        # Directional xy vector towards target
        x_tar, y_tar = (current_target.pose.position.x - position.x,
                        current_target.pose.position.y - position.y)

        x_force, y_force = self.force_dir_data.x, self.force_dir_data.y

        # Delta to target
        det_tar = x_rob * y_tar - y_rob * x_tar
        dot_tar = x_rob * x_tar + y_rob * y_tar
        theta_delta_tar = np.arctan2(det_tar, dot_tar)

        # Delta to force_dir
        det_force = x_rob * y_force - y_rob * x_force
        dot_force = x_rob * x_force + y_rob * y_force
        theta_delta_force = np.arctan2(det_force, dot_force)

        # Relative target dir (for strafing)
        if force_dir:
            xy_lateral = [np.cos(theta_delta_force), np.sin(theta_delta_force)]
            theta_delta = theta_delta_force
        else:
            xy_lateral = [np.cos(theta_delta_tar), np.sin(theta_delta_tar)]
            theta_delta = theta_delta_tar

        return pos_delta, theta_delta, xy_lateral

    def calculate_cmd_from_dev_spot(self, pos_dev, angle_dev, xy_target_relative):
        #with self.force_dir_lock: force_dir = (self.force_dir_data.z == 1)
        cmd_angular = angle_dev * self.config["turn_vel_sensitivity"]
        cmd_x = xy_target_relative[0] * pos_dev * self.config["lin_vel_sensitivity"] \
            * np.maximum(self.config["turn_vel_thresh"] - abs(angle_dev) * self.config["turn_inhibition_sensitivity"], 0)
        cmd_y = xy_target_relative[1] * self.config["strafe_vel_sensitivity"]
        return cmd_x, cmd_y, cmd_angular

    def calculate_cmd_vel(self, pose_dict, current_target):
        if current_target is None:
            return 0, 0, 0, None

        # Calculate angle deviation
        pos_dev, angle_dev, xy_lateral = self.calculate_target_dev_spot(pose_dict["position"], pose_dict["euler"][2], current_target)

        # Feed through PD
        cmd_x, cmd_y, cmd_angular = self.calculate_cmd_from_dev_spot(pos_dev, angle_dev, xy_lateral)

        # Clip to max bounds
        cmd_x_clipped = np.clip(cmd_x, -self.config["cmd_vel_lin_clip"], self.config["cmd_vel_lin_clip"])
        cmd_y_clipped = np.clip(cmd_y, -self.config["cmd_vel_strafe_clip"], self.config["cmd_vel_strafe_clip"])
        cmd_angular_clipped = np.clip(cmd_angular, -self.config["cmd_vel_ang_clip"], self.config["cmd_vel_ang_clip"])

        with self.force_dir_lock: force_dir = (self.force_dir_data.z == 1)
        if force_dir:
            cmd_y_clipped = np.clip(cmd_y_clipped, -self.config["force_max_strafe_vel"],
                                    self.config["force_max_strafe_vel"])
            cmd_x_clipped = np.clip(cmd_x_clipped, -self.config["force_max_fw_vel"],
                                    self.config["force_max_fw_vel"])

        return cmd_x_clipped, cmd_y_clipped, cmd_angular_clipped, pos_dev

    def publish_cmd_vel(self, linear, angular, strafe=0):
        if self.config["enable_cmd_vel"]:
            msg = Twist()
            msg.linear.x = linear
            msg.linear.y = strafe
            msg.angular.z = angular
            self.cmd_vel_publisher.publish(msg)

    def publish_current_target(self):
        if self.current_target is None: return
        msg = PointStamped()
        msg.point.x = self.current_target.pose.position.x
        msg.point.y = self.current_target.pose.position.y
        msg.point.z = self.current_target.pose.position.z
        msg.header.frame_id = self.config["root_frame"]
        self.current_target_publisher.publish(msg)

    def step(self):
        # Find nearest point in path
        current_target, current_path = self.get_current_target_and_path()

        # Get robot orientation
        pose_dict = self.get_robot_pose_dict()

        # Calculate turn deltas
        cmd_x, cmd_y, cmd_angular, dist_to_target = self.calculate_cmd_vel(pose_dict, current_target)

        # Check if we have reached target
        if current_target is not None and np.abs(dist_to_target) < self.config["waypoint_reach_distance"]:
            if len(current_path.poses) == 0:
                cmd_x, cmd_y, cmd_angular = 0, 0, 0
                rospy.loginfo("Reached terminal waypoint.")
            else:
                self.set_current_target(current_path.poses[0])
                self.delete_pose_waypoint(0)

        # Publish cmd values
        self.publish_cmd_vel(cmd_x, cmd_angular, cmd_y)

        # Publish debug info
        self.publish_current_target()

        # Sleep
        self.ros_rate.sleep()

    def start_tracking(self):
        rospy.loginfo("{} starting tracking...".format(self.config["tracker_node_name"]))

        while not rospy.is_shutdown():
            self.step()

def main():
    import yaml
    with open("../control/configs/smart_tracker_config.yaml") as f:
        tracker_config = yaml.load(f, Loader=yaml.FullLoader)

    tracker = SpotTracker(tracker_config)
    tracker.start_tracking()

if __name__=="__main__":
    main()
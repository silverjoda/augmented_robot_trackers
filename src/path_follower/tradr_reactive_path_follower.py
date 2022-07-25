#!/usr/bin/python

import os
import sys
import time
import threading

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion, quaternion_matrix
from augmented_robot_trackers.msg import BumperActivations

class PathFollower:
    def __init__(self, config):
        self.config = config
        self.linear_tracking_momentum = 0
        self.init_ros()

    def init_ros(self):
        rospy.init_node("path_follower_node")

        self.ros_rate = rospy.Rate(10)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.path_lock = threading.Lock()
        self.bumper_activations_lock = threading.Lock()

        self.last_state_change_time = time.time()

        self.tracks_vel_publisher = rospy.Publisher("X1/cmd_vel_tar",
                                                    Twist,
                                                    queue_size=1)

        rospy.Subscriber("bumper_activations",
                         BumperActivations,
                         self._ros_bumper_activations_in_callback, queue_size=3)

        rospy.Subscriber("static_path_out",
                         Path,
                         self._ros_path_callback, queue_size=1)

    def _ros_bumper_activations_in_callback(self, data):
        with self.bumper_activations_lock:
            self.bumper_activations_data = data

    def _ros_path_callback(self, data):
        with self.path_lock:
            self.path_data = data

    def update_linear_momentum(self, cmd_vel_linear):
        if self.linear_tracking_momentum >= 0:
            if cmd_vel_linear >= self.linear_tracking_momentum:
                update_val = np.minimum(self.config["tracker_momentum_vel_x_increment"], cmd_vel_linear - self.linear_tracking_momentum)
            else:
                update_val = np.maximum(-self.config["tracker_momentum_vel_x_decrement"], cmd_vel_linear - self.linear_tracking_momentum)
        else:
            if cmd_vel_linear >= self.linear_tracking_momentum:
                update_val = np.minimum(self.config["tracker_momentum_vel_x_decrement"],
                                        cmd_vel_linear - self.linear_tracking_momentum)
            else:
                update_val = np.maximum(-self.config["tracker_momentum_vel_x_increment"],
                                        cmd_vel_linear - self.linear_tracking_momentum)

        self.linear_tracking_momentum = np.clip(self.linear_tracking_momentum + update_val,
                                                -self.config["cmd_vel_lin_clip"],
                                                self.config["cmd_vel_lin_clip"])

    def calculate_cmd_vel(self, pose_dict, current_target):
        if current_target is None or pose_dict is None:
            return 0, 0, 0, 0, False

        position = pose_dict["position"]
        yaw = pose_dict["euler"][2]

        # Distance between robot pose and target
        pos_delta = np.sqrt(np.square(position.x - current_target.pose.position.x)
                            + np.square(position.y - current_target.pose.position.y)
                            + np.square(position.z - current_target.pose.position.z))

        # Vector in x,y in which robot is facing
        x1, y1 = [np.cos(yaw), np.sin(yaw)]

        # Directional xy vector towards target
        x2, y2 = (current_target.pose.position.x - position.x,
                  current_target.pose.position.y - position.y)

        det = x1 * y2 - y1 * x2
        dot = x1 * x2 + y1 * y2

        theta_delta = np.arctan2(det, dot)

        changed_dir = False
        if self.config["bidirectional_tracking"]:
            x1_bw, y1_bw = -x1, -y1

            det_bw = x1_bw * y2 - y1_bw * x2
            dot_bw = x1_bw * x2 + y1_bw * y2
            theta_delta_rev = np.arctan2(det_bw, dot_bw)
            if abs(theta_delta_rev) < abs(theta_delta):
                if self.current_base_link_frame == self.config["robot_prefix"] + "base_link_zrp":
                    self.current_base_link_frame = self.config["robot_prefix"] + "base_link_zrp_rev"
                else:
                    self.current_base_link_frame = self.config["robot_prefix"] + "base_link_zrp"
                theta_delta = theta_delta_rev
                changed_dir = True


        with self.bumper_activations_lock:
            if hasattr(self, "bumper_activations_data"):
                bumper_active = self.bumper_activations_data.front_left.data \
                                or self.bumper_activations_data.front_right.data \
                                or self.bumper_activations_data.rear_left.data \
                                or self.bumper_activations_data.rear_right.data \
                                or self.bumper_activations_data.side_front_left.data \
                                or self.bumper_activations_data.side_rear_left.data \
                                or self.bumper_activations_data.side_front_right.data \
                                or self.bumper_activations_data.side_rear_right.data \

                adaptive_turn_vel_thresh = self.config["turn_vel_thresh"] + int(bumper_active) * 2
            else:
                adaptive_turn_vel_thresh = self.config["turn_vel_thresh"]

        cmd_angular = theta_delta * self.config["turn_vel_sensitivity"]
        cmd_linear = pos_delta * self.config["lin_vel_sensitivity"] \
                     * np.maximum(self.config["turn_vel_thresh"] - abs(theta_delta) * self.config["turn_inhibition_sensitivity"], 0)

        self.update_linear_momentum(cmd_linear)

        # Clip to max bounds
        cmd_linear_clipped = np.clip(self.linear_tracking_momentum, -self.config["cmd_vel_lin_clip"], self.config["cmd_vel_lin_clip"])
        cmd_angular_clipped = np.clip(cmd_angular, -self.config["cmd_vel_ang_clip"], self.config["cmd_vel_ang_clip"])

        return cmd_linear_clipped, cmd_angular_clipped, pos_delta, theta_delta, changed_dir

    def get_robot_pose_dict(self):
        # Get pose using TF
        try:
            trans = self.tf_buffer.lookup_transform(self.config["root_frame"],
                                                    "X1/base_link",
                                                    rospy.Time(0),
                                                    rospy.Duration(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn("Get_robot_pose_dict: TRANSFORMATION ERROR, err: {}".format(err))
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

    def publish_track_vel(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.linear.y = 0
        msg.angular.z = angular
        self.tracks_vel_publisher.publish(msg)

    def update_path(self, pos_delta):
        if hasattr(self, "path_data") and len(self.path_data.poses) > 1:
            if np.abs(pos_delta) < self.config["waypoint_reach_dist"]:
                del self.path_data.poses[0]

    def step(self):
        # Get required observations
        robot_pose_dict = self.get_robot_pose_dict()

        # Calculate tracking velocity
        cmd_linear, cmd_angular = 0, 0
        with self.path_lock:
            if hasattr(self, "path_data") and len(self.path_data.poses) > 0:
                target = self.path_data.poses[0]
                cmd_linear, cmd_angular, pos_delta, theta_delta, changed_dir = self.calculate_cmd_vel(robot_pose_dict, target)

                self.update_path(pos_delta)

        # Step the linear momentum variables
        self.update_linear_momentum(cmd_linear)

        self.publish_track_vel(self.linear_tracking_momentum, cmd_angular)
        self.ros_rate.sleep()

    def loop(self):
        rospy.loginfo("{} starting path following...".format("path_follower_node"))
        while not rospy.is_shutdown():
            self.step()

def main():
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) == 2:
        config_name = myargv[1]
    else:
        config_name = "tradr_reactive_path_follower_config.yaml"

    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/{}".format(config_name)), 'r') as f:
        tracker_config = yaml.load(f, Loader=yaml.FullLoader)

    path_follower = PathFollower(tracker_config)
    path_follower.loop()

if __name__=="__main__":
    main()


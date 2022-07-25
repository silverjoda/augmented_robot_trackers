#!/usr/bin/env python

import os
import threading
import time
from copy import deepcopy

import numpy as np
import ros_numpy
import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from augmented_robot_trackers.msg import BumperActivations
from visualization_msgs.msg import Marker

class VirtualBumper:
    def __init__(self, config):
        self.config = config
        self.init_ros(name="virtual_bumper_tradr")
        self.bumper_activated = False

    def init_ros(self, name):
        rospy.init_node(name)
        self.ros_rate = rospy.Rate(self.config["ros_rate"])

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.laser_data = None
        self.cmd_vel_in_data = None

        rospy.Subscriber("X1/points", # points_filtered
                         PointCloud2,
                         self._ros_laser_callback, queue_size=3)

        rospy.Subscriber("X1/cmd_vel_tar",
                         Twist,
                         self._ros_cmd_vel_in_callback, queue_size=3)

        self.cmd_vel_publisher = rospy.Publisher("X1/cmd_vel", #cmd_vel_desired_speed
                                                 Twist,
                                                 queue_size=1)

        self.bumpers_vis_publisher = rospy.Publisher("bumpers_vis",
                                                 PointCloud2,
                                                 queue_size=1)

        self.bumper_activation_publisher = rospy.Publisher("bumper_activations",
                                                     BumperActivations,
                                                     queue_size=1)

        self.marker_publisher = rospy.Publisher("bumper_bbx",
                                                Marker,
                                                queue_size=1)

        self.laser_lock = threading.Lock()
        self.cmd_vel_in_lock = threading.Lock()

        time.sleep(1)

    def _ros_laser_callback(self, data):
        with self.laser_lock:
            self.laser_data = data

    def _ros_cmd_vel_in_callback(self, data):
        with self.cmd_vel_in_lock:
            self.cmd_vel_in_data = data

    def loop_virtual_bumper(self):
        while not rospy.is_shutdown():
            # Read and process cmd_vel_in
            target_lin, target_ang = self.read_cmd_vel()

            # Read and process cloud data. Bumper srough_waypoint_list_reactiveize is scaled according to velocity
            bumper_activation_dict, bumper_dist_dict, bumper_pc_dict = self.get_bumper_activations(target_lin, target_ang)

            # Calculate new cmd_vel
            cmd_vel_lin, cmd_vel_ang = self.calculate_cmd_vel(target_lin, target_ang, bumper_activation_dict, bumper_dist_dict)

            # Publish result
            self.publish_cmd_vel(cmd_vel_lin, cmd_vel_ang)
            self.publish_bumper_activations(bumper_activation_dict)
            self.publish_bumpers_vis(bumper_pc_dict)
            self.publish_bumpers_bbx(bumper_activation_dict)

            self.ros_rate.sleep()

    def read_cmd_vel(self):
        with self.cmd_vel_in_lock:
            if self.cmd_vel_in_data is None:
                lin_cmd = 0.00001
                ang_cmd = 0
            else:
                lin_cmd = self.cmd_vel_in_data.linear.x
                ang_cmd = self.cmd_vel_in_data.angular.z
        return lin_cmd, ang_cmd

    def get_bumper_activations(self, target_lin, target_ang):
        bumper_list = ["fl", "fr", "rl", "rr", "sfl", "sfr", "srl", "srr"]

        # Fill in zero values
        bumper_activation_dict = {}
        bumper_dist_dict = {}
        for k in bumper_list:
            bumper_activation_dict[k] = 0
            bumper_dist_dict[k] = 0

        # If laser stopped coming or is old then assume no obstacles
        with self.laser_lock:
            if self.laser_data is None: return bumper_activation_dict, bumper_dist_dict, None

        pc = self.get_laser_in_baselink()
        if pc is None:
            return bumper_activation_dict, bumper_dist_dict, None

        # Make bumper point regions out of point cloud
        bumper_pc_dict = {}
        for k in bumper_list:
            # Rescaled bounds
            rescaled_bnds = self.get_dynamic_bnds(k, target_lin)

            # Get points from bounds
            bumper_pc_dict[k] = self.get_bnd_pts(pc, rescaled_bnds)

            # Simple rule to determine whether obstacle exists
            bumper_activation_dict[k] = len(bumper_pc_dict[k][0]) > self.config["bumper_detection_points_threshold"]

            # Distance of obstacle for each bumper (normalized 0,1)
            bumper_dist_dict[k] = self.get_bumper_dist(k, bumper_pc_dict[k], rescaled_bnds)

        self.bumper_activated = np.any(bumper_activation_dict.values())

        return bumper_activation_dict, bumper_dist_dict, bumper_pc_dict

    def get_laser_in_baselink(self):
        # Get local point cloud
        with self.laser_lock:
            pc = ros_numpy.numpify(self.laser_data).ravel()
            source_frame = self.laser_data.header.frame_id

        # Array of size (4, N_points). 4th coordinate is for homogeneous transformation
        pc_array = np.stack([pc[f] for f in ['x', 'y', 'z']] + [np.ones(pc.size)])
        pc_array_clean = pc_array[:, np.logical_not(np.any(np.isnan(pc_array), axis=0))]

        # Transform pc to baselink coordinates
        try:
            trans = self.tf_buffer.lookup_transform("X1",  # base_link
                                                    source_frame,
                                                    rospy.Time(0),
                                                    rospy.Duration(0))
            trans = ros_numpy.numpify(trans.transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn("Transforming laser to base_link in virtual bumper failed. ", err)
            return None

        pc_baselink = np.matmul(trans, pc_array_clean)[:3, :]

        return pc_baselink

    def get_static_bnds(self, bumper_name, target_lin):
        bnds = deepcopy(self.config["{}_bumper_bnds".format(bumper_name)])

        if bumper_name == "fl" or bumper_name == "fr":
            bumper_thickness = bnds[1] - bnds[0]
            bnds[1] = bnds[0] + bumper_thickness

        if bumper_name == "rl" or bumper_name == "rr":
            bumper_thickness = bnds[1] - bnds[0]
            bnds[0] = bnds[1] - bumper_thickness

        if bumper_name == "sfl":
            bumper_thickness = bnds[3] - bnds[2]
            bnds[3] = bnds[2] + bumper_thickness
        if bumper_name == "srr":
            bumper_thickness = bnds[3] - bnds[2]
            bnds[2] = bnds[3] - bumper_thickness
        if bumper_name == "sfr":
            bumper_thickness = bnds[3] - bnds[2]
            bnds[2] = bnds[3] - bumper_thickness
        if bumper_name == "srl":
            bumper_thickness = bnds[3] - bnds[2]
            bnds[3] = bnds[2] + bumper_thickness
        return bnds

    def get_dynamic_bnds(self, bumper_name, target_lin):
        bnds = deepcopy(self.config["{}_bumper_bnds".format(bumper_name)])

        if bumper_name == "fl" or bumper_name == "fr":
            bumper_thickness = bnds[1] - bnds[0]
            bnds[1] = bnds[0] \
                      + bumper_thickness \
                      + self.bumper_activated * self.config["bumper_hysteresis_thickness"]
        if bumper_name == "rl" or bumper_name == "rr":
            bumper_thickness = bnds[1] - bnds[0]
            bnds[0] = bnds[1] \
                      - bumper_thickness \
                      + self.bumper_activated * self.config["bumper_hysteresis_thickness"]

        if bumper_name == "sfl":
            bumper_thickness = bnds[3] - bnds[2]
            bnds[3] = bnds[2] + bumper_thickness
        if bumper_name == "srr":
            bumper_thickness = bnds[3] - bnds[2]
            bnds[2] = bnds[3] - bumper_thickness
        if bumper_name == "sfr":
            bumper_thickness = bnds[3] - bnds[2]
            bnds[2] = bnds[3] - bumper_thickness
        if bumper_name == "srl":
            bumper_thickness = bnds[3] - bnds[2]
            bnds[3] = bnds[2] + bumper_thickness
        return bnds

    def get_bnd_pts(self, pc, bnds):
        # Default bumper sizes
        min_x, max_x, min_y, max_y, min_z, max_z = bnds
        pts = pc[:, np.logical_and(pc[0, :] > min_x, pc[0, :] < max_x)]  # x
        pts = pts[:, np.logical_and(pts[1, :] > min_y, pts[1, :] < max_y)]  # y
        pts = pts[:, np.logical_and(pts[2, :] > min_z, pts[2, :] < max_z)]  # z
        return pts

    def get_bumper_dist(self, bumper_name, pc, bnds):
        if len(pc[0]) == 0: return 1
        min_x, max_x, min_y, max_y, min_z, max_z = bnds
        dist_norm = 0
        if bumper_name == "fl" or bumper_name == "fr":
            dist = np.median(pc[0, :])
            offset = min_x
            range = max_x - min_x
            dist_norm = (dist - offset) / range

        if bumper_name == "rl" or bumper_name == "rr":
            dist = np.median(pc[0, :])
            offset = max_x
            range = max_x - min_x
            dist_norm = -(dist - offset) / range

        if bumper_name == "sfl" or bumper_name == "srl":
            dist = np.median(pc[1, :])
            offset = min_y
            range = max_y - min_y
            dist_norm = (dist - offset) / range

        if bumper_name == "sfr" or bumper_name == "srr":
            dist = np.median(pc[1, :])
            offset = max_y
            range = max_y - min_y
            dist_norm = -(dist - offset) / range

        return dist_norm

    def calculate_cmd_vel(self, target_lin, target_ang, bumper_activation_dict, bumper_dist_dict):
        out_lin = target_lin
        out_ang = target_ang

        # Front partially blocked
        if target_lin > 0 and bumper_activation_dict["fl"]:
            out_lin = 0# target_lin * np.clip(1 * bumper_dist_dict["fl"], 0, 1)
            out_ang = -np.max([target_lin, np.abs(target_ang)])
        if target_lin > 0 and bumper_activation_dict["fr"]:
            out_lin = 0# target_lin * np.clip(1 * bumper_dist_dict["fr"], 0, 1)
            out_ang = np.max([target_lin, np.abs(target_ang)])

        # Back partially blocked
        if target_lin <= 0 and bumper_activation_dict["rl"]:
            out_lin = target_lin * np.clip(1 * bumper_dist_dict["rl"], 0, 1)
            out_ang = np.max([-target_lin, np.abs(target_ang), 0])
        if target_lin <= 0 and bumper_activation_dict["rr"]:
            out_lin = target_lin * np.clip(1 * bumper_dist_dict["rr"], 0, 1)
            out_ang = -np.max([-target_lin, np.abs(target_ang), 0])

        # Sides blocked
        out_ang_clipped = out_ang
        if bumper_activation_dict["sfl"]:# or bumper_activations["srr"]:
            out_ang_clipped = np.minimum(out_ang, 0)
        if bumper_activation_dict["sfr"]:# or bumper_activations["srl"]:
            out_ang_clipped = np.maximum(out_ang_clipped, 0)

        # Convert angular to forward linear (experimental)
        if not (bumper_activation_dict["fl"] or bumper_activation_dict["fr"]):
            if bumper_activation_dict["sfl"] and self.config["enable_angular_to_linear_translation"]:
                out_lin = np.maximum(target_ang, 0)
            if bumper_activation_dict["sfr"] and self.config["enable_angular_to_linear_translation"]:
                out_lin = -np.minimum(target_ang, 0)

        # Limit velocity if data is stale
        if self.laser_data is None:
            out_lin = np.clip(out_lin, -self.config["stale_laser_vel_val"], self.config["stale_laser_vel_val"])
            out_ang_clipped = np.clip(out_ang_clipped, -self.config["stale_laser_vel_val"],
                                      self.config["stale_laser_vel_val"])
            rospy.logwarn(
                "Warning, laser data is stale, limiting velocity to : {}".format(self.config["stale_laser_vel_val"]))

        # Completely zero the velocity if target input velocity is stale
        with self.cmd_vel_in_lock:
            if self.cmd_vel_in_data is None:
                out_lin = 0
                out_ang_clipped = 0

        return out_lin, out_ang_clipped

    def calculate_cmd_vel_experimental(self, target_lin, target_ang, bumper_activation_dict, bumper_dist_dict):
        out_lin = target_lin
        out_ang = target_ang

        # Front partially blocked
        if target_lin > 0 and bumper_activation_dict["fl"]:
            out_lin = target_lin * np.clip(1 * bumper_dist_dict["fl"], 0, 1)
            out_ang = -np.max([target_lin, np.abs(target_ang)])
        if target_lin > 0 and bumper_activation_dict["fr"]:
            out_lin = target_lin * np.clip(1 * bumper_dist_dict["fr"], 0, 1)
            out_ang = np.max([target_lin, np.abs(target_ang)])

        # Sides blocked
        out_ang_clipped = out_ang
        if bumper_activation_dict["sfl"]:
            out_ang_clipped = np.minimum(out_ang, 0, -(1 - bumper_dist_dict["sfl"]))
        if bumper_activation_dict["sfr"]:
            out_ang_clipped = np.maximum(out_ang_clipped, 0, (1 - bumper_dist_dict["sfr"]))

        # Convert angular to forward linear (experimental)
        if not (bumper_activation_dict["fl"] or bumper_activation_dict["fr"]):
            if bumper_activation_dict["sfl"] and self.config["enable_angular_to_linear_translation"]:
                out_lin = np.maximum(target_ang, 0)
            if bumper_activation_dict["sfr"] and self.config["enable_angular_to_linear_translation"]:
                out_lin = -np.minimum(target_ang, 0)

        # Limit velocity if data is stale
        if self.laser_data is None:
            out_lin = np.clip(out_lin, -self.config["stale_laser_vel_val"], self.config["stale_laser_vel_val"])
            out_ang_clipped = np.clip(out_ang_clipped, -self.config["stale_laser_vel_val"],
                                      self.config["stale_laser_vel_val"])
            rospy.logwarn(
                "Warning, laser data is stale, limiting velocity to : {}".format(self.config["stale_laser_vel_val"]))

        return out_lin, out_ang_clipped

    def publish_cmd_vel(self, lin, ang):
        lin_cmd = lin
        ang_cmd = ang

        msg = Twist()
        msg.linear.x = lin_cmd
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = ang_cmd

        self.cmd_vel_publisher.publish(msg)

    def publish_bumpers_vis(self, pc_dict):
        if pc_dict is None: return
        pc_bnds = np.concatenate((pc_dict["fl"],
                                  pc_dict["fr"],
                                  pc_dict["rl"],
                                  pc_dict["rr"],
                                  pc_dict["sfl"],
                                  pc_dict["sfr"],
                                  pc_dict["srl"],
                                  pc_dict["srr"]), axis=1)

        pc_bnds_data = np.zeros(len(pc_bnds[0]), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('vectors', np.float32, (3,))
        ])

        pc_bnds_data['x'] = pc_bnds[0, :]
        pc_bnds_data['y'] = pc_bnds[1, :]
        pc_bnds_data['z'] = pc_bnds[2, :]
        pc_bnds_data['vectors'] = np.arange(len(pc_bnds[0]))[:, np.newaxis]

        msg = ros_numpy.msgify(PointCloud2, pc_bnds_data)
        msg.header.frame_id = "X1/base_link"
        msg.header.stamp = rospy.Time.now()

        self.bumpers_vis_publisher.publish(msg)

    def publish_bumper_activations(self, bumper_activation_dict):
        msg = BumperActivations()
        msg.header.frame_id = "X1/base_link"
        msg.header.stamp = rospy.Time.now()

        msg.front_left.data = bumper_activation_dict["fl"]
        msg.front_right.data = bumper_activation_dict["fr"]
        msg.rear_left.data = bumper_activation_dict["rl"]
        msg.rear_left.data = bumper_activation_dict["rr"]
        msg.side_front_left.data = bumper_activation_dict["sfl"]
        msg.side_rear_left.data = bumper_activation_dict["srl"]
        msg.side_front_right.data = bumper_activation_dict["sfr"]
        msg.side_rear_right.data = bumper_activation_dict["srr"]

        self.bumper_activation_publisher.publish(msg)

    def publish_bumpers_bbx(self, bumper_activation_dict):
        self.publish_bbx(self.config["fl_bumper_bnds"], 1, [0.2 + 0.8 * bumper_activation_dict["fl"], 0.6, 0.6])
        self.publish_bbx(self.config["fr_bumper_bnds"], 2, [0.2 + 0.8 * bumper_activation_dict["fr"], 0.6, 0.6])
        self.publish_bbx(self.config["rl_bumper_bnds"], 3, [0.2 + 0.8 * bumper_activation_dict["rl"], 0.6, 0.6])
        self.publish_bbx(self.config["rr_bumper_bnds"], 4, [0.2 + 0.8 * bumper_activation_dict["rr"], 0.6, 0.6])

        self.publish_bbx(self.config["sfl_bumper_bnds"], 5, [0.2 + 0.8 * bumper_activation_dict["sfl"], 0.6, 0.2])
        self.publish_bbx(self.config["sfr_bumper_bnds"], 6, [0.2 + 0.8 * bumper_activation_dict["sfr"], 0.6, 0.2])
        self.publish_bbx(self.config["srl_bumper_bnds"], 7, [0.2 + 0.8 * bumper_activation_dict["srl"], 0.6, 0.2])
        self.publish_bbx(self.config["srr_bumper_bnds"], 8, [0.2 + 0.8 * bumper_activation_dict["srr"], 0.6, 0.2])

    def publish_bbx(self, bnds, id, rgb):
        xl, xu, yl, yu, zl, zu = bnds

        marker_msg = Marker()
        marker_msg.header.frame_id = "X1/base_link"
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.id = id
        marker_msg.type = marker_msg.CUBE
        marker_msg.action = marker_msg.ADD
        marker_msg.pose.position.x = (xl + xu) / 2
        marker_msg.pose.position.y = (yl + yu) / 2
        marker_msg.pose.position.z = (zl + zu) / 2
        marker_msg.pose.orientation.x = 0.0
        marker_msg.pose.orientation.y = 0.0
        marker_msg.pose.orientation.z = 0.0
        marker_msg.pose.orientation.w = 1.0
        marker_msg.scale.x = xu - xl
        marker_msg.scale.y = yu - yl
        marker_msg.scale.z = zu - zl
        marker_msg.color.a = 0.6

        marker_msg.color.r = rgb[0]
        marker_msg.color.g = rgb[1]
        marker_msg.color.b = rgb[2]

        self.marker_publisher.publish(marker_msg)

def main():
    with open(os.path.join(os.path.dirname(__file__), "configs/virtual_bumper_tradr_config.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    vb = VirtualBumper(config)
    vb.loop_virtual_bumper()

if __name__=="__main__":
    main()

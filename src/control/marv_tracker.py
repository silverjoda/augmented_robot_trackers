#!/usr/bin/python

import os
import pickle
import threading
import time
from scipy.spatial import KDTree
import torch as T

import sys
import numpy as np
import ros_numpy
import rospy
import tf2_ros
from augmented_robot_trackers.srv import GetTrackerParams, SetTrackerParams
from std_srvs.srv import Trigger, TriggerResponse, SetBool
from geometry_msgs.msg import Twist, PointStamped, Pose
from marv_msgs.msg import Float64MultiArray as MarvFloat64MultiArray
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_matrix
from visualization_msgs.msg import Marker

from src.utilities import utilities
from src.policies.policies import TrackerNNDual

class MarvTracker:
    def __init__(self, config):
        self.config = config

        self.base_link_dict = {"base_link" : self.config["robot_prefix"] + "base_link_zrp",
                               "base_link_rev" : self.config["robot_prefix"] + "base_link_zrp_rev"}

        self.all_states_list = ["NEUTRAL",
                                "ASCENDING_FRONT",
                                "ASCENDING_REAR",
                                "DESCENDING_FRONT",
                                "DESCENDING_REAR",
                                "UP_STAIRS",
                                "DOWN_STAIRS"]

        self.enable_flippers = self.config["enable_flippers"]

        # Initialize system variables (momentum, baselink frames, etc)
        self.init_system_params()

        # Initialize state machine policy
        self.init_smp()
        self.init_nn()

        # Initialize neural network policy
        # self.init_nn()

        # Possible states: Neutral, rough_terrain, climbing_front, climbing_rear,
        # descending front, descending rear, up_stairs, down_stairs

        self.reset_time = time.time()
        self.init_ros()

    def init_ros(self):
        rospy.init_node(self.config["node_name"])

        self.ros_rate = rospy.Rate(self.config["ros_rate"])
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.config["tf_buffer_size"]))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.current_path = []
        self.current_target = None

        self.path_data = None
        self.path_lock = threading.Lock()

        self.filtered_pc_data = None
        self.filtered_pc_array = None
        self.filtered_pc_lock = threading.Lock()

        self.reg_pc_data = None
        self.reg_pc_array = None
        self.reg_pc_lock = threading.Lock()

        self.trav_vis_pc_data = None
        self.trav_vis_pc_array = None
        self.trav_vis_kdtree = None
        self.trav_vis_pc_lock = threading.Lock()

        self.tracks_vel_publisher = rospy.Publisher("marv/cartesian_controller/cmd_vel",
                                                    Twist,
                                                    queue_size=1)

        self.flippers_pos_publisher = rospy.Publisher("marv/flippers_position_controller/cmd_vel",
                                                      MarvFloat64MultiArray,
                                                      queue_size=1)

        self.flippers_max_torque_publisher = rospy.Publisher("marv/flippers_max_torque_controller/cmd_vel",
                                                      MarvFloat64MultiArray,
                                                      queue_size=1)

        # Debug publishers
        self.current_target_publisher = rospy.Publisher("art/current_tracker_target",
                                                        PointStamped,
                                                        queue_size=1)

        self.pc_bl_zpr_publisher = rospy.Publisher("art/debug/pc_bl_zpr",
                                                   PointCloud2,
                                                   queue_size=1)

        self.text_publisher = rospy.Publisher("art/debug/text_info",
                                              String,
                                              queue_size=1)

        self.pc_bnds_publisher = rospy.Publisher("art/debug/pc_bnds",
                                                 PointCloud2,
                                                 queue_size=1)

        self.tracker_state_publisher = rospy.Publisher("art/marv_tracker_state",
                                                 String,
                                                 queue_size=1)

        self.current_pose_publisher = rospy.Publisher("art/current_pose",
                                                       Pose,
                                                       queue_size=1)

        self.marker_publisher = rospy.Publisher("art/feat_bbx_out",
                                                      Marker,
                                                      queue_size=1)

        self.pc_feat_vec_publisher = rospy.Publisher("art/pc_feat_vec",
                                                Float64MultiArray,
                                                queue_size=1)

        rospy.Subscriber("static_path_out", #"art/joy_plan , static_path_out
                         Path,
                         self._ros_path_callback, queue_size=1)

        rospy.Subscriber("dense_map",
                         PointCloud2,
                         self._ros_reg_pc_callback, queue_size=1)

        rospy.Subscriber("rds/traversability_visual",
                         PointCloud2,
                         self._ros_trav_vis_pc_callback, queue_size=1)

        # Services
        self.set_tracker_params_service = rospy.Service("set_tracker_params", SetTrackerParams, self.set_tracker_params_handler)
        self.get_tracker_params_service = rospy.Service("get_tracker_params", GetTrackerParams, self.get_tracker_params_handler)
        self.tracker_reset_service = rospy.Service("tracker_reset_service", Trigger, self.reset_handler)
        self.flipper_enabling_service = rospy.Service("flipper_enabling_service", SetBool, self.flipper_enabling_handler)

        time.sleep(0.1)

    def init_system_params(self):
        self.current_base_link_frame = self.base_link_dict["base_link"]

        self.linear_tracking_momentum = 0
        self.n_vel_timeout_secs = 3
        self.cmd_vel_transition_scalar = 1.0
        self.cmd_vel_transition_block_time = 0

    def init_nn(self):
        self.nn_policy = TrackerNNDual(self.config)
        if self.config["load_learned_parameters_nn"]:
            param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "opt/agents/imitation_nn.pth")
            self.nn_policy.load_state_dict(T.load(param_path), strict=False)

    def init_smp(self):
        self.current_state = "NEUTRAL"
        # Define learnable parameters (None means load from config)
        self.learnable_param_dict = {}

        if self.config["load_learned_parameters_sm"]:
            param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "opt/agents/TRN.pkl")
            self.learnable_param_dict["vector"] = pickle.load(open(param_path, "rb"))
        else:
            self.learnable_param_dict["vector"] = [0] * (12 + len(self.all_states_list) * 2)

        self.last_state_change_time = time.time()

    def set_tracker_params_handler(self, input_params):
        self.set_parameters_from_vector(input_params.params.data)
        return True

    def get_tracker_params_handler(self, _):
        params = Float64MultiArray()
        params.data = self.get_vector_from_current_parameters()
        return params

    def flipper_enabling_handler(self, input_val):
        self.enable_flippers = input_val.data
        return True, "ok"

    def reset_handler(self, _):
        self.reset()
        return TriggerResponse(
            success=True,
            message="Reseting path_follower"
        )

    def _ros_path_callback(self, data):
        with self.path_lock:
            self.path_data = data
            self.new_path = True

    def _ros_filtered_pc_callback(self, data):
        filtered_pc_array = self.make_array_from_reg_pc_data(data)
        with self.filtered_pc_lock:
            self.filtered_pc_data = data
            self.filtered_pc_array = filtered_pc_array

    def _ros_reg_pc_callback(self, data):
        reg_pc_array = self.make_array_from_reg_pc_data(data)
        with self.reg_pc_lock:
            self.reg_pc_data = data
            self.reg_pc_array = reg_pc_array

    def _ros_trav_vis_pc_callback(self, data):
        trav_vis_pc_array = self.make_array_from_trav_data(data)
        if len(trav_vis_pc_array) == 0: return
        trav_vis_kdtree = KDTree(trav_vis_pc_array[:, :3])

        with self.trav_vis_pc_lock:
            self.trav_vis_pc_data = data
            self.trav_vis_pc_array = trav_vis_pc_array
            self.trav_vis_kdtree = trav_vis_kdtree

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

    def make_array_from_reg_pc_data(self, data):
        pc = ros_numpy.numpify(data).ravel()
        pc = np.stack([pc[f] for f in ['x', 'y', 'z']]).T
        return pc

    def make_array_from_trav_data(self, data):
        pc = ros_numpy.numpify(data).ravel()
        #pc = np.stack([pc[f] for f in ['x', 'y', 'z', 'trav', 'slopex', 'slopey', 'cost']]).T
        pc = np.stack([pc[f] for f in ['x', 'y', 'z']]).T
        n_pts = len(pc)
        if n_pts > self.config["max_pc_points"]:
            decim_coeff = self.config["max_pc_points"] / float(n_pts)
            pc = pc[np.random.rand(n_pts) < decim_coeff, :]
        return pc

    def get_current_target_and_path(self):
        pose_dict = self.get_robot_pose_dict()

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

    def calculate_cmd_vel(self, pose_dict, current_target):
        if current_target is None:
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

        cmd_angular = theta_delta * self.config["turn_vel_sensitivity"]
        cmd_linear = pos_delta * self.config["lin_vel_sensitivity"] \
                     * np.maximum(self.config["turn_vel_thresh"] - abs(theta_delta) * self.config["turn_inhibition_sensitivity"], 0)

        # Clip to max bounds
        cmd_linear_clipped = np.clip(cmd_linear, -self.config["cmd_vel_lin_clip"], self.config["cmd_vel_lin_clip"])
        cmd_angular_clipped = np.clip(cmd_angular, -self.config["cmd_vel_ang_clip"], self.config["cmd_vel_ang_clip"])

        return cmd_linear_clipped, cmd_angular_clipped, pos_delta, theta_delta, changed_dir

    def get_robot_pose_dict(self):
        # Get pose using TF
        try:
            trans = self.tf_buffer.lookup_transform(self.config["root_frame"],
                                                    "X1/base_link", # TODO: Fix this so it works for rev as well
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

    def get_all_observation_dict(self):
        pose_dict = self.get_robot_pose_dict()
        if pose_dict is None: return None

        proprioceptive_data_dict = self.get_proprioceptive_data_dict()
        if proprioceptive_data_dict is None: return None

        exteroceptive_data_dict = self.get_exteroceptive_data_dict(pose_dict)
        if exteroceptive_data_dict is None: return None

        return utilities.merge_dicts([pose_dict, proprioceptive_data_dict, exteroceptive_data_dict])

    def get_proprioceptive_data_dict(self):
        flippers_dict = self.get_flipper_state_dict()
        data_dict = {"flippers_state" : flippers_dict}
        return data_dict

    def get_exteroceptive_data_dict(self, pose_dict):
        # Frame of baselink which has a z of 0 and zero roll and pitch
        #lmap = self.get_filtered_pc(self.current_base_link_frame)
        #lmap = self.get_local_map(self.current_base_link_frame)
        lmap = self.get_local_trav_map(self.current_base_link_frame)

        if lmap is None:
            return None

        # Frontal low features
        frontal_low_pc = self.get_bnd_pts(lmap, self.config["front_low_feat_bnd"])
        frontal_low_feat = self.get_pc_feat(frontal_low_pc)

        # Frontal mid features
        frontal_mid_pc = self.get_bnd_pts(lmap, self.config["front_mid_feat_bnd"])
        frontal_mid_feat = self.get_pc_feat(frontal_mid_pc)

        # rear low features
        rear_low_pc = self.get_bnd_pts(lmap, self.config["rear_low_feat_bnd"])
        rear_low_feat = self.get_pc_feat(rear_low_pc)

        # flipper features
        fl_flipper_pc = self.get_bnd_pts(lmap, self.config["fl_flipper_feat_bnd"])
        fr_flipper_pc = self.get_bnd_pts(lmap, self.config["fr_flipper_feat_bnd"])
        rl_flipper_pc = self.get_bnd_pts(lmap, self.config["rl_flipper_feat_bnd"])
        rr_flipper_pc = self.get_bnd_pts(lmap, self.config["rr_flipper_feat_bnd"])
        fl_flipper_feat = self.get_pc_feat(fl_flipper_pc)
        fr_flipper_feat = self.get_pc_feat(fr_flipper_pc)
        rl_flipper_feat = self.get_pc_feat(rl_flipper_pc)
        rr_flipper_feat = self.get_pc_feat(rr_flipper_pc)

        # Stairs features under robot
        stairs_feats = None # self.get_stairs_feats(lmap, pose_dict)

        data_dict = {"lmap_pc" : lmap,
                     "frontal_low_pc": frontal_low_pc,
                     "frontal_mid_pc": frontal_mid_pc,
                     "rear_low_pc": rear_low_pc,
                     "fl_flipper_pc": fl_flipper_pc,
                     "fr_flipper_pc": fr_flipper_pc,
                     "rl_flipper_pc": rl_flipper_pc,
                     "rr_flipper_pc": rr_flipper_pc,
                     "frontal_low_feat" : frontal_low_feat,
                     "frontal_mid_feat" : frontal_mid_feat,
                     "rear_low_feat" : rear_low_feat,
                     "fl_flipper_feat" : fl_flipper_feat,
                     "fr_flipper_feat" : fr_flipper_feat,
                     "rl_flipper_feat" : rl_flipper_feat,
                     "rr_flipper_feat" : rr_flipper_feat,
                     "stairs_feats" : stairs_feats}

        self.publish_pc_bl(lmap)
        self.publish_pc_bnds(data_dict, self.current_base_link_frame)

        return data_dict

    def get_stairs_feats(self, lmap, pose_dict, from_trav=True):
        position = pose_dict["position"]
        rot_mat = pose_dict["matrix"]

        position_array = np.array([position.x, position.y, position.z])

        # Get the two positions that we will be considering
        x_rob_fr = np.array([0.3, 0, 0])
        x_rob_rr = np.array([-0.3, 0, 0])

        # Two positions in pointcloud
        xyz_fr = position_array + np.matmul(rot_mat, x_rob_fr)
        xyz_rr = position_array + np.matmul(rot_mat, x_rob_rr)

        # Get point clouds
        lmap_filt_fr = self.filter_pc_xyz(lmap, xyz_fr, 0.2)
        lmap_filt_rr = self.filter_pc_xyz(lmap, xyz_rr, 0.2)

        if from_trav:
            slope_xy_array = lmap_filt_fr[:, 4:6]
            mean_slope_vec = np.mean(slope_xy_array, axis=0)

            slope = mean_slope_vec[np.argmax(np.abs(mean_slope_vec))]

            denom = np.sum(np.abs(mean_slope_vec))
            if denom > 0:
                mean_slope_dir = mean_slope_vec / denom
            else:
                mean_slope_dir = [0, 0]
            slant_dir_in_map_frame_fr, slant_dir_in_map_frame_rr = mean_slope_dir
        else:
            slant_dir_in_map_frame_fr = self.estimate_slant(lmap_filt_fr)
            slant_dir_in_map_frame_rr = self.estimate_slant(lmap_filt_rr)

        return slant_dir_in_map_frame_fr, slant_dir_in_map_frame_rr, slope

    def estimate_slant(self, pc):
        pass

    def filter_pc_xyz(self, pc, xyz, bnd):
        pc_filt = pc[np.logical_and(pc[:, 0] < xyz[0] + bnd, pc[:, 0] > xyz[0] - bnd), :]
        pc_filt = pc_filt[np.logical_and(pc_filt[:, 1] < xyz[1] + bnd, pc_filt[:, 1] > xyz[1] - bnd), :]
        pc_filt = pc_filt[np.logical_and(pc_filt[:, 2] < xyz[2] + bnd, pc_filt[:, 2] > xyz[2] - bnd), :]
        return pc_filt

    def get_pc_feat(self, pc):
        def n_to_intensity(n):
            return np.minimum(float(n) / self.config["n_max_intensity"], 1.)

        # Calculate mean height, and intensity (how many points)
        if len(pc) == 0:
            return 0., 0., 0., 0.

        if len(pc) < 3:
            return np.median(pc[:, 2]), np.min(pc[:, 2]), np.max(pc[:, 2]), n_to_intensity(1)

        indeces_pc_sorted_by_z = np.argsort(pc[:, 2])
        min_bnd = np.median(pc[indeces_pc_sorted_by_z[:np.minimum(10, len(pc))], 2]) - self.config["pc_z_median_offset"]
        max_bnd = np.median(pc[indeces_pc_sorted_by_z[-np.minimum(10, len(pc)):], 2]) - self.config["pc_z_median_offset"]

        avg_height = np.median(pc[:, 2]) - self.config["pc_z_median_offset"]
        intensity = n_to_intensity(len(pc))

        return avg_height, min_bnd, max_bnd, intensity

    def get_flipper_state_dict(self):
        flipper_name_dict = ["front_left_flipper",
                             "front_right_flipper",
                             "rear_left_flipper",
                             "rear_right_flipper"]

        flipper_name_dict_rev = ["rear_right_flipper",
                                 "rear_left_flipper",
                                 "front_right_flipper",
                                 "front_left_flipper"]

        flipper_dict = {}
        for fn, fn_rev in zip(flipper_name_dict, flipper_name_dict_rev):
            if self.current_base_link_frame == self.config["robot_prefix"] + "base_link_zrp":
                fn_c = fn
            else:
                fn_c = fn_rev

            try:
                trans = self.tf_buffer.lookup_transform(self.current_base_link_frame,
                                                        self.config["robot_prefix"] + fn_c,
                                                        rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
                rospy.logwarn("Get_flipper_state_dict: TRANSFORMATION FAILED, err: {}".format(err))
                return None

            # Orientation
            quat = trans.transform.rotation
            mat = quaternion_matrix((quat.x, quat.y, quat.z, quat.w))[:3,:3]

            if fn == "front_right_flipper":
                angle = np.arctan2(-mat[0,2], -mat[0,0])

            if fn == "front_left_flipper":
                angle = np.arctan2(mat[0,2], mat[0,0])

            if fn == "rear_right_flipper":
                angle = np.arctan2(mat[0,2], mat[0,0])

            if fn == "rear_left_flipper":
                angle = np.arctan2(-mat[0,2], -mat[0,0])

            flipper_dict[fn] = angle
        return flipper_dict

    def get_bnd_pts(self, lmap, bnds):
        pts = lmap[np.logical_and(lmap[:, 0] > bnds[0], lmap[:, 0] < bnds[1]), :] # x
        pts = pts[np.logical_and(pts[:, 1] > bnds[2], pts[:, 1] < bnds[3]), :] # y
        pts = pts[np.logical_and(pts[:, 2] > bnds[4], pts[:, 2] < bnds[5]), :] # z

        # If more than max points, decimate
        if len(pts) > self.config["max_bnd_pc_count"]:
           pts = pts[np.random.choice(np.arange(len(pts)), self.config["max_bnd_pc_count"], replace=False), :]

        return pts

    def get_filtered_pc(self, frame):
        # Get local point cloud
        with self.filtered_pc_lock:
            if self.filtered_pc_lock is None:
                return None
            pc_arr_bl = self.localize_and_cut_pc(self.filtered_pc_array, frame, self.filtered_pc_data, 2.)
        return pc_arr_bl

    def get_local_map(self, frame):
        # Get local point cloud
        with self.reg_pc_lock:
            if self.reg_pc_array is None:
                return None
            pc_arr_bl = self.localize_and_cut_pc(self.reg_pc_array, frame, self.reg_pc_data, 2.)
        return pc_arr_bl

    def get_local_trav_map(self, frame):
        # Get local point cloud
        with self.trav_vis_pc_lock:
            if self.trav_vis_pc_array is None:
                return None
            pc_arr_bl = self.localize_and_cut_pc(self.trav_vis_pc_array, frame, self.trav_vis_pc_data, 2.)
        return pc_arr_bl

    def localize_and_cut_pc(self, pc_array, frame_name, msg, bnd):
        # Filter pc which is of a radius greater than pc_cutoff_radius
        pose_dict = self.get_robot_pose_dict()
        if pose_dict is None or pc_array is None or len(pc_array) == 0: return None

        #robot_pos = pose_dict["position"]
        #(x, y, z) = 0, 0, 0  #robot_pos.x, robot_pos.y, robot_pos.z
        #pc_arr_filt = np.copy(pc_array)
        #pc_arr_filt = pc_arr_filt[np.logical_and(pc_arr_filt[:, 0] > x - bnd, pc_arr_filt[:, 0] < x + bnd), :]
        #pc_arr_filt = pc_arr_filt[np.logical_and(pc_arr_filt[:, 1] > y - bnd, pc_arr_filt[:, 1] < y + bnd), :]
        #pc_arr_filt = pc_arr_filt[np.logical_and(pc_arr_filt[:, 2] > z - 1.0, pc_arr_filt[:, 2] < z + 1.0), :]
        #pc_arr_filt = pc_arr_filt[~np.isnan(pc_arr_filt).any(axis=1), :]

        source_frame = msg.header.frame_id
        if msg.header.frame_id[0] == '/':
            source_frame = msg.header.frame_id[1:]

        # Array of size (4, N_points). 4th coordinate is for homogeneous transformation
        pc_arr_filt_homog = np.append(pc_array, np.ones((len(pc_array), 1)), axis=1)

        # Transform pc to baselink coordinates
        try:
            tf_bl = self.tf_buffer.lookup_transform_full(
                target_frame=frame_name,
                target_time=rospy.Time(0),
                source_frame=source_frame,
                source_time=msg.header.stamp,
                fixed_frame=self.config["root_frame"],
                timeout=rospy.Duration(0)
            )
            tf_bl = ros_numpy.numpify(tf_bl.transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn("Get_local_map with frame {}: TRANSFORMATION FAILED, err: {}".format(frame_name, err))
            return None

        # Get the zero-roll-pitch transform
        pc_arr_bl = np.matmul(tf_bl, pc_arr_filt_homog.T).T[:, :3]

        return pc_arr_bl

    def decide_next_state_handcrafted(self, obs_dict, current_target, current_path):
        # Intrinsics
        pitch = obs_dict["euler"][1]
        flat_pitch_dev = abs(pitch) < 0.2
        small_pitch = pitch < -0.2
        large_pitch = pitch < -0.4
        small_dip = pitch > 0.2
        large_dip = pitch > 0.4

        # Extrinsics general
        flat_ground = obs_dict["frontal_low_feat"][1] > -0.04 and obs_dict["frontal_low_feat"][2] < 0.04 and \
                      obs_dict["rear_low_feat"][1] > -0.04 and obs_dict["rear_low_feat"][2] < 0.04

        # Extrinsics front
        untraversable_elevation = obs_dict["frontal_mid_feat"][3] > 0.1
        small_frontal_elevation = obs_dict["frontal_low_feat"][2] > 0.06
        large_frontal_elevation = obs_dict["frontal_low_feat"][2] > 0.12
        small_frontal_lowering = obs_dict["frontal_low_feat"][1] < -0.06
        large_frontal_lowering = obs_dict["frontal_low_feat"][1] < -0.12
        low_frontal_point_presence = obs_dict["frontal_low_feat"][3] < 0.15

        # Extrinsics rear
        small_rear_elevation = obs_dict["rear_low_feat"][2] > 0.06
        large_rear_elevation = obs_dict["rear_low_feat"][2] > 0.12
        small_rear_lowering = obs_dict["rear_low_feat"][1] < -0.06
        large_rear_lowering = obs_dict["rear_low_feat"][1] < -0.12
        not_rear_lowering = obs_dict["rear_low_feat"][2] > -0.03
        low_rear_point_presence = obs_dict["rear_low_feat"][3] < 0.5
        low_rear_point_presence = obs_dict["rear_low_feat"][3] < 0.5

        prev_state = self.current_state

        if self.current_state == "NEUTRAL":
            if (small_frontal_elevation) and not untraversable_elevation:
                 self.current_state = "ASCENDING_FRONT"
            elif low_frontal_point_presence or small_frontal_lowering:
                self.current_state = "DESCENDING_FRONT"
            elif small_pitch and not small_frontal_elevation:
                self.current_state = "ASCENDING_REAR"
        elif self.current_state == "ASCENDING_FRONT":
            # -> ascending_rear
            if small_pitch and not large_frontal_elevation:
                self.current_state = "ASCENDING_REAR"
            # -> up_stairs
            elif large_pitch and small_frontal_elevation:
                self.current_state = "UP_STAIRS"
            # -> neutral
            elif flat_ground and flat_pitch_dev:
                self.current_state = "NEUTRAL"
        elif self.current_state == "ASCENDING_REAR":
            if not_rear_lowering or flat_ground:
                self.current_state = "NEUTRAL"
        elif self.current_state == "DESCENDING_FRONT":
            # -> descending_rear
            if small_rear_lowering or low_rear_point_presence:
                self.current_state = "DESCENDING_REAR"
            # -> down_stairs
            elif (large_frontal_lowering or low_frontal_point_presence) and large_dip:
                self.current_state = "DOWN_STAIRS"
            # -> neutral
            elif flat_ground and flat_pitch_dev:
                self.current_state = "NEUTRAL"
        elif self.current_state == "DESCENDING_REAR":
            # -> neutral
            if flat_pitch_dev or not small_frontal_lowering:
                self.current_state = "NEUTRAL"
        elif self.current_state == "UP_STAIRS":
            # -> ascending_rear
            if low_frontal_point_presence or not large_frontal_elevation:
                self.current_state = "ASCENDING_REAR"
            elif flat_ground and flat_pitch_dev:
                self.current_state = "NEUTRAL"
        elif self.current_state == "DOWN_STAIRS":
            # -> descending_rear
            if not large_frontal_lowering or flat_ground:
                self.current_state = "DESCENDING_REAR"
            elif flat_ground and flat_pitch_dev:
                self.current_state = "NEUTRAL"
        else:
            raise NotImplementedError

        return self.current_state != prev_state

    def decide_next_state_learned(self, obs_dict, current_target, current_path):
        # Intrinsics
        pitch = obs_dict["euler"][1]
        flat_pitch_dev = abs(pitch) < self.learnable_param_dict["vector"][0] # 0.2
        small_pitch = pitch < -self.learnable_param_dict["vector"][1] # -0.2
        large_pitch = pitch < -self.learnable_param_dict["vector"][2] # -0.4
        small_dip = pitch > self.learnable_param_dict["vector"][3] # 0.2
        large_dip = pitch > self.learnable_param_dict["vector"][4] # 0.4

        # Extrinsics general
        flat_ground = obs_dict["frontal_low_feat"][1] > -self.learnable_param_dict["vector"][5] and obs_dict["frontal_low_feat"][2] < self.learnable_param_dict["vector"][5] and \
                      obs_dict["rear_low_feat"][1] > -self.learnable_param_dict["vector"][5] and obs_dict["rear_low_feat"][2] < self.learnable_param_dict["vector"][5]

        # Extrinsics front
        untraversable_elevation = obs_dict["frontal_mid_feat"][3] > self.learnable_param_dict["vector"][6] #0.1
        small_frontal_elevation = obs_dict["frontal_low_feat"][2] > self.learnable_param_dict["vector"][7] #0.06
        large_frontal_elevation = obs_dict["frontal_low_feat"][2] > self.learnable_param_dict["vector"][8] #0.12
        small_frontal_lowering = obs_dict["frontal_low_feat"][1] < -self.learnable_param_dict["vector"][9] #-0.06
        large_frontal_lowering = obs_dict["frontal_low_feat"][1] < -self.learnable_param_dict["vector"][10] #-0.12
        low_frontal_point_presence = obs_dict["frontal_low_feat"][3] < self.learnable_param_dict["vector"][11] #0.15

        # Extrinsics rear
        small_rear_elevation = obs_dict["rear_low_feat"][2] > self.learnable_param_dict["vector"][12] #0.06
        large_rear_elevation = obs_dict["rear_low_feat"][2] > self.learnable_param_dict["vector"][13] #0.12
        small_rear_lowering = obs_dict["rear_low_feat"][1] < -self.learnable_param_dict["vector"][14] #-0.06
        large_rear_lowering = obs_dict["rear_low_feat"][1] < -self.learnable_param_dict["vector"][15] #-0.12
        not_rear_lowering = obs_dict["rear_low_feat"][2] > -self.learnable_param_dict["vector"][16]
        low_rear_point_presence = obs_dict["rear_low_feat"][3] < self.learnable_param_dict["vector"][17] #0.5

        prev_state = self.current_state

        if self.current_state == "NEUTRAL":
            if (small_frontal_elevation) and not untraversable_elevation:
                self.current_state = "ASCENDING_FRONT"
            elif low_frontal_point_presence or small_frontal_lowering:
                self.current_state = "DESCENDING_FRONT"
            elif small_pitch and not small_frontal_elevation:
                self.current_state = "ASCENDING_REAR"
        elif self.current_state == "ASCENDING_FRONT":
            # -> ascending_rear
            if small_pitch and not large_frontal_elevation:
                self.current_state = "ASCENDING_REAR"
            # -> up_stairs
            elif large_pitch and small_frontal_elevation:
                self.current_state = "UP_STAIRS"
            # -> neutral
            elif flat_ground and flat_pitch_dev:
                self.current_state = "NEUTRAL"
        elif self.current_state == "ASCENDING_REAR":
            if not_rear_lowering or flat_ground:
                self.current_state = "NEUTRAL"
        elif self.current_state == "DESCENDING_FRONT":
            # -> descending_rear
            if small_rear_lowering or low_rear_point_presence:
                self.current_state = "DESCENDING_REAR"
            # -> down_stairs
            elif (large_frontal_lowering or low_frontal_point_presence) and large_dip:
                self.current_state = "DOWN_STAIRS"
            # -> neutral
            elif flat_ground and flat_pitch_dev:
                self.current_state = "NEUTRAL"
        elif self.current_state == "DESCENDING_REAR":
            # -> neutral
            if flat_pitch_dev or not small_frontal_lowering:
                self.current_state = "NEUTRAL"
        elif self.current_state == "UP_STAIRS":
            # -> ascending_rear
            if low_frontal_point_presence or not large_frontal_elevation:
                self.current_state = "ASCENDING_REAR"
            elif flat_ground and flat_pitch_dev:
                self.current_state = "NEUTRAL"
        elif self.current_state == "DOWN_STAIRS":
            # -> descending_rear
            if not large_frontal_lowering or flat_ground:
                self.current_state = "DESCENDING_REAR"
            elif flat_ground and flat_pitch_dev:
                self.current_state = "NEUTRAL"
        else:
            raise NotImplementedError

        return self.current_state != prev_state

    def swap_state_dirs(self):
        if self.current_state == "ASCENDING_FRONT":
            self.current_state = "DESCENDING_REAR"
        elif self.current_state == "DESCENDING_REAR":
            self.current_state = "ASCENDING_FRONT"

        elif self.current_state == "ASCENDING_REAR":
            self.current_state = "DESCENDING_FRONT"
        elif self.current_state == "DESCENDING_FRONT":
            self.current_state = "ASCENDING_REAR"

        elif self.current_state == "DESCENDING_FRONT":
            self.current_state = "ASCENDING_REAR"
        elif self.current_state == "ASCENDING_REAR":
            self.current_state = "DESCENDING_FRONT"

        elif self.current_state == "DESCENDING_REAR":
            self.current_state = "ASCENDING_FRONT"
        elif self.current_state == "ASCENDING_FRONT":
            self.current_state = "DESCENDING_REAR"

        elif self.current_state == "DOWN_STAIRS":
            self.current_state = "UP_STAIRS"
        elif self.current_state == "UP_STAIRS":
            self.current_state = "DOWN_STAIRS"

    def calculate_flipper_action_handcrafted(self, obs_dict, current_target, current_path, theta_delta):
        roll, pitch, yaw = obs_dict["euler"]

        fl_flipper_stab_correction = 0
        fr_flipper_stab_correction = 0
        rl_flipper_stab_correction = 0
        rr_flipper_stab_correction = 0

        if self.config["enable_flipper_stabilization"]:
            if self.current_state == "ASCENDING_FRONT":
                fl_flipper_stab_correction = -roll * self.config["roll_stabilization_coeff"]
                fr_flipper_stab_correction = roll * self.config["roll_stabilization_coeff"]

            if self.current_state == "ASCENDING_REAR":
                #rl_flipper_stab_correction = roll * self.config["roll_stabilization_coeff"]
                #rr_flipper_stab_correction = -roll * self.config["roll_stabilization_coeff"]
                rear_flipper_stab_correction = pitch * 1
                rl_flipper_stab_correction = rr_flipper_stab_correction = rear_flipper_stab_correction

            #if self.current_state == "DESCENGING_REAR":
            #    rl_flipper_stab_correction = roll * self.config["roll_stabilization_coeff"]
            #    rr_flipper_stab_correction = -roll * self.config["roll_stabilization_coeff"]

        flipper_state = self.current_state
        # If turning and not critical state, lift up flippers
        if abs(theta_delta) > self.config["theta_delta_thresh_neutral"] and self.current_state == "ASCENDING_FRONT":
            flipper_state = "NEUTRAL"

        flipper_commands_dict = {}
        flipper_commands_dict["front_left"] = self.config["FLIPPERS_{}".format(flipper_state)][0] + fl_flipper_stab_correction
        flipper_commands_dict["front_right"] = self.config["FLIPPERS_{}".format(flipper_state)][1] + fr_flipper_stab_correction
        flipper_commands_dict["rear_left"] = self.config["FLIPPERS_{}".format(flipper_state)][2] + rl_flipper_stab_correction
        flipper_commands_dict["rear_right"] = self.config["FLIPPERS_{}".format(flipper_state)][3] + rr_flipper_stab_correction

        flipper_torques_dict = {}
        flipper_torques_dict["front_left"] = self.config["FLIPPERS_CURRENT_{}".format(flipper_state)][0]
        flipper_torques_dict["front_right"] = self.config["FLIPPERS_CURRENT_{}".format(flipper_state)][0]
        flipper_torques_dict["rear_left"] = self.config["FLIPPERS_CURRENT_{}".format(flipper_state)][1]
        flipper_torques_dict["rear_right"] = self.config["FLIPPERS_CURRENT_{}".format(flipper_state)][1]

        return flipper_commands_dict, flipper_torques_dict

    def calculate_flipper_action_learned(self, obs_dict, current_target, current_path, theta_delta):
        flipper_commands_dict = {}

        index_offset = 12
        p1 = index_offset + self.learnable_param_dict["vector"][self.all_states_list.index(self.current_state)]
        p2 = index_offset + self.learnable_param_dict["vector"][self.all_states_list.index(self.current_state) + len(self.all_states_list)]

        flipper_commands_dict["front_left"] = self.config["FLIPPERS_{}".format(self.current_state)][0] + p1
        flipper_commands_dict["front_right"] = self.config["FLIPPERS_{}".format(self.current_state)][1] + p1
        flipper_commands_dict["rear_left"] = self.config["FLIPPERS_{}".format(self.current_state)][2] + p2
        flipper_commands_dict["rear_right"] = self.config["FLIPPERS_{}".format(self.current_state)][3] + p2

        flipper_torques_dict = {}

        flipper_torques_dict["front_left"] = self.config["FLIPPERS_CURRENT_{}".format(self.current_state)][0]
        flipper_torques_dict["front_right"] = self.config["FLIPPERS_CURRENT_{}".format(self.current_state)][0]
        flipper_torques_dict["rear_left"] = self.config["FLIPPERS_CURRENT_{}".format(self.current_state)][1]
        flipper_torques_dict["rear_right"] = self.config["FLIPPERS_CURRENT_{}".format(self.current_state)][1]

        return flipper_commands_dict, flipper_torques_dict

    def calculate_flipper_action_nn(self, obs_dict, current_target, current_path, theta_delta):
        feat_vec = self.get_pc_feat_vec(obs_dict)
        feat_tensor = T.tensor(feat_vec).unsqueeze(0)
        act_A, act_B = self.nn_policy(feat_tensor)

        act = act_A[0].detach().numpy()
        if time.time() % 10 < 5:
            act = act_B[0].detach().numpy()

        flipper_commands_dict = {}
        flipper_commands_dict["front_left"] = act[0]
        flipper_commands_dict["front_right"] = act[1]
        flipper_commands_dict["rear_left"] = act[2]
        flipper_commands_dict["rear_right"] = act[3]

        flipper_torques_dict = {}
        flipper_torques_dict["front_left"] = 60
        flipper_torques_dict["front_right"] = 60
        flipper_torques_dict["rear_left"] = 60
        flipper_torques_dict["rear_right"] = 60

        return flipper_commands_dict, flipper_torques_dict

    def calculate_cmd_vel_state_scalar(self, obs_dict, current_target, current_path):
        if self.current_state == "NEUTRAL":
            # No-change
            cmd_vel_scalar = 1
        elif self.current_state == "ROUGH_TERRAIN":
            cmd_vel_scalar = self.config["rough_terrain_vel_factor"]
        elif self.current_state == "ASCENDING_FRONT":
            cmd_vel_scalar = self.config["ascending_vel_factor"]
        elif self.current_state == "ASCENDING_REAR":
            cmd_vel_scalar = self.config["ascending_vel_factor"]
        elif self.current_state == "DESCENDING_FRONT":
            cmd_vel_scalar = self.config["descending_vel_factor"]
        elif self.current_state == "DESCENDING_REAR":
            cmd_vel_scalar = self.config["descending_vel_factor"]
        elif self.current_state == "UP_STAIRS":
            cmd_vel_scalar = self.config["stairs_vel_factor"]
        elif self.current_state == "DOWN_STAIRS":
            cmd_vel_scalar = self.config["stairs_vel_factor"]
        else:
            raise NotImplementedError

        return cmd_vel_scalar

    def set_parameters_from_vector(self, w):
        p_dict = {}
        p_dict["vector"] = w[:]

        self.learnable_param_dict = p_dict

    def get_vector_from_current_parameters(self):
        w = []
        w.extend(self.learnable_param_dict["vector"])
        return w

    def get_pc_feat_vec(self, obs_dict):
        vec = obs_dict["fl_flipper_feat"] + \
            obs_dict["fr_flipper_feat"] + \
            obs_dict["rl_flipper_feat"] + \
            obs_dict["rr_flipper_feat"] + \
            obs_dict["euler"] + \
            obs_dict["front_left_flipper"] + \
            obs_dict["front_right_flipper"] + \
            obs_dict["rear_left_flipper"] + \
            obs_dict["rear_right_flipper"]
        return vec

    def publish_track_vel(self, linear, angular):
        if self.config["enable_cmd_vel"]:
            msg = Twist()

            if self.current_base_link_frame == self.config["robot_prefix"] + "base_link_zrp":
                msg.linear.x = linear
            else:
                msg.linear.x = -linear

            msg.linear.y = 0
            msg.angular.z = angular
            self.tracks_vel_publisher.publish(msg)

    def publish_flipper_pos(self, flipper_dict):
        if not self.enable_flippers:
            return

        # Publish flippers vel
        flippers_pos_msg = MarvFloat64MultiArray()
        if self.current_base_link_frame == self.config["robot_prefix"] + "base_link_zrp":
            flippers_pos_msg.data = [flipper_dict["front_left"], flipper_dict["front_right"],
                                     flipper_dict["rear_left"], flipper_dict["rear_right"]]
        else:
            flippers_pos_msg.data = [flipper_dict["rear_right"], flipper_dict["rear_left"],
                                     flipper_dict["front_right"],flipper_dict["front_left"]]
        self.flippers_pos_publisher.publish(flippers_pos_msg)

    def publish_flipper_torque_limits(self, flipper_dict):
        if not self.enable_flippers:
            return

        # Publish flippers vel
        flippers_torque_msg = MarvFloat64MultiArray()
        if self.current_base_link_frame == self.config["robot_prefix"] + "base_link_zrp":
            flippers_torque_msg.data = [flipper_dict["front_left"], flipper_dict["front_right"],
                                     flipper_dict["rear_left"], flipper_dict["rear_right"]]
        else:
            flippers_torque_msg.data = [flipper_dict["rear_right"], flipper_dict["rear_left"],
                                     flipper_dict["front_right"], flipper_dict["front_left"]]
        self.flippers_max_torque_publisher.publish(flippers_torque_msg)

    def publish_pc_bnds(self, pc_bnds_dict, frame):
        pc_bnds = np.concatenate((pc_bnds_dict["frontal_low_pc"],
                                  pc_bnds_dict["rear_low_pc"],
                                  pc_bnds_dict["fl_flipper_pc"],
                                  pc_bnds_dict["fr_flipper_pc"],
                                  pc_bnds_dict["rl_flipper_pc"],
                                  pc_bnds_dict["rr_flipper_pc"]
                                  ), axis=0)

        pc_bnds_data = np.zeros(len(pc_bnds), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32)
        ])

        pc_bnds_data['x'] = pc_bnds[:, 0]
        pc_bnds_data['y'] = pc_bnds[:, 1]
        pc_bnds_data['z'] = pc_bnds[:, 2]

        msg = ros_numpy.msgify(PointCloud2, pc_bnds_data)
        msg.header.frame_id = frame
        msg.header.stamp = rospy.Time.now()

        self.pc_bnds_publisher.publish(msg)

    def publish_pc_bl(self, pc_bl):
        pc_bl_zpr_data = np.zeros(len(pc_bl), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('vectors', np.float32, (3,))
        ])

        pc_bl_zpr_data['x'] = pc_bl[:, 0]
        pc_bl_zpr_data['y'] = pc_bl[:, 1]
        pc_bl_zpr_data['z'] = pc_bl[:, 2]
        pc_bl_zpr_data['vectors'] = np.arange(len(pc_bl))[:, np.newaxis]

        msg = ros_numpy.msgify(PointCloud2, pc_bl_zpr_data)
        msg.header.frame_id = self.current_base_link_frame
        msg.header.stamp = rospy.Time.now()

        self.pc_bl_zpr_publisher.publish(msg)

    def publish_current_target(self):
        if self.current_target is None: return
        msg = PointStamped()
        msg.point.x = self.current_target.pose.position.x
        msg.point.y = self.current_target.pose.position.y
        msg.point.z = self.current_target.pose.position.z
        msg.header.frame_id = self.config["root_frame"]
        self.current_target_publisher.publish(msg)

    def publish_txt_info(self, obs_dict):
        strs = []
        strs.append("State: {} \n".format(self.current_state))
        strs.append("Pitch: {} \n".format(round(obs_dict["euler"][1], 4)))
        strs.append("frontal_low_feat: {} \n".format([round(o,4) for o in obs_dict["frontal_low_feat"]]))
        strs.append("frontal_mid_feat: {} \n".format([round(o,4) for o in obs_dict["frontal_mid_feat"]]))
        strs.append("rear_low_feat: {} \n".format([round(o,4) for o in obs_dict["rear_low_feat"]]))
        strs.append("fl_flipper_feat: {} \n".format([round(o,4) for o in obs_dict["fl_flipper_feat"]]))
        strs.append("fr_flipper_feat: {} \n".format([round(o,4) for o in obs_dict["fr_flipper_feat"]]))
        strs.append("rl_flipper_feat: {} \n".format([round(o,4) for o in obs_dict["rl_flipper_feat"]]))
        strs.append("rr_flipper_feat: {} \n".format([round(o,4) for o in obs_dict["rr_flipper_feat"]]))

        txt = ""
        for s in strs:
            txt += s

        msg = String()
        msg.data = txt
        self.text_publisher.publish(msg)

    def publish_pc_feat_vect(self, obs_dict):
        msg = Float64MultiArray()
        msg.data = self.get_pc_feat_vec(obs_dict)
        self.pc_bnds_publisher.publish(msg)

    def publish_tracker_state(self):
        msg = String()
        msg.data = self.current_state
        self.tracker_state_publisher.publish(msg)

    def publish_all_bbx(self):
        self.publish_bbx(self.config["front_mid_feat_bnd"], 0, [0, 0, 1])
        self.publish_bbx(self.config["front_low_feat_bnd"], 1, [0, 1, 1])
        self.publish_bbx(self.config["fl_flipper_feat_bnd"], 2, [0.6, 0.3, 0])
        self.publish_bbx(self.config["fr_flipper_feat_bnd"], 3, [0.5, 0.4, 0])
        self.publish_bbx(self.config["rl_flipper_feat_bnd"], 4, [0.4, 0.5, 0])
        self.publish_bbx(self.config["rr_flipper_feat_bnd"], 5, [0.3, 0.6, 0])

    def publish_bbx(self, bnds, id, rgb):
        xl, xu, yl, yu, zl, zu = bnds

        marker_msg = Marker()
        marker_msg.header.frame_id = self.config["robot_prefix"] + "base_link_zrp"
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
        marker_msg.color.a = 0.45

        marker_msg.color.r = rgb[0]
        marker_msg.color.g = rgb[1]
        marker_msg.color.b = rgb[2]

        self.marker_publisher.publish(marker_msg)

    def save(self, name):
        if not os.path.exists("agents"):
            os.makedirs("agents")
        if name is None:
            name = "agents/saved_params.pickle"
        pickle.dump(self.learnable_param_dict, open(name, "wb"))

    def load(self, name):
        self.learnable_param_dict = pickle.load(open(name, "rb"))

    def step(self):
        # Find nearest point in path
        current_target, current_path = self.get_current_target_and_path()

        # Get required observations
        obs_dict = self.get_all_observation_dict()

        if obs_dict is None:
            rospy.loginfo_throttle(1, "Obs_dict was none, skipping iteration")
            try:
                self.ros_rate.sleep()
            except rospy.ROSInterruptException:
                pass
            return None

        changed_state = False
        # Decide on state change if sufficient time has passed
        if time.time() - self.last_state_change_time > 1.2:
            if self.config["mode"] == "handcrafted":
                changed_state = self.decide_next_state_handcrafted(obs_dict, current_target, current_path)
            elif self.config["mode"] == "learned":
                changed_state = self.decide_next_state_learned(obs_dict, current_target, current_path)

        if changed_state:
            rospy.loginfo("Current state: {}".format(self.current_state))

        if changed_state:
            self.last_state_change_time = time.time()

        cmd_vel_transition_scalar = 1.0
        if time.time() - self.last_state_change_time < 0.5:
            cmd_vel_transition_scalar = 1.0 # This could be less

        # Burn in time
        if time.time() - self.reset_time < self.config["tracker_burn_in_time"]:
            self.current_state = "NEUTRAL"

        # Calculate turn deltas
        cmd_linear, cmd_angular, pos_delta, theta_delta, changed_dir = self.calculate_cmd_vel(obs_dict, current_target)

        # if changed_dir:
        #     self.swap_state_dirs()
        #     try:
        #         self.ros_rate.sleep()
        #     except rospy.ROSInterruptException:
        #         pass
        #     return

        # Decide on flipper control and cmd_vel
        if self.config["mode"] == "handcrafted":
            flipper_commands_dict, flipper_torques_dict = self.calculate_flipper_action_handcrafted(obs_dict, current_target, current_path, theta_delta)
        elif self.config["mode"] == "learned":
            #flipper_commands_dict, flipper_torques_dict = self.calculate_flipper_action_learned(obs_dict, current_target, current_path, theta_delta)
            flipper_commands_dict, flipper_torques_dict = self.calculate_flipper_action_handcrafted(obs_dict,
                                                                                                    current_target,
                                                                                                    current_path,
                                                                                                    theta_delta)
        elif self.config["mode"] == "nn":
            flipper_commands_dict, flipper_torques_dict = self.calculate_flipper_action_nn(obs_dict, current_target, current_path, theta_delta)
        else:
            raise NotImplementedError

        # Check if we have reached target
        if current_target is not None and np.abs(pos_delta) < self.config["waypoint_reach_distance"]:
            if self.current_path.poses is not None and len(self.current_path.poses) == 0:
                self.current_target = None
                cmd_linear, cmd_angular = 0, 0
                self.linear_tracking_momentum = 0
                #rospy.loginfo("Reached terminal waypoint.")
            elif self.current_path.poses is not None:
                self.set_current_target(current_path.poses[0])
                self.delete_pose_waypoint(0)

        # Step the linear momentum variables
        self.update_linear_momentum(cmd_linear)

        # Scale velocity by state scalar
        state_scalar = self.calculate_cmd_vel_state_scalar(obs_dict, current_target, current_path)
        #cmd_linear_state_scaled = self.linear_tracking_momentum * state_scalar * cmd_vel_transition_scalar
        cmd_linear_state_scaled = cmd_linear * state_scalar * cmd_vel_transition_scalar

        self.publish_track_vel(cmd_linear_state_scaled, cmd_angular)
        self.publish_flipper_pos(flipper_commands_dict)
        self.publish_flipper_torque_limits(flipper_torques_dict)
        self.publish_current_target()
        self.publish_txt_info(obs_dict)
        self.publish_tracker_state()
        self.publish_all_bbx()

        obs_dict["cmd_linear_state_scaled"] = cmd_linear_state_scaled
        obs_dict["cmd_angular"] = cmd_angular
        obs_dict["current_target"] = current_target
        obs_dict["current_path"] = current_path

        try:
            self.ros_rate.sleep()
        except rospy.ROSInterruptException:
            pass

        return obs_dict

    def reset(self):
        self.current_state = "NEUTRAL"
        self.current_tracking_orientation = "fw"
        self.reset_time = time.time()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.config["tf_buffer_size"]))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def start_tracking(self):
        rospy.loginfo("{} starting tracking...".format(self.config["node_name"]))

        while not rospy.is_shutdown():
            self.step()

    def record_tracking(self, n_secs):
        rospy.loginfo("{} starting path_follower recording for dataset creation purposes...".format(self.config["node_name"]))

        start_time = time.time()

        data_dict_list = []
        while not rospy.is_shutdown():
            obs_dict = self.step()

            if obs_dict is not None:
                obs_dict["quat"] = (obs_dict["quat"].w, obs_dict["quat"].x, obs_dict["quat"].y, obs_dict["quat"].z)
                obs_dict["position"] = (obs_dict["position"].x, obs_dict["position"].y, obs_dict["position"].z)
                obs_dict["pc_feat_vec"] = self.get_pc_feat_vec(obs_dict)
                data_dict_list.append(obs_dict)

            if time.time() - start_time > n_secs:
                break

        rospy.loginfo("Saving dataset")

        # Save dataset
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/supervised_dataset")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Find last indexed dataset
        for i in range(100):
            file_path = os.path.join(dataset_dir, "dataset_{}.pkl".format(i))
            if not os.path.exists(file_path):
                file_path = os.path.join(dataset_dir, "dataset_{}.pkl".format(i))
                pickle.dump(data_dict_list, open(file_path, "wb"))
                break

        rospy.loginfo("Dataset recording done")

def main():
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) == 2:
        config_name = myargv[1]
    else:
        config_name = "marv_tracker_config.yaml"

    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/{}".format(config_name)), 'r') as f:
        tracker_config = yaml.load(f, Loader=yaml.FullLoader)

    tracker = MarvTracker(tracker_config)
    tracker.start_tracking()
    #path_follower.record_tracking(300)

if __name__=="__main__":
    main()


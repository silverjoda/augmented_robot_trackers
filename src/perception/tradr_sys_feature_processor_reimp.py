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

from src.utilities import utilities, ros_utilities
from copy import deepcopy

class TradrFeatureProcessor:
    def __init__(self, config):
        self.config = config

        self.state_list = ["NEUTRAL",
                           "ASCENDING_FRONT",
                           "UP_STAIRS",
                           "ASCENDING_REAR",
                           "DESCENDING_FRONT",
                           "DOWN_STAIRS",
                           "DESCENDING_REAR"]

        self.state_to_short_name_dict = {"NEUTRAL": "N",
                                         "ASCENDING_FRONT": "AF",
                                         "UP_STAIRS": "US",
                                         "ASCENDING_REAR": "AR",
                                         "DESCENDING_FRONT": "DF",
                                         "DOWN_STAIRS": "DS",
                                         "DESCENDING_REAR": "DR"}

        self.short_to_state_name_dict = {v: k for k, v in self.state_to_short_name_dict.items()}

        self.current_state = "NEUTRAL"

        self.init_ros()


    def init_ros(self):
        rospy.init_node("tradr_feature_processor")
        self.ros_rate = rospy.Rate(self.config["ros_rate"])
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.trav_vis_pc_data = None
        self.trav_vis_pc_array = None
        self.trav_vis_pc_lock = threading.Lock()

        self.current_base_link_frame = self.config["robot_prefix"] + "base_link_zrp"


        self.text_publisher = rospy.Publisher("art/debug/text_info",
                                              String,
                                              queue_size=1)

        self.pc_bnds_publisher = rospy.Publisher("art/debug/pc_bnds",
                                                 PointCloud2,
                                                 queue_size=1)

        self.marker_publisher = rospy.Publisher("art/feat_bbx_out",
                                                      Marker,
                                                      queue_size=1)

        self.pc_feat_vec_publisher = rospy.Publisher("art/pc_feat_vec",
                                                Float64MultiArray,
                                                queue_size=1)

        self.pc_feat_msg_publisher = rospy.Publisher("art/pc_feat_msg",
                                                     MarvPCFeats,
                                                     queue_size=1)

        self.flipper_position_publisher = rospy.Publisher("art/flipper_positions",
                                                     Float64MultiArray,
                                                     queue_size=1)

        self.stagnation_publisher = rospy.Publisher("art/tradr_progress_stagnation",
                                                    Float64,
                                                    queue_size=1)

        self.haar_feat_publisher = rospy.Publisher("art/haar_feats",
                                                          Float64MultiArray,
                                                          queue_size=1)

        rospy.Subscriber("rds/traversability_visual",
                         PointCloud2,
                         self._ros_trav_vis_pc_callback, queue_size=1)

        self.teleop_state_subscriber = ros_utilities.subscriber_factory("teleop/state", String)
        self.controller_state_subscriber = ros_utilities.subscriber_factory("art/tradr_flipper_controller_state", String)
        self.track_vel_subscriber = ros_utilities.subscriber_factory("cmd_vel", Twist)
        self.dsm_distrib_subscriber = ros_utilities.subscriber_factory("art/dsm_distrib", Float64MultiArray)

    def _ros_trav_vis_pc_callback(self, msg):
        trav_vis_pc_array = self.make_array_from_trav_data(msg)
        if len(trav_vis_pc_array) == 0: return

        with self.trav_vis_pc_lock:
            self.trav_vis_pc_data = msg
            self.trav_vis_pc_array = trav_vis_pc_array

    def make_array_from_trav_data(self, data):
        pc = ros_numpy.numpify(data).ravel()
        #pc = np.stack([pc[f] for f in ['x', 'y', 'z', 'trav', 'slopex', 'slopey', 'cost']]).T
        pc = np.stack([pc[f] for f in ['x', 'y', 'z']]).T
        n_pts = len(pc)
        if n_pts > self.config["max_pc_points"]:
            decim_coeff = self.config["max_pc_points"] / float(n_pts)
            pc = pc[np.random.rand(n_pts) < decim_coeff, :]
        return pc

    def filter_pc_xyz(self, pc, xyz, bnd):
        pc_filt = pc[np.logical_and(pc[:, 0] < xyz[0] + bnd, pc[:, 0] > xyz[0] - bnd), :]
        pc_filt = pc_filt[np.logical_and(pc_filt[:, 1] < xyz[1] + bnd, pc_filt[:, 1] > xyz[1] - bnd), :]
        pc_filt = pc_filt[np.logical_and(pc_filt[:, 2] < xyz[2] + bnd, pc_filt[:, 2] > xyz[2] - bnd), :]
        return pc_filt

    def localize_and_cut_pc(self, pc_array, frame_name, msg, bnd, pose_dict):
        # Filter pc which is of a radius greater than pc_cutoff_radius
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

        ## Add a bit of noise
        #pc_arr_bl += np.random.randn(*pc_arr_bl.shape) * 0.02

        return pc_arr_bl

    def get_all_observation_dict(self):
        pose_dict = ros_utilities.get_robot_pose_dict(self.config["root_frame"], "base_link", self.tf_buffer, rospy.Time(0))
        if pose_dict is None: return None

        proprioceptive_data_dict = self.get_proprioceptive_data_dict()
        if proprioceptive_data_dict is None: return None

        exteroceptive_data_dict = self.get_exteroceptive_data_dict(pose_dict)
        if exteroceptive_data_dict is None: return None

        return utilities.merge_dicts([pose_dict, proprioceptive_data_dict, exteroceptive_data_dict])

    def get_proprioceptive_data_dict(self):
        return self.get_flipper_state_dict()

    def get_exteroceptive_data_dict(self, pose_dict):
        lmap = self.get_local_trav_map(self.current_base_link_frame, pose_dict)

        if lmap is None:
            return None

        height_offset = 0#-pose_dict["euler"][1] * 0.2

        # Frontal mid features
        frontal_mid_pc = self.get_bnd_pts(lmap, self.config["front_mid_feat_bnd"])
        frontal_mid_feat = self.get_pc_feat(frontal_mid_pc)

        # Frontal low features
        self.front_low_feat_bnd = deepcopy(self.config["front_low_feat_bnd"])
        self.front_low_feat_bnd[4] += height_offset
        self.front_low_feat_bnd[5] += height_offset
        frontal_low_pc = self.get_bnd_pts(lmap, self.front_low_feat_bnd)
        frontal_low_feat = self.get_pc_feat(frontal_low_pc)

        # rear low features
        self.rear_low_feat_bnd = deepcopy(self.config["rear_low_feat_bnd"])
        self.rear_low_feat_bnd[4] -= height_offset
        self.rear_low_feat_bnd[5] -= height_offset
        rear_low_pc = self.get_bnd_pts(lmap, self.rear_low_feat_bnd)
        rear_low_feat = self.get_pc_feat(rear_low_pc)

        # flipper features
        self.fl_flipper_feat_bnd = deepcopy(self.config["fl_flipper_feat_bnd"])
        self.fl_flipper_feat_bnd[4] += height_offset
        self.fl_flipper_feat_bnd[5] += height_offset
        fl_flipper_pc = self.get_bnd_pts(lmap, self.fl_flipper_feat_bnd)

        self.fr_flipper_feat_bnd = deepcopy(self.config["fr_flipper_feat_bnd"])
        self.fr_flipper_feat_bnd[4] += height_offset
        self.fr_flipper_feat_bnd[5] += height_offset
        fr_flipper_pc = self.get_bnd_pts(lmap, self.fr_flipper_feat_bnd)

        self.rl_flipper_feat_bnd = deepcopy(self.config["rl_flipper_feat_bnd"])
        self.rl_flipper_feat_bnd[4] -= height_offset
        self.rl_flipper_feat_bnd[5] -= height_offset
        rl_flipper_pc = self.get_bnd_pts(lmap, self.rl_flipper_feat_bnd)

        self.rr_flipper_feat_bnd = deepcopy(self.config["rr_flipper_feat_bnd"])
        self.rr_flipper_feat_bnd[4] -= height_offset
        self.rr_flipper_feat_bnd[5] -= height_offset
        rr_flipper_pc = self.get_bnd_pts(lmap, self.rr_flipper_feat_bnd)

        fl_flipper_feat = self.get_pc_feat(fl_flipper_pc, type="flipper")
        fr_flipper_feat = self.get_pc_feat(fr_flipper_pc, type="flipper")
        rl_flipper_feat = self.get_pc_feat(rl_flipper_pc, type="flipper")
        rr_flipper_feat = self.get_pc_feat(rr_flipper_pc, type="flipper")

        # Haar like features
        bnd_boxes = []
        bnd_boxes.append([-0.6, 0.6, -0.4, 0.4, -0.5, 0.5])

        bnd_boxes.append([0.0, 0.6, -0.4, 0.4, -0.5, 0.5])
        bnd_boxes.append([-0.2, 0.4, -0.4, 0.4, -0.5, 0.5])
        bnd_boxes.append([-0.4, 0.2, -0.4, 0.4, -0.5, 0.5])
        bnd_boxes.append([-0.6, 0.0, -0.4, 0.4, -0.5, 0.5])

        bnd_boxes.append([0.0, 0.6, 0, 0.4, -0.5, 0.5])
        bnd_boxes.append([-0.2, 0.4, 0, 0.4, -0.5, 0.5])
        bnd_boxes.append([-0.4, 0.2, 0, 0.4, -0.5, 0.5])
        bnd_boxes.append([-0.6, 0.0, 0, 0.4, -0.5, 0.5])

        bnd_boxes.append([0.0, 0.6, -0.4, 0, -0.5, 0.5])
        bnd_boxes.append([-0.2, 0.4, -0.4, 0, -0.5, 0.5])
        bnd_boxes.append([-0.4, 0.2, -0.4, 0, -0.5, 0.5])
        bnd_boxes.append([-0.6, 0.0, -0.4, 0, -0.5, 0.5])

        haar_feats = []
        for bbx in bnd_boxes:
            haar_feats.append(self.get_pc_feat_haar(lmap, bbx))

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
                     "stairs_feats" : stairs_feats,
                     "haar_feats" : haar_feats}

        return data_dict

    def get_pc_feat(self, pc, type="full"):
        def n_to_intensity(n):
            if type=="full":
                return np.minimum(float(n) / self.config["n_max_intensity"], 1.)
            else:
                return np.minimum(float(n) / self.config["n_max_intensity_flipper"], 1.)

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

    def get_pc_feat_haar(self, pc, bnd_box, type="full"):
        bbx_pos = (bnd_box[1] + bnd_box[0]) / 2, bnd_box[1], bnd_box[2], bnd_box[3], bnd_box[4], bnd_box[5]
        bbx_neg = bnd_box[0], (bnd_box[1] + bnd_box[0]) / 2, bnd_box[2], bnd_box[3], bnd_box[4], bnd_box[5]
        pc_pos = self.get_bnd_pts(pc, bbx_pos)
        pc_neg = self.get_bnd_pts(pc, bbx_neg)

        bbx_area = (bnd_box[1] - bnd_box[0]) * (bnd_box[3] - bnd_box[2])

        if len(pc_pos) == 0:
            sum_pos = 0
        else:
            sum_pos = np.sum(pc_pos[:, 2])

        if len(pc_neg) == 0:
            sum_neg = 0
        else:
            sum_neg = np.sum(pc_neg[:, 2])

        feat = (sum_pos - sum_neg) / bbx_area

        return feat

    def get_flipper_state_dict(self):
        flipper_name_dict = ["front_left_flipper",
                             "front_right_flipper",
                             "rear_left_flipper",
                             "rear_right_flipper"]

        flipper_dict = {}
        for fn in flipper_name_dict:
            try:
                trans = self.tf_buffer.lookup_transform(self.current_base_link_frame,
                                                        self.config["robot_prefix"] + fn,
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

    def get_local_trav_map(self, frame, pose_dict):
        with self.trav_vis_pc_lock:
            if self.trav_vis_pc_array is None:
                return None
            pc_arr_bl = self.localize_and_cut_pc(self.trav_vis_pc_array, frame, self.trav_vis_pc_data, 2., pose_dict)
        return pc_arr_bl

    def get_pc_feat_vec(self, obs_dict):
        vec = obs_dict["fl_flipper_feat"] + \
            obs_dict["fr_flipper_feat"] + \
            obs_dict["rl_flipper_feat"] + \
            obs_dict["rr_flipper_feat"] + \
            obs_dict["euler"][:2]
            #(obs_dict["front_left_flipper"], obs_dict["front_right_flipper"], obs_dict["rear_left_flipper"], obs_dict["rear_right_flipper"]) + \
        return vec

    def get_flipper_transformations(self):
        flipper_name_dict = ["front_left_flipper",
                             "front_right_flipper",
                             "rear_left_flipper",
                             "rear_right_flipper"]

        flipper_tf_dict = {}
        for fn in flipper_name_dict:
            try:
                trans = self.tf_buffer.lookup_transform(self.config["root_frame"],
                                                        self.config["robot_prefix"] + fn,
                                                        rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
                rospy.logwarn("Get_flipper_transformations: TRANSFORMATION FAILED, err: {}".format(err))
                return None

            flipper_tf_dict[fn] = trans

        return flipper_tf_dict

    def get_sampled_flipper_transformations(self, flipper_tf_dict):
        pass

    def get_flipper_collision_transformations(self):
        pass

    def get_current_state(self):
        with self.controller_state_subscriber.lock:
            if self.controller_state_subscriber.msg is not None:
                return self.controller_state_subscriber.msg.data
        with self.teleop_state_subscriber.lock:
            if self.teleop_state_subscriber.msg is not None:
                return self.short_to_state_name_dict[self.teleop_state_subscriber.msg.data]

    def process_flipper_contacts(self, obs_dict):
        # Get flipper transformations in world coordinates
        flipper_tf_dict = self.get_flipper_transformations()

        # Generate sampled transformations for each flipper
        flipper_tf_sampled_dict = self.get_sampled_flipper_transformations(flipper_tf_dict)

        # Go over sampled transformations and check for collisions
        flipper_collision_tf_dict = self.get_flipper_collision_transformations(flipper_tf_sampled_dict)

        # Publish flipper tfs
        self.publish_flipper_tfs(flipper_collision_tf_dict)

        # Return target flipper positions and bounding boxes
        flipper_angles_dict = self.get_flipper_angles_from_tf(flipper_collision_tf_dict)

        return flipper_collision_tf_dict, flipper_angles_dict

    def publish_pc_bnds(self, pc_bnds_dict, frame):
        pc_bnds = np.concatenate((#pc_bnds_dict["frontal_low_pc"],
                                  #pc_bnds_dict["rear_low_pc"],
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

    def publish_txt_info(self, obs_dict):
        strs = []
        strs.append("Pitch: {} \n".format(round(obs_dict["euler"][1], 4)))
        strs.append("frontal_low_feat: {} \n".format([round(o,4) for o in obs_dict["frontal_low_feat"]]))
        strs.append("frontal_mid_feat: {} \n".format([round(o,4) for o in obs_dict["frontal_mid_feat"]]))
        strs.append("rear_low_feat: {} \n".format([round(o,4) for o in obs_dict["rear_low_feat"]]))
        strs.append("fl_flipper_feat: {} \n".format([round(o,4) for o in obs_dict["fl_flipper_feat"]]))
        strs.append("fr_flipper_feat: {} \n".format([round(o,4) for o in obs_dict["fr_flipper_feat"]]))
        strs.append("rl_flipper_feat: {} \n".format([round(o,4) for o in obs_dict["rl_flipper_feat"]]))
        strs.append("rr_flipper_feat: {} \n".format([round(o,4) for o in obs_dict["rr_flipper_feat"]]))
        strs.append("stagnation: {} \n".format(obs_dict["stagnation"]))

        txt = ""
        for s in strs:
            txt += s

        msg = String()
        msg.data = txt
        self.text_publisher.publish(msg)

    def publish_pc_feat_vec(self, obs_dict):
        msg = Float64MultiArray()
        msg.data = self.get_pc_feat_vec(obs_dict)
        self.pc_feat_vec_publisher.publish(msg)

    def publish_pc_feat_msg(self, obs_dict):
        msg = MarvPCFeats()
        msg.frontal_low_feat.data = obs_dict["frontal_low_feat"]
        msg.frontal_mid_feat.data = obs_dict["frontal_mid_feat"]
        msg.rear_low_feat.data = obs_dict["rear_low_feat"]
        msg.fl_flipper_feat.data = obs_dict["fl_flipper_feat"]
        msg.fr_flipper_feat.data = obs_dict["fr_flipper_feat"]
        msg.rl_flipper_feat.data = obs_dict["rl_flipper_feat"]
        msg.rr_flipper_feat.data = obs_dict["rr_flipper_feat"]
        self.pc_feat_msg_publisher.publish(msg)

    def publish_all_bbx(self):
        #self.publish_bbx(self.config["front_mid_feat_bnd"], 0, [0, 0, 1])
        #self.publish_bbx(self.front_low_feat_bnd, 0, [0.2, 0.6, 0.6])
        #self.publish_bbx(self.rear_low_feat_bnd, 1, [0.4, 0.4, 0.4])
        self.publish_bbx(self.fl_flipper_feat_bnd, 2, [0.2, 0.6, 0.6])
        self.publish_bbx(self.fr_flipper_feat_bnd, 3, [0.2, 0.6, 0.6])
        self.publish_bbx(self.rl_flipper_feat_bnd, 4, [0.2, 0.6, 0.6])
        self.publish_bbx(self.rr_flipper_feat_bnd, 5, [0.2, 0.6, 0.6])

    def publish_dsm_distrib_bbx(self):
        with self.dsm_distrib_subscriber.lock:
            if self.dsm_distrib_subscriber.msg is not None:
                dsm_distrib = deepcopy(self.dsm_distrib_subscriber.msg).data
                max_idx = np.argmax(dsm_distrib)
                for i in range(len(dsm_distrib)):
                    bnd_bx = [i * 0.1 -0.35,i * 0.1 + 0.1 -0.35, -0.05, 0.05, 1, 1 + dsm_distrib[i]]
                    color = [0.2, 0.6, 0.6]
                    if i == max_idx:
                        color = [0.0, 0.8, 0.2]
                    self.publish_bbx(bnd_bx, 6 + i, color)

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
        marker_msg.color.a = 0.6

        marker_msg.color.r = rgb[0]
        marker_msg.color.g = rgb[1]
        marker_msg.color.b = rgb[2]

        self.marker_publisher.publish(marker_msg)

    def publish_flipper_positions(self, flipper_positions):
        msg = Float64MultiArray()
        msg.data = flipper_positions
        self.flipper_position_publisher.publish(msg)

    def publish_haar_feats(self, obs_dict):
        msg = Float64MultiArray()
        msg.data = obs_dict["haar_feats"]
        self.haar_feat_publisher.publish(msg)

    def publish_stagnation(self):
        self.stagnation_publisher.publish(Float64(data=self.stagnation))

    def update_robot_progress(self, obs_dict):
        new_state = self.get_current_state()
        changed_state = False
        if new_state != self.current_state:
            changed_state = True
        self.current_state = new_state

        position_msg = obs_dict["position"]
        robot_xyz = np.array([position_msg.x, position_msg.y, position_msg.z])
        (_, _, robot_yaw) = obs_dict["euler"]

        with self.track_vel_subscriber.lock:
            track_vel_msg = self.track_vel_subscriber.msg

        if not hasattr(self, "prev_robot_xyz") or track_vel_msg is None:
            self.prev_robot_xyz = robot_xyz
            self.prev_robot_yaw = robot_yaw
            self.stagnation = 0. # [0,1]
            obs_dict["stagnation"] = self.stagnation
            return

        max_pos_delta = np.max(np.abs(robot_xyz - self.prev_robot_xyz))
        max_body_linear_vel_est = max_pos_delta * self.config["ros_rate"] * 1.4

        if not hasattr(self, "velocity_queue"):
            self.velocity_queue = []
        self.velocity_queue.append(max_body_linear_vel_est)
        if len(self.velocity_queue) > 5:
            del self.velocity_queue[0]
        avg_lin_vel = sum(self.velocity_queue) / len(self.velocity_queue)

        max_yaw_delta = np.abs(robot_yaw - self.prev_robot_yaw)
        max_body_yaw_vel_est = max_yaw_delta * self.config["ros_rate"]

        track_vel_x = track_vel_msg.linear.x
        track_vel_ang_z = track_vel_msg.angular.z

        if not hasattr(self, "track_vel_x_queue"):
            self.track_vel_x_queue = []
        self.track_vel_x_queue.append(abs(track_vel_x))
        if len(self.track_vel_x_queue) > 5:
            del self.track_vel_x_queue[0]
        avg_track_vel_x = sum(self.track_vel_x_queue) / len(self.track_vel_x_queue)

        linear_stagnation_condition = (avg_lin_vel < 0.04) and (abs(avg_track_vel_x) > 0.15)
        #angular_stagnation_condition = (abs(track_vel_ang_z) - max_body_yaw_vel_est) > (0.8 * abs(track_vel_ang_z))

        if linear_stagnation_condition:
            self.stagnation = np.minimum(self.stagnation + 0.10, 1)
        else:
            self.stagnation = np.maximum(self.stagnation - 0.05, 0)

        if changed_state:
            self.stagnation = 0

        obs_dict["stagnation"] = self.stagnation

        self.prev_robot_xyz = robot_xyz
        self.prev_robot_yaw = robot_yaw

    def loop(self):
        while not rospy.is_shutdown():
            self.step()

    def step(self):
        obs_dict = self.get_all_observation_dict()

        if obs_dict is None:
            rospy.loginfo_throttle(1, "Obs_dict was none, skipping iteration")
            self.ros_rate.sleep()
            return

        # Test flipper collisions
        #flipper_collision_tf_dict, flipper_angles_dict = self.process_flipper_contacts(obs_dict)

        # Update robot progress
        self.update_robot_progress(obs_dict)

        self.publish_txt_info(obs_dict)
        self.publish_all_bbx()
        self.publish_dsm_distrib_bbx()
        self.publish_pc_feat_msg(obs_dict)
        self.publish_pc_feat_vec(obs_dict)
        self.publish_pc_bnds(obs_dict, self.current_base_link_frame)
        self.publish_flipper_positions([obs_dict["front_left_flipper"], obs_dict["front_right_flipper"],
                                       obs_dict["rear_left_flipper"], obs_dict["rear_right_flipper"]])
        self.publish_haar_feats(obs_dict)

        self.publish_stagnation()
        #self.publish_flipper_bounding_boxes(flipper_collision_tf_dict)
        #sself.publish_flipper_collision_angles(flipper_angles_dict)

        try:
            self.ros_rate.sleep()
        except rospy.ROSInterruptException:
            pass

        return obs_dict

    def make_dataset(self, n_secs, suffix=None):
        rospy.loginfo("Starting recording for dataset creation purposes...")

        start_time = time.time()

        data_dict_list = []
        while not rospy.is_shutdown():
            obs_dict = self.step()

            if obs_dict is not None:
                del obs_dict["lmap_pc"]
                obs_dict["quat"] = (obs_dict["quat"].w, obs_dict["quat"].x, obs_dict["quat"].y, obs_dict["quat"].z)
                obs_dict["position"] = (obs_dict["position"].x, obs_dict["position"].y, obs_dict["position"].z)
                obs_dict["pc_feat_vec"] = self.get_pc_feat_vec(obs_dict)
                with self.teleop_state_subscriber.lock:
                    if self.teleop_state_subscriber.msg is not None:
                        obs_dict["teleop_state"] = self.teleop_state_subscriber.msg.data
                    else:
                        obs_dict["teleop_state"] = "N"
                data_dict_list.append(obs_dict)

            if time.time() - start_time > n_secs:
                break

        time.sleep(1)
        rospy.loginfo("Saving dataset")

        # Save dataset
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/supervised_dataset/tradr/sm")
        if suffix is not None:
            dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/supervised_dataset/tradr/sm_{}".format(suffix))
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
        config_name = "tradr_sys_feature_processor_config.yaml"

    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/{}".format(config_name)), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    fp = TradrFeatureProcessor(config)
    fp.loop()
    #fp.make_dataset(600, "reimp")

if __name__=="__main__":
    main()


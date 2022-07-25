#!/usr/bin/python

import os
import pickle
import sys
import threading
import time

import rospy
import tf2_ros
from augmented_robot_trackers.msg import MarvPCFeats
from marv_msgs.msg import Float64MultiArray as MarvFloat64MultiArray
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String

from src.policies.policies import *
from src.utilities import utilities, ros_utilities
from copy import deepcopy

class MarvFlipperController:
    def __init__(self, config):
        self.config = config

        if self.config["linear"]:
            self.policy_suffix = "lin"
        else:
            self.policy_suffix = "non_lin"

        self.base_link_dict = {"base_link" : self.config["robot_prefix"] + "base_link_zrp",
                               "base_link_rev" : self.config["robot_prefix"] + "base_link_zrp_rev"}

        self.state_list = ["NEUTRAL",
                            "ASCENDING_FRONT",
                            "UP_STAIRS",
                            "ASCENDING_REAR",
                            "DESCENDING_FRONT",
                            "DOWN_STAIRS",
                            "DESCENDING_REAR"]

        self.enable_flippers = self.config["enable_flippers"]

        self.state_to_short_name_dict = {"NEUTRAL": "N",
                                         "ASCENDING_FRONT": "AF",
                                         "UP_STAIRS": "US",
                                         "ASCENDING_REAR": "AR",
                                         "DESCENDING_FRONT": "DF",
                                         "DOWN_STAIRS": "DS",
                                         "DESCENDING_REAR": "DR"}

        self.short_to_state_name_dict = {v: k for k, v in self.state_to_short_name_dict.items()}

        self.state_transition_dict = {"N": ["N", "AF", "DF"],
                                     "AF": ["AF", "N", "US", "AR"],
                                     "US": ["US", "AR"],
                                     "AR": ["AR", "N"],
                                     "DF": ["DF", "N", "DS", "DR"],
                                     "DS": ["DS", "DR"],
                                     "DR": ["DR", "N"]}

        self.current_base_link_frame = self.base_link_dict["base_link"]

        # Initialize state machine policy
        self.init_control_policy()

        self.reset_time = time.time()
        self.init_ros()

    def init_control_policy(self):
        self.current_state = "NEUTRAL"
        if self.config["mode"] == "handcrafted_sm":
            self.policy = SM()
            self.policy.set_handcrafted_params()
        elif self.config["mode"] == "bbx_sm":
            self.policy = SM()
            param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "opt/agents/imitation_sm.pkl")
            self.policy.set_params(pickle.load(open(param_path, "rb")))
        elif self.config["mode"] == "differentiable_sm":
            self.policy = DSM(feat_dim=self.config["feat_vec_dim"],
                              state_transition_dict=self.state_transition_dict, state_list=self.state_list,
                              initial_state="N", linear=self.config["linear"])
            param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "opt/agents/{}/imitation_dsm_{}.p".format(self.config["data_dir_name"], self.policy_suffix))
            self.policy.load_state_dict(T.load(param_path), strict=False)
        elif self.config["mode"] == "differentiable_detached_sm":
            self.policy = DSM(feat_dim=self.config["feat_vec_dim"],
                              state_transition_dict=self.state_transition_dict,
                              initial_state="N", linear=self.config["linear"])
            param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "opt/agents/{}/imitation_dsm_detached_{}.p".format(self.config["data_dir_name"], self.policy_suffix))
            self.policy.load_state_dict(T.load(param_path), strict=False)
        elif self.config["mode"] == "state_classification":
            self.policy = StateClassifier(feat_dim=self.config["feat_vec_dim"], state_list=self.state_list, linear=self.config["linear"])
            param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "opt/agents/{}/imitation_state_classification_{}.p".format(self.config["data_dir_name"], self.policy_suffix))
            self.policy.load_state_dict(T.load(param_path), strict=False)
        elif self.config["mode"] == "reimp":
            self.policy = StateClassifier(feat_dim=13, state_list=self.state_list, linear=self.config["linear"])
            param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "opt/agents/{}/imitation_reimp_{}.p".format(self.config["data_dir_name"], self.policy_suffix))
            self.policy.load_state_dict(T.load(param_path), strict=False)
        elif self.config["mode"] == "neutral":
            self.policy = NeutralClassifier()
        elif self.config["mode"] == "random":
            self.policy = RandomClassifier(self.state_list)
        else:
            raise NotImplementedError
        self.last_state_change_time = time.time()

    def init_ros(self):
        rospy.init_node("art_flipper_controller")

        self.ros_rate = rospy.Rate(self.config["ros_rate"])
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.flippers_pos_publisher = rospy.Publisher("marv/flippers_position_controller/cmd_vel",
                                                      MarvFloat64MultiArray,
                                                      queue_size=1)

        self.flippers_max_torque_publisher = rospy.Publisher("marv/flippers_max_torque_controller/cmd_vel",
                                                      MarvFloat64MultiArray,
                                                      queue_size=1)

        self.tracker_state_publisher = rospy.Publisher("art/marv_flipper_controller_state",
                                                 String,
                                                 queue_size=1)

        self.tracker_dsm_distrib_publisher = rospy.Publisher("art/dsm_distrib",
                                                 Float64MultiArray,
                                                 queue_size=1)

        self.flipper_positions_subscriber = ros_utilities.subscriber_factory("art/flipper_positions", Float64MultiArray)
        self.pc_feat_vec_subscriber = ros_utilities.subscriber_factory("art/pc_feat_vec", Float64MultiArray)
        self.pc_feat_msg_subscriber = ros_utilities.subscriber_factory("art/pc_feat_msg", MarvPCFeats)
        self.haar_feats_subscriber = ros_utilities.subscriber_factory("art/haar_feats", Float64MultiArray)

        time.sleep(0.1)

    def get_current_obs_dict(self):
        pose_dict = ros_utilities.get_robot_pose_dict(self.config["root_frame"],
                                                      "X1/base_link",
                                                      self.tf_buffer,
                                                      rospy.Time(0))
        if pose_dict is None: return None

        with self.flipper_positions_subscriber.lock:
            if self.flipper_positions_subscriber.msg is None:
                return None
            proprioceptive_data = self.flipper_positions_subscriber.msg.data
            proprioceptive_data_dict = {"front_left_flipper" : proprioceptive_data[0],
                                        "front_right_flipper" : proprioceptive_data[1],
                                        "rear_left_flipper": proprioceptive_data[2],
                                        "rear_right_flipper": proprioceptive_data[3]}
        if proprioceptive_data_dict is None: return None

        with self.pc_feat_msg_subscriber.lock:
            if self.pc_feat_msg_subscriber.msg is None: return None
            exteroceptive_data_dict = {"frontal_low_feat" : self.pc_feat_msg_subscriber.msg.frontal_low_feat.data,
                                       "frontal_mid_feat" : self.pc_feat_msg_subscriber.msg.frontal_mid_feat.data,
                                       "rear_low_feat" : self.pc_feat_msg_subscriber.msg.rear_low_feat.data,
                                       "fl_flipper_feat" : self.pc_feat_msg_subscriber.msg.fl_flipper_feat.data,
                                       "fr_flipper_feat" : self.pc_feat_msg_subscriber.msg.fr_flipper_feat.data,
                                       "rl_flipper_feat" : self.pc_feat_msg_subscriber.msg.rl_flipper_feat.data,
                                       "rr_flipper_feat" : self.pc_feat_msg_subscriber.msg.rr_flipper_feat.data}

        with self.pc_feat_vec_subscriber.lock:
            if self.pc_feat_vec_subscriber.msg is None: return None
            exteroceptive_data_dict["pc_feat_vec"] = self.pc_feat_vec_subscriber.msg.data

        return utilities.merge_dicts([pose_dict, proprioceptive_data_dict, exteroceptive_data_dict])

    def get_current_feature_vec(self, obs_dict):
        if obs_dict is None: return None
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

        feature_vec = [pitch] + list(obs_dict["frontal_low_feat"]) + list(obs_dict["rear_low_feat"])

        #feature_vec = [pitch, flat_pitch_dev, small_pitch, large_pitch, small_dip, large_dip, flat_ground,
        #               untraversable_elevation, small_frontal_elevation, large_frontal_elevation,
        #               small_frontal_lowering, large_frontal_lowering, low_frontal_point_presence,
        #               small_rear_elevation, large_rear_elevation, small_rear_lowering, large_rear_lowering,
        #               not_rear_lowering, low_rear_point_presence]  # 19
        return feature_vec

    def calculate_flipper_action_basic(self):
        fl_flipper_stab_correction = 0
        fr_flipper_stab_correction = 0
        rl_flipper_stab_correction = 0
        rr_flipper_stab_correction = 0

        flipper_state = self.current_state

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

    def calculate_flipper_action(self, current_state, obs_dict):
        flipper_commands_dict = {}
        flipper_commands_dict["front_left"] = self.config["FLIPPERS_{}".format(current_state)][0]
        flipper_commands_dict["front_right"] = self.config["FLIPPERS_{}".format(current_state)][1]
        flipper_commands_dict["rear_left"] = self.config["FLIPPERS_{}".format(current_state)][2]
        flipper_commands_dict["rear_right"] = self.config["FLIPPERS_{}".format(current_state)][3]

        flipper_torques_dict = {}
        flipper_torques_dict["front_left"] = self.config["FLIPPERS_CURRENT_{}".format(current_state)][0]
        flipper_torques_dict["front_right"] = self.config["FLIPPERS_CURRENT_{}".format(current_state)][0]
        flipper_torques_dict["rear_left"] = self.config["FLIPPERS_CURRENT_{}".format(current_state)][1]
        flipper_torques_dict["rear_right"] = self.config["FLIPPERS_CURRENT_{}".format(current_state)][1]

        return flipper_commands_dict, flipper_torques_dict

    def decide_new_state(self, obs_dict, obs_feat_vec):
        if self.config["mode"] == "handcrafted_sm":
            new_state = self.policy.decide_next_state(obs_dict, self.current_state)
        elif self.config["mode"] == "bbx_sm":
            new_state = self.policy.decide_next_state(obs_dict, self.current_state)
        elif self.config["mode"] == "differentiable_sm":
            if not hasattr(self, "dsm_state_distrib"):
                self.dsm_state_distrib = None
            new_state, self.dsm_state_distrib = self.policy.calculate_next_state_diff(obs_feat_vec, self.dsm_state_distrib)
        elif self.config["mode"] == "differentiable_detached_sm":
            new_state, _ = self.policy.calculate_next_state_detached(obs_feat_vec, self.current_state)
        elif self.config["mode"] == "state_classification":
            new_state, _ = self.policy.decide_next_state(obs_feat_vec)
        elif self.config["mode"] == "reimp":
            while True:
                with self.haar_feats_subscriber.lock:
                    if self.haar_feats_subscriber.msg is not None:
                        feats = deepcopy(self.haar_feats_subscriber.msg.data)
                        break
            new_state, _ = self.policy.decide_next_state(feats)
        elif self.config["mode"] == "neutral":
            new_state = self.policy.decide_next_state()
        elif self.config["mode"] == "random":
            new_state = self.policy.decide_next_state()
        else:
            raise NotImplementedError

        return new_state

    def publish_tracker_state(self):
        msg = String()
        msg.data = self.current_state
        self.tracker_state_publisher.publish(msg)

    def publish_dsm_distrib(self):
        msg = Float64MultiArray()
        msg.data = self.dsm_state_distrib
        self.tracker_dsm_distrib_publisher.publish(msg)

    def step(self, new_state, obs_dict):
        if new_state != self.current_state:
            rospy.loginfo("Current state: {}".format(self.current_state))
            self.last_state_change_time = time.time()

        self.current_state = new_state

        self.publish_tracker_state()

        if self.config["mode"] == "differentiable_sm":
            self.publish_dsm_distrib()

    def reset(self):
        self.current_state = "NEUTRAL"
        self.current_tracking_orientation = "fw"
        self.reset_time = time.time()

    def loop(self):
        rospy.loginfo("Art flipper controller starting flipper control...")

        while not rospy.is_shutdown():
            # Observations
            obs_dict = self.get_current_obs_dict()
            obs_feat_vec = self.get_current_feature_vec(obs_dict)
            if obs_dict is None:
                rospy.loginfo_throttle(1, "Obs_dict was none, skipping iteration")
                self.ros_rate.sleep()
                continue

            # Calculate new state
            if time.time() - self.last_state_change_time > 0:
                new_state = self.decide_new_state(obs_dict, obs_feat_vec)
            else:
                new_state = self.current_state

            # Perform control step
            self.step(new_state, obs_dict)

            self.ros_rate.sleep()

def main():
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) == 2:
        config_name = myargv[1]
    else:
        config_name = "marv_flipper_controller_config.yaml"

    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/{}".format(config_name)), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    controller = MarvFlipperController(config)
    controller.loop()

if __name__=="__main__":
    main()


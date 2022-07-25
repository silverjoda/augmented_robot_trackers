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
from tf.transformations import quaternion_matrix
from std_msgs.msg import String

from src.utilities import utilities


class MarvTeleop:
    def __init__(self, config):
        self.config = config

        self.control_modes = ["basic", "tracks_only", "semi", "full", "waypoints", "states", "states_no_tracks"]
        self.control_mode_event_translator = utilities.ButtonToEventTranslator(len(self.control_modes), self.control_modes.index(self.config["control_mode"]))
        self.control_mode = self.control_modes[self.control_mode_event_translator.idx]

        self.base_link_states = [self.config["robot_prefix"] + "base_link",
                                 self.config["robot_prefix"] + "base_link_rev"]
        self.base_link_event_translator = utilities.ButtonToEventTranslator(2)
        self.current_base_link_frame = self.base_link_states[self.base_link_event_translator.idx]

        self.deadmans_button_event_translator = utilities.ButtonToEventTranslator(2, 1)
        self.deadmans_button_state = self.deadmans_button_event_translator.idx

        # D-pad translators
        self.dpad_event_translator_list = [utilities.ButtonToEventTranslator(1, 0) for _ in range(4)]

        self.neutral_flipper_state = [-2, -2, 1.5, 1.5]

        self.flipper_pos_dict = {"N" : [-2, -2, 1.5, 1.5],
                                 "AF" : [-0.4, -0.4, 0.0, 0.0],
                                 "US": [0.1, 0.1, -0.1, -0.1],
                                 "AR" : [0.1, 0.1, -0.6, -0.5],
                                 "DF" : [0.35, 0.35, -0.7, -0.7],
                                 "DS":  [0, 0, 0.05, 0.05],
                                 "DR" : [-0.3, -0.3, 0.4, 0.4]}

        self.current_state = "N"
        self.init_ros(self.config["node_name"])

        # Initiate current tracked flipper positions
        self.init_virtual_flipper_states()

        rospy.loginfo("Starting marv teleop in {} mode".format(self.control_mode))

    def init_virtual_flipper_states(self):
        time.sleep(0.3)
        # Get velocity from tracking error between virtual and real positions
        with self.flippers_state_lock:
            if self.flippers_state_data is not None:
                flippers_state = deepcopy(self.flippers_state_data)
                flippers_state_array = np.array(
                    [flippers_state.frontLeft, flippers_state.frontRight, flippers_state.rearLeft,
                     flippers_state.rearRight])
                self.virtual_flipper_positions = flippers_state_array
            else:
                self.virtual_flipper_positions = np.array([-np.pi/4, -np.pi/4, np.pi/4, np.pi/4])

    def step_virtual_flipper_states(self, dirs):
        # Get flipper transformation
        flipper_dict = self.get_flipper_state_dict()

        if flipper_dict is not None and False:
            flippers_state_array = np.array(
                [flipper_dict["front_left_flipper"], flipper_dict["front_right_flipper"],
                 flipper_dict["rear_left_flipper"], flipper_dict["rear_right_flipper"]])
            lb = flippers_state_array - self.config["max_flippers_dev"]
            ub = flippers_state_array + self.config["max_flippers_dev"]
            self.virtual_flipper_positions = np.clip(self.virtual_flipper_positions + np.array(dirs) * self.config["flipper_position_integration_constant"], lb, ub)
        else:
            self.virtual_flipper_positions += np.array(dirs) * self.config["flipper_position_integration_constant"]

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
            if self.current_base_link_frame == self.config["robot_prefix"] + "base_link":
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

    def init_ros(self, name):
        rospy.init_node(name)

        self.ros_rate = rospy.Rate(self.config["ros_rate"])
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.config["tf_buffer_size"]))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.flippers_state_lock = threading.Lock()
        self.flippers_state_data = None

        self.joy_lock = threading.Lock()
        self.joy_data = None

        self.tracks_vel_publisher = rospy.Publisher(self.config["tracks_velocity_control_out_joy"],
                                                    Twist,
                                                    queue_size=1)

        self.flippers_pos_publisher = rospy.Publisher(self.config["flippers_position_control_out_joy"],
                                                      Float64MultiArray,
                                                      queue_size=1)

        self.joy_plan_publisher = rospy.Publisher(self.config["joy_plan_out"],
                                                  Path,
                                                  queue_size=1)

        self.cmd_vel_ub_publisher = rospy.Publisher(self.config["cmd_vel_ub_out"],
                                                    Twist,
                                                    queue_size=1)

        self.text_publisher = rospy.Publisher("teleop/text_info",
                                              String,
                                              queue_size=1)

        self.state_publisher = rospy.Publisher("teleop/state",
                                              String,
                                              queue_size=1)

        rospy.Subscriber(self.config["joy_in_topic"],
                         Joy,
                         self._ros_joy_callback, queue_size=1)

        time.sleep(0.1)

        rospy.loginfo("Marv teleop node {}: initialized ros".format(self.config["node_name"]))

    def _ros_joy_callback(self, data):
        with self.joy_lock:
            self.joy_data = data

            control_switch_event, control_idx = self.control_mode_event_translator.update(data.buttons[1])
            if control_switch_event:
                self.control_mode = self.control_modes[control_idx]
                rospy.loginfo("Changed control mode to: {}".format(self.control_mode))

            bl_switch_event, bl_idx =self.base_link_event_translator.update(data.buttons[2])
            if bl_switch_event:
                self.current_base_link_frame = self.base_link_states[bl_idx]
                rospy.loginfo("Changed base_link frame to : {}".format(self.current_base_link_frame))

            dmb_switch_event, dmb_idx = self.deadmans_button_event_translator.update(data.buttons[0])
            if dmb_switch_event:
                self.deadmans_button_state = bool(dmb_idx)
                rospy.loginfo("Changed dmb state to : {}".format(self.deadmans_button_state))

            # Reset flippers to neutral position
            if data.buttons[7]:
                self.virtual_flipper_positions = np.array(self.neutral_flipper_state)

            right_dpad_event, _ = self.dpad_event_translator_list[0].update(data.axes[6] < 0)
            left_dpad_event, _ = self.dpad_event_translator_list[1].update(data.axes[6] > 0)
            down_dpad_event, _ = self.dpad_event_translator_list[2].update(data.axes[7] < 0)
            up_dpad_event, _ = self.dpad_event_translator_list[3].update(data.axes[7] > 0)

            self.decide_next_state([right_dpad_event, left_dpad_event, down_dpad_event, up_dpad_event])

    def loop(self):
        while not rospy.is_shutdown():
            with self.joy_lock:
                joy_data = deepcopy(self.joy_data)

            if joy_data is not None:
                if self.deadmans_button_state:
                    self.process_joy_and_publish(joy_data)

            self.publish_txt_info()
            self.publish_state()

            self.ros_rate.sleep()

    def process_joy_and_publish(self, joy_data):
        if self.control_mode == "basic":
            processed_commands = self._process_basic(joy_data)
            self.publish_commands(processed_commands)
        elif self.control_mode == "tracks_only":
            processed_commands = self._process_tracks_only(joy_data)
            self.publish_commands(processed_commands, publish_flippers=False)
        elif self.control_mode == "semi":
            processed_commands = self._process_semi(joy_data)
            self.publish_commands(processed_commands)
        elif self.control_mode == "full":
            processed_commands = self._process_full(joy_data)
            self.publish_commands(processed_commands)
        elif self.control_mode == "waypoints":
            processed_path, vel_ub = self._process_waypoints(joy_data)
            self.publish_plan(processed_path, vel_ub)
        elif self.control_mode == "states":
            processed_commands = self._process_states(joy_data)
            self.publish_commands(processed_commands, publish_flippers=False)
        elif self.control_mode == "states_no_tracks":
            processed_commands = self._process_states(joy_data)
            self.publish_commands(processed_commands, publish_flippers=False, publish_tracks=False)
        else:
            raise NotImplementedError

    def _process_basic(self, joy_data):
        tracks_vel_lin = joy_data.axes[4] * self.config["max_tracks_velocity"]
        tracks_vel_ang = joy_data.axes[3] * self.config["max_tracks_velocity"]
        return tracks_vel_ang, tracks_vel_lin, -1.57, -1.57, 1.57, 1.57

    def _process_tracks_only(self, joy_data):
        tracks_vel_lin = joy_data.axes[4] * self.config["max_tracks_velocity"]
        tracks_vel_ang = joy_data.axes[3] * self.config["max_tracks_velocity"]
        return tracks_vel_ang, tracks_vel_lin, None, None, None, None

    def _process_semi(self, joy_data):
        # Calculate tracks vel:
        LB = joy_data.buttons[4]
        RB = joy_data.buttons[5]
        RT = (-joy_data.axes[5] + 1.) * 0.5
        LT = (-joy_data.axes[2] + 1.) * 0.5

        tracks_vel_lin = RT * self.config["max_tracks_velocity"] * (1 - RB * 2)
        tracks_vel_ang = LT * self.config["max_tracks_velocity"] * (1 - LB * 2)

        # Step the virtual flippers
        self.step_virtual_flipper_states([joy_data.axes[1], joy_data.axes[1], joy_data.axes[4], joy_data.axes[4]])

        # Calculate flippers vel:
        fl_flipper_pos = self.virtual_flipper_positions[0]
        fr_flipper_pos = self.virtual_flipper_positions[0]
        rl_flipper_pos = self.virtual_flipper_positions[2]
        rr_flipper_pos = self.virtual_flipper_positions[2]

        return tracks_vel_ang, tracks_vel_lin, fl_flipper_pos, fr_flipper_pos, rl_flipper_pos, rr_flipper_pos

    def _process_full(self, joy_data):
        # Calculate tracks vel:
        LB = joy_data.buttons[4]
        RB = joy_data.buttons[5]
        RT = (-joy_data.axes[5] + 1.) * 0.5
        LT = (-joy_data.axes[2] + 1.) * 0.5

        tracks_vel_lin = RT * self.config["max_tracks_velocity"] * (1 - RB * 2)
        tracks_vel_ang = LT * self.config["max_tracks_velocity"] * (1 - LB * 2)

        # Step the virtual flippers
        self.step_virtual_flipper_states([joy_data.axes[1], joy_data.axes[4], joy_data.axes[0], -joy_data.axes[3]])

        # Calculate flippers vel:
        fl_flipper_pos = self.virtual_flipper_positions[0]
        fr_flipper_pos = self.virtual_flipper_positions[1]
        rl_flipper_pos = self.virtual_flipper_positions[2]
        rr_flipper_pos = self.virtual_flipper_positions[3]

        return tracks_vel_ang, tracks_vel_lin, fl_flipper_pos, fr_flipper_pos, rl_flipper_pos, rr_flipper_pos

    def _process_waypoints(self, joy_data):
        # Get plan commands
        vel_ub = joy_data.axes[4]
        theta = joy_data.axes[3]

        # Get most recent transform to root frame
        try:
            trans = self.tf_buffer.lookup_transform(self.config["root_frame"],
                                                    self.current_base_link_frame,
                                                    rospy.Time(0),
                                                    rospy.Duration(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn_throttle(1, "Marv teleop: process waypoints: TRANSFORMATION OLD, err: {}".format(err))
            return [], 0

        trans_np = ros_numpy.numpify(trans.transform)

        processed_path = []
        for i in range(self.config["joy_plan_n_waypoints"]):
            wp_x = i * self.config["joy_plan_waypoints_gap"] * np.cos(theta * i * self.config["joy_plan_waypoints_curvature"])
            wp_y = i * self.config["joy_plan_waypoints_gap"] * np.sin(theta * i * self.config["joy_plan_waypoints_curvature"])
            wp_glob = np.matmul(trans_np, np.array([wp_x, wp_y, 0, 1]))[:3]
            processed_path.append(wp_glob)
            if abs(vel_ub) < 0.05: break

        return processed_path, vel_ub

    def _process_states(self, joy_data):
        tracks_vel_lin = joy_data.axes[4] * self.config["max_tracks_velocity"]
        tracks_vel_ang = joy_data.axes[3] * self.config["max_tracks_velocity"]

        # Decide on next state
        flipper_positions = self.flipper_pos_dict[self.current_state]

        # Calculate flippers vel:
        fl_flipper_pos = flipper_positions[0]
        fr_flipper_pos = flipper_positions[1]
        rl_flipper_pos = flipper_positions[2]
        rr_flipper_pos = flipper_positions[3]

        return tracks_vel_ang, tracks_vel_lin, fl_flipper_pos, fr_flipper_pos, rl_flipper_pos, rr_flipper_pos

    def decide_next_state(self, dpad_event_list):
        right_dpad_event, left_dpad_event, down_dpad_event, up_dpad_event = dpad_event_list

        if self.current_state == "N":
            if up_dpad_event:
                self.current_state = "AF"
            elif down_dpad_event:
                self.current_state = "DF"
        # Up
        elif self.current_state == "AF":
            if up_dpad_event:
                self.current_state = "US"
            elif right_dpad_event:
                self.current_state = "AR"
            elif left_dpad_event:
                self.current_state = "N"
        elif self.current_state == "US":
            if right_dpad_event:
                self.current_state = "AR"
        elif self.current_state == "AR":
            if right_dpad_event:
                self.current_state = "N"
        # Down
        elif self.current_state == "DF":
            if down_dpad_event:
                self.current_state = "DS"
            elif right_dpad_event:
                self.current_state = "DR"
            elif left_dpad_event:
                self.current_state = "N"
        elif self.current_state == "DS":
            if right_dpad_event:
                self.current_state = "DR"
        elif self.current_state == "DR":
            if right_dpad_event:
                self.current_state = "N"

    def publish_commands(self, commands, publish_flippers=True, publish_tracks=True):
        tracks_vel_ang, tracks_vel_lin, fl_flipper_pos, fr_flipper_pos, rl_flipper_pos, rr_flipper_pos = commands

        if publish_tracks:
            # Publish tracks vel
            tracks_vel_msg = Twist()
            tracks_vel_msg.linear.x = tracks_vel_lin
            tracks_vel_msg.linear.y = 0
            tracks_vel_msg.angular.z = tracks_vel_ang
            self.tracks_vel_publisher.publish(tracks_vel_msg)

        if publish_flippers:
            # Publish flippers vel
            flippers_msg = Float64MultiArray()
            flippers_msg.data = [fl_flipper_pos, fr_flipper_pos, rl_flipper_pos, rr_flipper_pos]

            self.flippers_pos_publisher.publish(flippers_msg)

    def publish_plan(self, processed_path, vel_ub):
        # Publish tracks vel
        joy_plan_msg = Path()
        joy_plan_msg.header.frame_id = self.config["root_frame"]
        for p in processed_path:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self.config["root_frame"]
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = p
            joy_plan_msg.poses.append(pose_msg)
        self.joy_plan_publisher.publish(joy_plan_msg)

        vel_ub_msg = Twist()
        vel_ub_msg.linear.x = vel_ub
        vel_ub_msg.linear.y = vel_ub
        self.cmd_vel_ub_publisher.publish(vel_ub_msg)

    def publish_txt_info(self):
        strs = []
        strs.append("Js state: {} \n".format(self.current_state))
        strs.append("Js active: {} \n".format(bool(self.deadmans_button_state)))
        strs.append("Js control mode: {} \n".format(self.control_mode))

        txt = ""
        for s in strs:
            txt += s

        msg = String()
        msg.data = txt
        self.text_publisher.publish(msg)

    def publish_state(self):
        msg = String()
        msg.data = self.current_state
        self.state_publisher.publish(msg)

def main():
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) == 2:
        config_name = myargv[1]
    else:
        config_name = "marv_teleop_config.yaml"

    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/{}".format(config_name)), 'r') as f:
        tracker_config = yaml.load(f, Loader=yaml.FullLoader)

    tracker = MarvTeleop(tracker_config)
    tracker.loop()

if __name__=="__main__":
    main()

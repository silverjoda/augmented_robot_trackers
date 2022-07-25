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
from tf.transformations import euler_from_quaternion, quaternion_matrix
import src.utilities as utilities
from marv_msgs.msg import Float64MultiArray as MarvFloat64MultiArray

class MarvHelicopter:
    def __init__(self, config):
        self.config = config

        self.flipper_button_event_translator = utilities.ButtonToEventTranslator(2, 0)
        self.flipper_button_state = self.flipper_button_event_translator.idx

        rear_offset = np.pi / 2
        self.virtual_flipper_positions = np.array([0, 1.57, 3.14 + rear_offset, 3.14 + 1.57 + rear_offset])

        self.init_ros("art_marv_helicopter")

    def step_virtual_flipper_states(self, dirs):
        self.virtual_flipper_positions += np.array(dirs) * self.config["flipper_step_constant"]

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
        rospy.loginfo("Starting marv helicopter")
        rospy.init_node(name)

        self.ros_rate = rospy.Rate(self.config["ros_rate"])
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.config["tf_buffer_size"]))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.joy_lock = threading.Lock()
        self.joy_data = None
        self.trigger_flipper = False

        self.flippers_pos_publisher = rospy.Publisher("marv/flippers_position_controller/cmd_vel",
                                                      MarvFloat64MultiArray,
                                                      queue_size=1)

        self.flippers_max_torque_publisher = rospy.Publisher("marv/flippers_torque_controller",
                                                             MarvFloat64MultiArray,
                                                             queue_size=1)

        self.tracks_vel_publisher = rospy.Publisher("/X1/cmd_vel",
                                                    Twist,
                                                    queue_size=1)

        rospy.Subscriber("joy", Joy, self._ros_joy_callback, queue_size=1)

        time.sleep(0.1)

    def _ros_joy_callback(self, data):
        with self.joy_lock:
            self.joy_data = data

            switch_event, idx = self.flipper_button_event_translator.update(data.buttons[0])
            if switch_event:
                rospy.loginfo("Triggering flip")
                self.trigger_flipper = True

    def loop(self):
        while not rospy.is_shutdown():
            with self.joy_lock:
                joy_data = deepcopy(self.joy_data)

            if joy_data is not None:
                processed_commands = self._process(joy_data)
                self.publish_commands(processed_commands)

            self.ros_rate.sleep()

    def _process(self, joy_data):
        LB = joy_data.buttons[4]
        RB = joy_data.buttons[5]
        RT = (-joy_data.axes[5] + 1.) * 0.5
        LT = (-joy_data.axes[2] + 1.) * 0.5

        # Calculate tracks vel:
        tracks_vel_lin = RT * self.config["max_tracks_velocity"] * (1 - RB * 2)
        tracks_vel_ang = LT * self.config["max_tracks_velocity"] * (1 - LB * 2)

        # Calculate forward flipper movement and flipper switch
        self.step_virtual_flipper_states([joy_data.axes[4], joy_data.axes[4], joy_data.axes[4], joy_data.axes[4]])

        #if self.trigger_flipper:

        # Calculate flippers vel:
        fl_flipper_pos = self.virtual_flipper_positions[0]
        fr_flipper_pos = self.virtual_flipper_positions[1]
        rl_flipper_pos = self.virtual_flipper_positions[2]
        rr_flipper_pos = self.virtual_flipper_positions[3]

        return tracks_vel_ang, tracks_vel_lin, fl_flipper_pos, fr_flipper_pos, rl_flipper_pos, rr_flipper_pos

    def publish_commands(self, commands):
        tracks_vel_ang, tracks_vel_lin, fl_flipper_pos, fr_flipper_pos, rl_flipper_pos, rr_flipper_pos = commands

        # Publish tracks vel
        tracks_vel_msg = Twist()
        tracks_vel_msg.linear.x = tracks_vel_lin
        tracks_vel_msg.linear.y = 0
        tracks_vel_msg.angular.z = tracks_vel_ang
        self.tracks_vel_publisher.publish(tracks_vel_msg)

        # Publish flippers vel
        flippers_msg = Float64MultiArray()
        flippers_msg.data = [fl_flipper_pos, fr_flipper_pos, rl_flipper_pos, rr_flipper_pos]

        self.flippers_pos_publisher.publish(flippers_msg)

def main():
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) == 2:
        config_name = myargv[1]
    else:
        config_name = "marv_helicopter_config.yaml"

    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/{}".format(config_name)), 'r') as f:
        tracker_config = yaml.load(f, Loader=yaml.FullLoader)

    tracker = MarvHelicopter(tracker_config)
    tracker.loop()

if __name__=="__main__":
    main()

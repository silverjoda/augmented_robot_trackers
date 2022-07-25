#!/usr/bin/env python
import os
import threading
import time
from copy import deepcopy

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_srvs.srv import Trigger, TriggerResponse, SetBool, TriggerRequest
from spot_msgs.msg import BehaviorFaultState

class SpotTeleop:
    def __init__(self, config):
        # TODO: Read behavior topic once and determine if spot is sitting or standing initially
        self.current_state = "sit" # sit/stand
        self.stairs_mode = False
        self.latest_stance_timestamp = time.time()

        self.config = config
        self.init_ros(self.config["node_name"])

        rospy.loginfo("Starting spot teleop")

    def init_ros(self, name):
        rospy.init_node(name)

        self.ros_rate = rospy.Rate(self.config["ros_rate"])

        self.joy_lock = threading.Lock()
        self.behavior_status_lock = threading.Lock()
        self.joy_data = None
        self.behavior_status_data = None

        rospy.Subscriber(self.config["joy_in"],
                         Joy,
                         self._ros_joy_callback, queue_size=1)

        rospy.Subscriber("/spot/status/behavior_faults",
                         BehaviorFaultState,
                         self._ros_behaviorfaultstate_callback, queue_size=1)

        self.cmd_vel_publisher = rospy.Publisher(self.config["cmd_vel_out"],
                                                    Twist,
                                                    queue_size=1)

        time.sleep(2)

        rospy.loginfo("Spot teleop node {}: initialized ros".format(self.config["node_name"]))

    def _ros_joy_callback(self, data):
        with self.joy_lock:
            self.joy_data = data

    def _ros_behaviorfaultstate_callback(self, data):
        with self.behavior_status_lock:
            self.behavior_status_data = data

    def loop(self):
        while not rospy.is_shutdown():
            with self.joy_lock:
                joy_data = deepcopy(self.joy_data)

            if joy_data is not None:
                self.process_joy_and_publish(joy_data)

            self.ros_rate.sleep()

    def process_joy_and_publish(self, joy_data):
        # Dead man's button
        dmb = joy_data.buttons[5]

        # Calculate command velocity
        lin_x = joy_data.axes[1] * self.config["vel_x"]
        lin_y = joy_data.axes[0] * self.config["vel_y"]
        ang_z = joy_data.axes[3] * self.config["ang_vel_z"]

        # Only publish if we are in stand mode
        if dmb == 1: # if self.current_state == "stand"
            self.publish_cmd_vel(lin_x, lin_y, ang_z)

        # Change state if correct button
        if joy_data.buttons[0]:
            allow_stance_change = time.time() - self.latest_stance_timestamp > self.config["stance_change_cooldown"]
            if self.current_state == "sit" and allow_stance_change:
                if self.set_spot_stance("stand").success:
                    self.current_state = "stand"
                    self.latest_stance_timestamp = time.time()
            elif allow_stance_change:
                if self.set_spot_stance("sit").success:
                    self.current_state = "sit"
                    self.latest_stance_timestamp = time.time()

        # Set stairs mode
        if joy_data.buttons[3]:
            allow_stance_change = time.time() - self.latest_stance_timestamp > self.config["stance_change_cooldown"]
            if self.current_state == "stand" and allow_stance_change:
                if self.set_stairs_mode().success:
                    self.latest_stance_timestamp = time.time()

        # Self right
        if joy_data.buttons[1]:
            allow_stance_change = time.time() - self.latest_stance_timestamp > self.config["stance_change_cooldown"]
            if allow_stance_change:
                self.set_spot_stance("self_right")
                self.latest_stance_timestamp = time.time()

        # Power on
        if joy_data.buttons[7]:
            allow_stance_change = time.time() - self.latest_stance_timestamp > self.config["stance_change_cooldown"]
            if allow_stance_change:
                self.power_on()
                self.latest_stance_timestamp = time.time()

        # Power off
        if joy_data.buttons[6]:
            allow_stance_change = time.time() - self.latest_stance_timestamp > self.config["stance_change_cooldown"]
            if allow_stance_change:
                self.power_off()
                self.latest_stance_timestamp = time.time()

    def publish_cmd_vel(self, lin_x, lin_y, ang_z):
        # Publish tracks vel
        tracks_vel_msg = Twist()
        tracks_vel_msg.linear.x = lin_x
        tracks_vel_msg.linear.y = lin_y
        tracks_vel_msg.angular.z = ang_z
        self.cmd_vel_publisher.publish(tracks_vel_msg)

    def set_spot_stance(self, stance_string):
        full_service_name = "/spot/" + stance_string
        # Wait for last cmd_vel msg to have gone
        time.sleep(0.2)
        try:
            rospy.wait_for_service(full_service_name, timeout=1)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Spot teleop: Timeout while waiting for {} service: {}".format(stance_string, e))
            return TriggerResponse(success=False)

        try:
            service_proxy = rospy.ServiceProxy(full_service_name, Trigger)
            res = service_proxy(TriggerRequest())
            rospy.loginfo("Spot teleop: Politely asking spot to {}".format(stance_string))
        except rospy.ServiceException as e:
            rospy.logwarn("Spot teleop: Setting {} failed, err: {}".format(stance_string, e))
            return TriggerResponse(success=False)
        return res

    def set_stairs_mode(self):
        full_service_name = "/spot/stair_mode"
        # Wait for last cmd_vel msg to have gone
        time.sleep(0.2)
        try:
            rospy.wait_for_service(full_service_name, timeout=1)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Spot teleop: Timeout while waiting for stairs_mode service: {}".format(e))
            return False

        try:
            service_proxy = rospy.ServiceProxy(full_service_name, SetBool)
            res = service_proxy(not self.stairs_mode)
            self.stairs_mode = not self.stairs_mode
            rospy.loginfo("Spot teleop: Switching stairs mode to {}".format(self.stairs_mode))
        except rospy.ServiceException as e:
            rospy.logwarn("Spot teleop: Switching stairs mode to {} failed, err: {}".format(self.stairs_mode, e))
            return False
        return res

    def power_on(self):
        full_service_name = "/spot/power_on"
        # Wait for last cmd_vel msg to have gone
        time.sleep(0.2)
        rospy.loginfo("Spot teleop: Registered power on attempt from joystick")

        try:
            rospy.wait_for_service(full_service_name, timeout=1)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Spot teleop: Timeout while waiting for power_on service: {}".format(e))
            return False

        try:
            service_proxy = rospy.ServiceProxy(full_service_name, Trigger)
            res = service_proxy(TriggerRequest())
            rospy.loginfo("Spot teleop: Attempt to power on: {}.  Is the red button pressed?".format(res))
        except rospy.ServiceException as e:
            rospy.logwarn("Spot teleop: Attempt to power on failed, err: {}".format(e))
            return False
        return res

    def power_off(self):
        full_service_name = "/spot/power_off"
        # Wait for last cmd_vel msg to have gone
        time.sleep(0.2)

        try:
            rospy.wait_for_service(full_service_name, timeout=1)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Spot teleop: Timeout while waiting for power_on service: {}".format(e))
            return False

        try:
            service_proxy = rospy.ServiceProxy(full_service_name, Trigger)
            res = service_proxy(TriggerRequest())
            rospy.loginfo("Spot teleop: Attempt to power off: {}".format(res))
        except rospy.ServiceException as e:
            rospy.logwarn("Spot teleop: Attempt to power off failed, err: {}".format(e))
            return False
        return res

def main():
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/spot_teleop_config.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    spotteleop = SpotTeleop(config)
    spotteleop.loop()

if __name__=="__main__":
    main()

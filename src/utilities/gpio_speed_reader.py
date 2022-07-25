#!/usr/bin/env python3
import os
import sys
import time

import RPi.GPIO as GPIO
import rospy
from std_msgs.msg import Float64
import threading

class SpeedReader:
    def __init__(self, config):
        self.config = config

        self.setup_gpio()
        self.init_ros()

    def setup_gpio(self):
        self.sensor_update_lock = threading.Lock()

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.config["sensor_gpio_pin"], GPIO.IN)
        GPIO.add_event_detect(self.config["sensor_gpio_pin"], GPIO.RISING, callback=self.sensor_update_callback,
                              bouncetime=7)

        self.trigger_counts = 0

    def init_ros(self):
        rospy.init_node("speed_reader_node")
        self.rate = rospy.Rate(self.config["ros_rate"])
        self.velocity_pulisher = rospy.Publisher("wheel_speed", Float64, queue_size=10)

    def sensor_update_callback(self, _):
        with self.sensor_update_lock:
            self.trigger_counts += 1

    def loop(self):
        while not rospy.is_shutdown():
            with self.sensor_update_lock:
                current_wheel_speed = self.trigger_counts / self.config["speed_sensor_scalar"]
                self.trigger_counts = 0

            # Publish ros message
            self.velocity_pulisher.publish(Float64(data=current_wheel_speed))
            self.rate.sleep()

def main():
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) == 2:
        config_name = myargv[1]
    else:
        config_name = "gpio_speed_reader.yaml"

    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/{}".format(config_name)), 'r') as f:
        speed_reader_config = yaml.load(f, Loader=yaml.FullLoader)

    sr = SpeedReader(speed_reader_config)
    sr.loop()

if __name__=="__main__":
    main()

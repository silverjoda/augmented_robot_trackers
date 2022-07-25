import os
import subprocess
import time

import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus


class TimeSyncChecker:
    def __init__(self, config):
        self.config = config
        self.init_ros()

    def init_ros(self):
        rospy.init_node(self.config["node_name"])
        self.ros_rate = rospy.Rate(self.config["ros_rate"])
        self.diag_publisher = rospy.Publisher('/diagnostics', DiagnosticArray, queue_size=1)
        time.sleep(2)

    def loop(self):
        timesync_info = self.check_time_sync()
        self.publish_diagnostic_msg(timesync_info)
        self.ros_rate.sleep()

    def check_time_sync(self):
        if self.config["server_hostname"] is not None:
            server_con_info = self.config["server_hostname"]
        else:
            server_con_info = self.config["server_ip"]

        bashCommand = "ntpdate -q {}".format(server_con_info)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        split_msg = output.split()
        info_dict = {"server": split_msg[1], "stratum": split_msg[3], "offset": split_msg[5], "delay": split_msg[7]}
        return info_dict

    def publish_diagnostic_msg(self, timesync_info):
        arr = DiagnosticArray()
        txt_msg = "Server: {}, stratum: {}, offset: {}, delay: {}".format(timesync_info["server"],
                                                                          timesync_info["stratum"],
                                                                          timesync_info["offset"],
                                                                          timesync_info["delay"])
        arr.status = [
            DiagnosticStatus(name='timesync ntpdate_message', message=txt_msg)
        ]

        self.diag_publisher.publish(arr)

def main():
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/timesync_config.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tsc = TimeSyncChecker(config)
    tsc.loop()

if __name__=="__main__":
    main()

import logging
import os
import sys
import threading
import time

from copy import deepcopy
import rospy
import torch as T
from sensor_msgs.msg import Image
from nifti_robot_driver_msgs.msg import (FlippersStateStamped)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
T.set_num_threads(1)

class FlipperDataSetFromBagMaker:
    def __init__(self, config):
        self.config = config
        self.init_ros()

    def init_ros(self):
        rospy.init_node(self.config["node_name"])
        self.ros_rate = rospy.Rate(self.config["ros_rate"])

        rospy.set_param("/use_sim_time", True)

        self.tracker_flipper_pos_bagfile_lock = threading.Lock()
        self.tracker_flipper_pos_bagfile_data = None

        self.trav_vis_hm_lock = threading.Lock()
        self.trav_vis_hm_data = None

        # Reading current target flipper positions of path_follower
        rospy.Subscriber(self.config["flippers_state_out_bagfile"],
                         FlippersStateStamped,
                         self._ros_tracker_flippers_pos_bagfile_callback, queue_size=1)

        # Reading trav vis map tp make pc out of
        rospy.Subscriber(self.config["trav_vis_hm_out"],
                         Image,
                         self._ros_trav_vis_hm_callback, queue_size=1)

        time.sleep(2)

        rospy.loginfo("Marv out of ros training node {}: initialized ros".format(self.config["node_name"]))

    def _ros_tracker_flippers_pos_bagfile_callback(self, data):
        with self.tracker_flipper_pos_bagfile_lock:
            self.tracker_flipper_pos_bagfile_data = data

    def _ros_trav_vis_hm_callback(self, data):
        with self.trav_vis_hm_lock:
            self.trav_vis_hm_data = data

    def play_bag(self, bf):
        def _play_bag():
            os.system("rosbag play {} --clock -r {} > /dev/null 2>&1".format(bf, self.config["bagfile_playrate"]))

        # Start playing bag
        thread_obj = threading.Thread(target=_play_bag)
        thread_obj.start()
        return thread_obj

    def make_dataset(self, w):
        # Read available bags
        bag_filenames = [os.path.join(self.config["path_to_bagfiles"], f)
                         for f in os.listdir(self.config["path_to_bagfiles"])]

        # For each bagfile, reset path_follower, run bagfile with increased rate and evaluate path_follower
        for bf in bag_filenames:
            # Play back bag in different thread
            bf_thread_obj = self.play_bag(bf)

            # Evaluate path_follower
            data = self.record_data(bf_thread_obj)


    def record_data(self, bf_thread_obj):
        data = {}
        while bf_thread_obj.is_alive():
            # Read flipper positions from bagfile
            with self.tracker_flipper_pos_bagfile_lock:
                flipper_pos_bag_data = deepcopy(self.tracker_flipper_pos_bagfile_data)

            if flipper_pos_bag_data is None:
                continue

            bag_flipper_list = [flipper_pos_bag_data.frontLeft, flipper_pos_bag_data.frontRight,
                                flipper_pos_bag_data.rearLeft, flipper_pos_bag_data.rearRight]

            time.sleep(1./self.config["ros_rate"])

        print("Bag done")
        time.sleep(0.1)
        return data


if __name__=="__main__":
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/make_flipper_dataset_from_bag.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists("agents"):
        os.makedirs("agents")

    # ID of this session
    config["session_ID"] = "TRN"

    dsmaker = FlipperDataSetFromBagMaker(config)
    dsmaker.make_dataset()



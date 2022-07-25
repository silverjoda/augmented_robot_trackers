
import logging
import os
import pickle
import sys
import threading
import time

import cma
from copy import deepcopy
import numpy as np
import rospy
import torch as T
from augmented_robot_trackers.srv import GetTrackerParams, SetTrackerParams
from marv_msgs.msg import Float64MultiArray as MarvFloat64MultiArray
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger, TriggerRequest
from src.policies.tracker_nn import TrackerNN
from nifti_robot_driver_msgs.msg import (FlippersStateStamped)
import roslaunch

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
T.set_num_threads(1)

class SupervisedMARVTrainer:
    def __init__(self, config):
        self.config = config
        self.init_ros()

    def init_ros(self):
        rospy.init_node(self.config["node_name"])
        self.ros_rate = rospy.Rate(self.config["ros_rate"])

        rospy.set_param("/use_sim_time", True)

        self.tracker_state_lock = threading.Lock()
        self.tracker_state_data = None

        self.tracker_flipper_pos_lock = threading.Lock()
        self.tracker_flipper_pos_data = None

        self.tracker_flipper_pos_bagfile_lock = threading.Lock()
        self.tracker_flipper_pos_bagfile_data = None

        self.trav_vis_hm_lock = threading.Lock()
        self.trav_vis_hm_data = None

        # Reading current state of path_follower
        rospy.Subscriber(self.config["tracker_state_out"],
                         String,
                         self._ros_tracker_state_callback, queue_size=1)

        # Reading current target flipper positions of path_follower
        rospy.Subscriber(self.config["flippers_position_control_out"],
                         MarvFloat64MultiArray,
                         self._ros_tracker_flippers_pos_callback, queue_size=1)

        # Reading current target flipper positions of path_follower
        rospy.Subscriber(self.config["flippers_state_out_bagfile"],
                         FlippersStateStamped,
                         self._ros_tracker_flippers_pos_bagfile_callback, queue_size=1)

        # Reading trav vis map tp make pc out of
        rospy.Subscriber(self.config["trav_vis_hm_out"],
                         Image,
                         self._ros_trav_vis_hm_callback, queue_size=1)

        self.time_reset_publisher = rospy.Publisher("/reset_time",
                                                    Empty,
                                                    queue_size=1)

        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)

        time.sleep(2)

        rospy.loginfo("Marv bbx training node {}: initialized ros".format(self.config["node_name"]))

    def _ros_tracker_state_callback(self, data):
        with self.tracker_state_lock:
            self.tracker_state_data = data

    def _ros_tracker_flippers_pos_callback(self, data):
        with self.tracker_flipper_pos_lock:
            self.tracker_flipper_pos_data = data

    def _ros_tracker_flippers_pos_bagfile_callback(self, data):
        with self.tracker_flipper_pos_bagfile_lock:
            self.tracker_flipper_pos_bagfile_data = data

    def _ros_trav_vis_hservicem_callback(self, data):
        with self.trav_vis_hm_lock:
            self.trav_vis_hm_data = data

    def _set_tracker_params(self, params):
        msg = Float64MultiArray()
        msg.data = params
        rospy.wait_for_service("set_tracker_params")
        set_tracker_params_service = rospy.ServiceProxy("set_tracker_params", SetTrackerParams)
        resp = set_tracker_params_service(msg)
        print("Setting params service results: {}".format(resp))

    def _get_tracker_params(self):
        get_tracker_params_service = rospy.ServiceProxy("get_tracker_params", GetTrackerParams)
        return get_tracker_params_service().params.data

    def _reset_tracker(self):
        tracker_reset_service = rospy.ServiceProxy("tracker_reset_service", Trigger)
        tracker_reset_service()

    def launch_system(self):
        self.launch = roslaunch.parent.ROSLaunchParent(self.uuid, [
            "/home/tim/SW/cras_subt/src/augmented_robot_trackers/launch/tradr_tracker_debug_utilities.launch"])
        self.launch.start()

    def kill_system(self):
        self.launch.shutdown()

    def objective(self, w):
        self.launch_system()

        # Set current path_follower params
        self._set_tracker_params(w)

        # Read available bags
        bag_filenames = [os.path.join(self.config["path_to_bagfiles"], f)
                         for f in os.listdir(self.config["path_to_bagfiles"])]

        # For each bagfile, reset path_follower, run bagfile with increased rate and evaluate path_follower
        cum_rew = 0
        for bf in bag_filenames:
            # Reset path_follower
            self._reset_tracker()

            # Play back bag in different thread
            bf_thread_obj = self.play_bag(bf)

            # Evaluate path_follower
            rew = self.evaluate_episode(bf_thread_obj)
            cum_rew += rew

            # Reset ros time
            self.time_reset_publisher.publish(Empty())

        print("REW: {}".format(cum_rew))

        self.kill_system()

        return -cum_rew

    def play_bag(self, bf):
        def _play_bag():
            os.system("rosbag play {} --clock -r {} > /dev/null 2>&1".format(bf, self.config["bagfile_playrate"]))

        # Start playing bag
        thread_obj = threading.Thread(target=_play_bag)
        thread_obj.start()
        return thread_obj

    def evaluate_episode(self, bf_thread_obj):
        cum_rew = 0
        current_tracker_state = "NEUTRAL"

        while bf_thread_obj.is_alive():
            # Read state from path_follower
            with self.tracker_state_lock:
                tracker_state = deepcopy(self.tracker_state_data)

            # Read flipper position from path_follower
            with self.tracker_flipper_pos_lock:
                flipper_pos_data = deepcopy(self.tracker_flipper_pos_data)

            # Read flipper positions from bagfile
            with self.tracker_flipper_pos_bagfile_lock:
                flipper_pos_bag_data = deepcopy(self.tracker_flipper_pos_bagfile_data)

            if tracker_state is None or flipper_pos_data is None or flipper_pos_bag_data is None:
                continue

            bag_flipper_list = [flipper_pos_bag_data.frontLeft, flipper_pos_bag_data.frontRight,
                                flipper_pos_bag_data.rearLeft, flipper_pos_bag_data.rearRight]

            # Evaluate mse penalty
            mse = np.mean(np.square(np.array(flipper_pos_data.data) - np.array(bag_flipper_list)))

            # Add the negative of loss
            cum_rew -= mse

            #self.ros_rate.sleep() # Issue: When bag stops playing, rate hangs
            time.sleep(1./self.config["ros_rate"])

        print("Bag done")
        time.sleep(0.1)
        return cum_rew

    def train(self):
        print("Starting training with cma")
        t1 = time.time()

        w = [0] * 13 #self._get_tracker_params()
        es = cma.CMAEvolutionStrategy(w, self.config["cma_std"])

        print('N_params: {}'.format(len(w)))

        it = 0
        try:
            while not es.stop():
                it += 1
                if it > self.config["iters"]:
                    break
                X = es.ask()
                es.tell(X, [self.objective(x) for x in X])
                es.disp()

        except KeyboardInterrupt:
            print("User interrupted process.")

        self._set_tracker_params(es.result.xbest)
        self.save_params(es.result.xbest, "agents/{}.pkl".format(self.config["session_ID"]))
        print("Saved agent, agents/{}.pkl".format(self.config["session_ID"]))

        t2 = time.time()
        print("Training time: {}".format(t2 - t1))
        print(self.config)

        return es.result.fbest

    def train_NN(self):
        print("Starting training NN")
        t1 = time.time()

        # Make same policy network as path_follower
        tracker_nn = TrackerNN(self.config)

        # Initialize optimizer and lossfun
        adam_opt = T.optim.Adam(tracker_nn.parameters(), lr=self.config["lr"])
        lossfun = T.nn.MSELoss()

        # Create supervised dataset from rosbags
        x_hm_trn, x_vec_trn, y_trn, x_hm_eval, x_vec_eval, y_eval = self.make_dataset_from_rosbags()

        # Train on dataset
        n_examples = x_hm_trn.shape[0]
        for ep in range(self.config["n_epochs"]):
            shuffled_indeces = np.random.choice(np.arange(n_examples), n_examples, replace=False)

            for it in range(np.floor(n_examples / self.config["batchsize"])):
                cur_indeces = shuffled_indeces[it * self.config["bachsize"] : it * self.config["bachsize"] + self.config["bachsize"]]
                X_hm = x_hm_trn[cur_indeces]
                X_vec = x_vec_trn[cur_indeces]
                Y_ = y_trn[cur_indeces]
                Y = tracker_nn.forward(X_hm, X_vec)
                loss = lossfun(Y, Y_)
                adam_opt.zero_grad()
                loss.backward()
                adam_opt.step()
                print("Epoch: {}, iter: {}, loss: {}".format(ep, it, loss))

            # Calc eval error after epoch
            loss_eval = lossfun(tracker_nn.forward(x_hm_eval, x_vec_eval), y_eval)
            print("EVAL: Epoch: {}, loss: {}".format(ep, loss_eval))

        eval_error = 0

        t2 = time.time()
        print("Training time: {}".format(t2 - t1))
        print(self.config)

        return eval_error

    def make_dataset_from_rosbags(self):
        bag_filenames = [os.path.join(self.config["path_to_bagfiles"], f)
                         for f in os.listdir(self.config["path_to_bagfiles"])]

        x_hm = []
        x_vec = []
        y = []

        for bf in bag_filenames:
            bag_done = False

            # Reset path_follower
            self.tracker_reset_service(TriggerRequest)

            # Start playing bag
            os.system("rosbag play {} -r {} --clock > /dev/null 2>&1 &".format(bf, self.config["bagfile_playrate"]))

            # loop and record data into dataset
            with self.tracker_flipper_pos_bagfile_lock:
                time_offset = rospy.Time.now() - self.tracker_flipper_pos_bagfile_data.header.stamp

            while not bag_done:
                # Read flipper position from path_follower
                with self.tracker_flipper_pos_lock:
                    flipper_pos_data = deepcopy(self.tracker_flipper_pos_data)

                # Read flipper positions from bagfile
                with self.tracker_flipper_pos_bagfile_lock:
                    flipper_pos_bag_data = deepcopy(self.tracker_flipper_pos_bagfile_data)

                # Read height map image from path_follower
                with self.tracker_flipper_pos_bagfile_lock:
                    trav_vis_hm_data = deepcopy(self.trav_vis_hm_data)
                    cv_image = self.bridge.imgmsg_to_cv2(trav_vis_hm_data, "passthrough")
                    np_hm = np.asarray(cv_image)

                # TODO: Find out how to convert point cloud to image
                # TODO: Combine hm and vector data and append to dataset
                self.ros_rate.sleep()

                # If message old, means that bagfile must have stopped playing
                bag_done = (rospy.Time.now() - time_offset - flipper_pos_bag_data.header.stamp > (
                            3 / self.config["ros_rate"]))

        # Split into trn and eval
        n_examples = len(x_hm)
        rnd_indeces = np.random.choice(np.arange(n_examples), n_examples, replace=False)
        split_point = int(n_examples) * self.config["trn_val_ratio"]
        trn_indeces = rnd_indeces[:split_point]
        eval_indeces = rnd_indeces[split_point:]

        x_hm_trn = T.Tensor(x_hm[trn_indeces])
        x_vec_trn = T.Tensor(x_vec[trn_indeces])
        y_trn = T.Tensor(y[trn_indeces])

        x_hm_eval = T.Tensor(x_hm[eval_indeces])
        x_vec_eval = T.Tensor(x_vec[eval_indeces])
        y_eval = T.Tensor(y[eval_indeces])

        return x_hm_trn, x_vec_trn, y_trn, x_hm_eval, x_vec_eval, y_eval

    def save_params(self, params, path):
        if not os.path.exists("agents"):
            os.makedirs("agents")
        if path is None:
            path = "agents/saved_params.pkl"
        pickle.dump(params, open(path, "wb"))

    def load_params(self, path):
        return pickle.load(open(path, "rb"))

if __name__=="__main__":
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/train_tracker_supervised.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists("agents"):
        os.makedirs("agents")

    # ID of this session
    config["session_ID"] = "TRN"

    trainer = SupervisedMARVTrainer(config)

    if config["train"]:
        trainer.train()

    if config["test"]:
        print("Loading policy")
        #policy.load("agents/flippers_ascent.pickle")

        #print("Testings")
        #avg_rew = test_agent(env, policy)

        #print("Avg test rew: {}".format(avg_rew))



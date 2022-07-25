import os
import subprocess
import time
from collections import namedtuple
from copy import deepcopy

import numpy as np
import roslaunch
import rospkg
import rospy
import tf2_ros
from augmented_robot_trackers.srv import ResetEnv, StepEnv, ResetEnvResponse, StepEnvResponse
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Float64, Bool
from tf.transformations import euler_from_quaternion, quaternion_matrix
import tf

class MARVRocketBalEnv:
    project_root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    def __init__(self, config):
        self.config = config

        self.act_history_horizon = 3
        self.obs_history_horizon = 3

        # Environment variables
        self.act_dim = 3
        self.obs_dim = 1 + self.act_history_horizon * self.act_dim + self.obs_history_horizon * 4

        self.episode_ctr = 0

        self.init_ros()

    def init_ros(self):
        print("{} initializing ros".format(self.__class__.__name__))
        rospy.init_node("marv_rocket_bal_env_node")

        # For aloam launching
        rospack = rospkg.RosPack()
        self.aloam_path = os.path.join(rospack.get_path('augmented_robot_trackers'), "launch/marv_sim/sim_marv_aloam.launch")
        self.aloam_uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.aloam_uuid)

        self.ros_rate = rospy.Rate(self.config["ros_rate"])
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.config["tf_buffer_size"]))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.tracks_vel_publisher = rospy.Publisher("/X1/cmd_vel",
                                                       Twist,
                                                       queue_size=3)

        self.flippers_pos_fl_publisher = rospy.Publisher("/X1/flippers_cmd_pos/front_left",
                                                         Float64,
                                                         queue_size=1)
        self.flippers_pos_fr_publisher = rospy.Publisher("/X1/flippers_cmd_pos/front_right",
                                                         Float64,
                                                         queue_size=1)
        self.flippers_pos_rl_publisher = rospy.Publisher("/X1/flippers_cmd_pos/rear_left",
                                                         Float64,
                                                         queue_size=1)
        self.flippers_pos_rr_publisher = rospy.Publisher("/X1/flippers_cmd_pos/rear_right",
                                                         Float64,
                                                         queue_size=1)

        self.rew_publisher = rospy.Publisher("/marv_rocket_bal_reward",
                                     Float64,
                                     queue_size=1)

        self.debug_tf_publisher = tf.TransformBroadcaster()

        self.step_service = rospy.Service("step_service", StepEnv, self.step_handler)
        self.reset_service = rospy.Service("reset_service", ResetEnv, self.reset_handler)

    def step_handler(self, request):
        obs, rew, done, _ = self.step(request.act.data)
        return StepEnvResponse(obs=Float64MultiArray(data=obs), rew=Float64(data=rew), done=Bool(data=done))

    def reset_handler(self, _):
        obs = self.reset()
        return ResetEnvResponse(obs=Float64MultiArray(data=obs))

    def get_flipper_positions(self):
        flipper_name_list = ["front_left_flipper",
                             "front_right_flipper",
                             "rear_left_flipper",
                             "rear_right_flipper"]

        flipper_dict = {}
        for fn in flipper_name_list:
            try:
                trans = self.tf_buffer.lookup_transform("X1",
                                                        "X1/" + fn,
                                                        rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
                rospy.logwarn("{}: Getting flipper pos transform of {} resulted in err: {}".format(MARVRocketBalEnv, fn, err))
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

        flipper_tuple = [flipper_dict[fn] for fn in flipper_name_list]
        return flipper_tuple

    def get_robot_pose(self):
        # Get pose using TF
        try:
            trans = self.tf_buffer.lookup_transform("world",
                                                    "X1",
                                                    rospy.Time(0),
                                                    rospy.Duration(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn("Get_robot_pose_dict: TRANSFORMATION ERROR, err: {}".format(err))
            return None, None, None

        # Translation
        pos = trans.transform.translation

        # Orientation
        quat = trans.transform.rotation
        rot_mat = quaternion_matrix((quat.x, quat.y, quat.z, quat.w))

        # self.debug_tf_publisher.sendTransform((pos.x, pos.y, pos.z),
        #                              (quat.x, quat.y, quat.z, quat.w),
        #                              rospy.Time(0),
        #                              "X1/debug",
        #                              "world")

        verticality = rot_mat[2, 0]
        pitch_dir = np.sign(rot_mat[0, 0])
        pitch_normed = verticality
        if pitch_dir < 0:
            pitch_normed = + (2 - verticality)
        pitch = pitch_normed * np.pi * 0.5

        pose_dict = {"position": pos,
                     "quat": quat,
                     "matrix": rot_mat,
                     "pitch" : pitch,
                     "pitch_normed" : pitch_normed,
                     "verticality" : verticality}

        state = [pos.x, (pose_dict["pitch_normed"] - 1) * 2]

        return state, pose_dict

    def ros_get_obs(self):
        # Get flipper positions
        flipper_tuple = self.get_flipper_positions()

        # Get body 6dof pose
        state, pose_dict = self.get_robot_pose()
        self.state_hist.extend(state + flipper_tuple[0:1] + flipper_tuple[2:3])
        del self.state_hist[0:4]

        # Check if any are None, then return none
        if any(dat is None for dat in [flipper_tuple, pose_dict]):
            return None, None

        time_feature = (float(self.step_ctr) / self.config["max_steps"]) * 2 - 1

        # Assemble complete obs vec
        obs = np.concatenate([self.act_hist, self.state_hist, [time_feature]])

        return obs, pose_dict

    def ros_publish_action(self, act):
        # Publish default flipper positions and sleep
        self.flippers_pos_fl_publisher.publish(Float64(data=act[0]))
        self.flippers_pos_fr_publisher.publish(Float64(data=act[0]))
        self.flippers_pos_rl_publisher.publish(Float64(data=act[1]))
        self.flippers_pos_rr_publisher.publish(Float64(data=act[1]))

        msg = Twist()
        msg.linear.x = np.clip(act[2], -1.5, 1.5)
        msg.angular.z = 0
        self.tracks_vel_publisher.publish(msg)

    def step(self, act):
        self.act_hist.extend(act)
        del self.act_hist[0:3]

        self.ros_publish_action(act)
        self.ros_rate.sleep()

        obs, pose_dict = self.ros_get_obs()

        rew = pose_dict["position"].z + pose_dict["verticality"] - np.square(pose_dict["position"].x) * 0.3

        far_x = abs(pose_dict["position"].x) > 1.5
        far_y = abs(pose_dict["position"].y) > 1.5
        toppled = abs(pose_dict["verticality"]) < 0.3 and self.step_ctr > 4
        max_steps_reached = self.step_ctr > self.config["max_steps"]
        done = far_x \
               or far_y \
               or toppled \
               or max_steps_reached

        self.step_ctr += 1
        self.rew_publisher.publish(Float64(rew))

        return obs, rew, done, {}

    def reset(self):
        # Kill the mapping
        # self.kill_aloam()
        self.step_ctr = 0
        self.episode_ctr += 1

        self.ros_publish_action([0,0,0])
        self._reset_pose()
        time.sleep(0.2)

        self.act_hist = [0] * self.act_history_horizon  * 3
        self.state_hist = [0] * self.obs_history_horizon * 4

        #print(self.episode_ctr)

        # Relaunch mapping
        #self.launch_aloam()
        #time.sleep(0.5)

        while True:
            obs, pose_dict = self.ros_get_obs()
            if obs is not None: break
            rospy.loginfo_throttle(1, "Getting None observations. ")

        return obs

    def _reset_pose(self):
        # Reset the robot pose
        #with open(os.devnull, 'w') as f:
        #    subprocess.call(os.path.join(MARVRocketBalEnv.project_root_path, "scripts/reset_robot_pose_upright.sh"),
        #                    stdout=f,
        #                    stderr=subprocess.STDOUT)

        proc = subprocess.Popen(os.path.join(MARVRocketBalEnv.project_root_path, "scripts/reset_robot_pose_upright_wheels.sh"), shell=True)


    def launch_aloam(self):
        self.aloam_launch = roslaunch.parent.ROSLaunchParent(self.aloam_uuid, [(self.aloam_path, ["info_output:=log"])], show_summary=False)
        self.aloam_launch.start()

    def kill_aloam(self):
        try:
            self.aloam_launch.shutdown()
        except:
            pass

    def demo(self):
        self.reset()
        while True:
            for i in range(self.config["max_steps"]):
                self.step(np.random.randn(2))
            self.reset()

    def kill(self):
        pass

if __name__=="__main__":
    import yaml
    with open("configs/marv_rocket_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env = MARVRocketBalEnv(config)
    #env.demo()
    rospy.loginfo("Marv rocket bal spinning..")
    rospy.spin()


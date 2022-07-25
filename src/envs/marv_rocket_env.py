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

class MARVRocketEnv:
    project_root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    def __init__(self, config):
        self.config = config

        # Environment variables
        self.act_dim = 2
        self.obs_dim = 18

        self.PoseTuple = namedtuple("PoseTuple", ["x", "y", "z", "qx", "qy", "qz", "qw"])
        self.default_action = [3.1415, 0]
        self.current_prog_ub = -0.7 # [-1,1]
        #self.previous_act = self.default_action

        # min height: 0.114, max_height level on flipper tips: 0.37, max height on ground looking up : 0.37
        self.init_ros()

    def init_ros(self):
        print("{} initializing ros".format(self.__class__.__name__))
        rospy.init_node("marv_rocket_env_node")

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
                                                       queue_size=1)

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

        self.rew_publisher = rospy.Publisher("/marv_rocket_reward",
                                     Float64,
                                     queue_size=1)

        self.prog_publisher = rospy.Publisher("/marv_rocket_prog",
                                             Float64,
                                             queue_size=1)

        self.step_service = rospy.Service("step_service", StepEnv, self.step_handler)
        self.reset_service = rospy.Service("reset_service", ResetEnv, self.reset_handler)

    def step_handler(self, request):
        obs, rew, done, _ = self.step(request.act.data)
        return StepEnvResponse(obs=Float64MultiArray(data=obs), rew=Float64(data=rew), done=Bool(data=done))

    def reset_handler(self, _):
        obs = self.reset()
        return ResetEnvResponse(obs=Float64MultiArray(data=obs))

    def calc_target_height_and_pitch(self):
        p_sc = (self.param_target + 1) * 0.5
        height = 0.11 + (0.26 * p_sc) + p_sc * (np.sin(p_sc) * 0.26)
        pitch = -p_sc * (np.pi * 0.4)
        return height, pitch

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
                rospy.logwarn("{}: Getting flipper pos transform of {} resulted in err: {}".format(MARVRocketEnv, fn, err))
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
        roll, pitch, yaw = euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))
        rot_mat = quaternion_matrix((quat.x, quat.y, quat.z, quat.w))

        # Directional vectors
        x1, y1 = [np.cos(yaw), np.sin(yaw)]

        pose_dict = {"position": pos,
                     "quat": quat,
                     "matrix": rot_mat,
                     "euler": (roll, pitch, yaw),
                     "dir_vec": (x1, y1)}

        current_pose = self.PoseTuple(x=pos.x, y=pos.y, z=pos.z, qx=quat.x, qy=quat.y, qz=quat.z, qw=quat.w)
        previous_pose = deepcopy(self.previous_pose)
        self.previous_pose = current_pose

        # Give results in quaterion, euler and vector form
        return current_pose, previous_pose, pose_dict

    def ros_get_obs(self):
        # Get flipper positions
        flipper_tuple = self.get_flipper_positions()

        # Get body 6dof pose
        current_pose, previous_pose, pose_dict = self.get_robot_pose()

        # Check if any are None, then return none
        if any(dat is None for dat in [flipper_tuple, current_pose, previous_pose, pose_dict]):
            return None, None

        flipper_obs = flipper_tuple[0], (flipper_tuple[2] + np.pi * 0.5) * (2 / np.pi)
        time_feature = (float(self.step_ctr) / self.config["max_steps"]) * 2 - 1

        # Assemble complete obs vec
        obs = np.concatenate([flipper_obs, current_pose, previous_pose, [self.param_target, time_feature]])
        return obs, pose_dict

    def ros_publish_action(self, act):
        self.flippers_pos_fl_publisher.publish(Float64(data=0))
        self.flippers_pos_fr_publisher.publish(Float64(data=0))
        self.flippers_pos_rl_publisher.publish(Float64(data=act[0]))
        self.flippers_pos_rr_publisher.publish(Float64(data=act[0]))

        msg = Twist()
        msg.linear.x = np.clip(act[1], -1.2, 1.2)
        msg.angular.z = 0
        self.tracks_vel_publisher.publish(msg)

    def step(self, act):
        t1 = time.time()
        target_flipper_pos = np.maximum(1.7 * np.pi + np.clip(act[0], -1, 1) * np.pi * 0.6, 3.15)
        target_track_vel = act[1]
        self.ros_publish_action([target_flipper_pos, target_track_vel])
        self.previous_act = deepcopy(act)

        obs, pose_dict = self.ros_get_obs()

        #rew = pose_dict["position"].z + 0.8 * pose_dict["matrix"][2, 0] # - 0.1 * abs(pose_dict["matrix"][2, 1])
        # TODO: Try to remove the pitch penalty, just use height penalty
        rew = 0.5 - 5 * np.square(pose_dict["position"].z - self.target_height) - np.square(pose_dict["euler"][1] - self.target_pitch) - 0.3 * np.square(pose_dict["position"].x)
        done = abs(pose_dict["position"].x) > 1.4\
               or abs(pose_dict["position"].y) > 1.4 \
               or abs(pose_dict["euler"][1]) > 2.5 \
               or abs(pose_dict["euler"][0]) > 2.5 \
               or self.step_ctr > self.config["max_steps"]

        if abs(pose_dict["euler"][1]) > 2.5:
            rew -= 10

        self.step_ctr += 1
        self.rew_publisher.publish(Float64(rew))
        self.ros_rate.sleep()
        t2 = time.time()
        print(t2-t1)

        return obs, rew, done, {}

    def reset(self):
        # Kill the mapping
        # self.kill_aloam()

        self.current_prog_ub = 1 #np.minimum(0.002 + self.current_prog_ub, 1.0)
        self.param_target = np.random.rand() * (self.current_prog_ub + 1) - 1
        self.target_height, self.target_pitch = self.calc_target_height_and_pitch()

        self.step_ctr = 0
        self.previous_pose = self.PoseTuple(*[0,0,0,0,0,0,1])
        self.previous_act = [0,0]

        # Publish default flipper positions and sleep
        self.ros_publish_action(self.default_action)
        time.sleep(0.5)

        # Reset the robot pose
        with open(os.devnull, 'w') as f:
            subprocess.call(os.path.join(MARVRocketEnv.project_root_path, "scripts/reset_robot_pose.sh"),
                            stdout=f,
                            stderr=subprocess.STDOUT)
        time.sleep(0.2)

        self.current_act = [0, 0]

        # Relaunch mapping
        #self.launch_aloam()
        #time.sleep(0.5)

        while True:
            obs, _ = self.ros_get_obs()
            if obs is not None: break
            rospy.loginfo_throttle(1, "Getting None observations. ")

        self.prog_publisher.publish(Float64(self.param_target))

        return obs

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

    env = MARVRocketEnv(config)
    #env.demo()
    rospy.loginfo("Marv rocket spinning..")
    rospy.spin()


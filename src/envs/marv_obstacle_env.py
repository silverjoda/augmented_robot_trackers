import os
import subprocess
import threading
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
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray, Float64, Bool
from tf.transformations import euler_from_quaternion, quaternion_matrix
from marv_msgs.msg import Float64MultiArray as MarvFloat64MultiArray
from std_msgs.msg import Float64MultiArray

import src.utilities as utilities


class MARVObstacleEnv:
    project_root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    def __init__(self, config):
        self.config = config

        # Environment variables
        self.act_dim = 2
        self.obs_dim = 14

        self.default_action = [0, 0]
        self.current_act = [0, 0]

        self.init_ros()

    def init_ros(self):
        print("{} initializing ros".format(self.__class__.__name__))
        rospy.init_node("marv_obstacle_env_node")

        # For aloam launching
        rospack = rospkg.RosPack()
        self.launch_path = os.path.join(rospack.get_path('augmented_robot_trackers'), "launch/sim_marv_obstacle.launch")
        self.launch_uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.launch_uuid)

        self.ros_rate = rospy.Rate(self.config["ros_rate"])
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.config["tf_buffer_size"]))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.current_path = []
        self.current_target = None

        self.path_data = None
        self.path_lock = threading.Lock()

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

        self.rew_publisher = rospy.Publisher("/marv_obstacle_reward",
                                     Float64,
                                     queue_size=1)

        self.flippers_max_torque_publisher = rospy.Publisher("marv/flippers_max_torque_controller/cmd_vel",
                                                             MarvFloat64MultiArray,
                                                             queue_size=1)

        # actual
        self.step_service = rospy.Service("step_service", StepEnv, self.step_handler)
        # Reset
        self.reset_service = rospy.Service("reset_service", ResetEnv, self.reset_handler)
        time.sleep(0.1)

    def _ros_path_callback(self, data):
        with self.path_lock:
            self.path_data = data
            self.new_path = True

    def step_handler(self, request):
        obs, rew, done, _ = self.step(request.act.data)
        msg = StepEnvResponse(obs=Float64MultiArray(data = obs), rew=Float64(data=rew), done=Bool(data=done))
        return msg

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
                rospy.logwarn("{}: Getting flipper pos transform of {} resulted in err: {}".format(MARVObstacleEnv, fn, err))
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
        return [flipper_tuple[0],flipper_tuple[2]]

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

        current_pose = [quat.x, quat.y, quat.z,quat.w]
        previous_pose = deepcopy(self.previous_pose)
        self.previous_pose = current_pose

        # Give results in quaterion, euler and vector form
        return current_pose, previous_pose, pose_dict

    def ros_get_obs(self, current_target):
        # Get flipper positions
        flipper_tuple = self.get_flipper_positions()

        # Get body 6dof pose
        current_pose, previous_pose, pose_dict = self.get_robot_pose()

        # Check if any are None, then return none
        if any(dat is None for dat in [flipper_tuple, current_pose, previous_pose, pose_dict]):
            return None, None

        delta_target = [pose_dict["position"].x - current_target.pose.position.x,
                        pose_dict["position"].y - current_target.pose.position.y]

        # Assemble complete obs vec
        obs = np.concatenate([flipper_tuple, current_pose, previous_pose, self.previous_act, delta_target])
        return obs, pose_dict

    def ros_publish_action(self, act):
        self.flippers_pos_fl_publisher.publish(Float64(data=act[0]))
        self.flippers_pos_fr_publisher.publish(Float64(data=act[0]))
        self.flippers_pos_rl_publisher.publish(Float64(data=act[1]))
        self.flippers_pos_rr_publisher.publish(Float64(data=act[1]))

    def ros_publish_reward(self, rew):
        self.rew_publisher.publish(Float64(data=rew))

    def ros_publish_flipper_torque_limits(self, flipper_dict):
        # Publish flippers vel
        flippers_torque_msg = MarvFloat64MultiArray()
        flippers_torque_msg.data = [flipper_dict["rear_right"], flipper_dict["rear_left"],
                                 flipper_dict["front_right"], flipper_dict["front_left"]]
        self.flippers_max_torque_publisher.publish(flippers_torque_msg)

    def get_dist_to_target(self, rob, tar):
        pos_delta = np.sqrt(np.square(rob["position"].x - tar.pose.position.x)
                            + np.square(rob["position"].y - tar.pose.position.y)
                            + np.square(rob["position"].z - tar.pose.position.z))
        return pos_delta

    def step(self, act):
        self.current_act[0] = np.clip(self.current_act[0] + act[0] * 0.4, -2 * np.pi, 2 * np.pi)
        self.current_act[1] = np.clip(self.current_act[1] + act[1] * 0.4, -2 * np.pi, 2 * np.pi)

        self.ros_publish_action(self.current_act)
        self.previous_act = deepcopy(act)

        current_target, current_path = self.get_current_target_and_path()
        obs, pose_dict = self.ros_get_obs(current_target)
        if self.current_target != current_target:
            self.current_target = current_target
            self.prev_target_dist = self.get_dist_to_target(pose_dict["position"], self.current_target)

        if current_target is None:
            pos_delta = 0
        else:
            # Distance between robot pose and target
            pos_delta = self.get_dist_to_target(pose_dict["position"], current_target)

        if self.prev_target_dist is None:
            target_reach_rew = 0
        else:
            target_reach_rew = (self.prev_target_dist - pos_delta) * 5
        self.prev_target_dist = pos_delta

        rew = target_reach_rew
        done = abs(pose_dict["position"].x) > 1\
               or abs(pose_dict["position"].y) > 1 \
               or abs(pose_dict["euler"][1]) > 2 \
               or abs(pose_dict["euler"][0]) > 2 \
               or self.step_ctr > self.config["max_steps"]

        self.rew_publisher.publish(Float64(rew))

        self.step_ctr += 1
        self.ros_rate.sleep()

        return obs, rew, done, {}

    def reset(self):
        # Kill the mapping
        self.kill_launch()

        self.ros_publish_flipper_torque_limits({"front_left" : 50, "front_right" : 50, "rear_left" : 50, "rear_right" : 50})

        self.step_ctr = 0
        self.previous_pose = [0,0,0,1]
        self.previous_act = [0,0]
        self.current_target, _ = self.get_current_target_and_path()

        # Publish default flipper positions and sleep
        self.ros_publish_action(self.default_action)
        time.sleep(0.3)

        # Reset the robot pose
        with open(os.devnull, 'w') as f:
            subprocess.call(os.path.join(MARVObstacleEnv.project_root_path, "scripts/reset_robot_pose.sh"),
                            stdout=f,
                            stderr=subprocess.STDOUT)
        time.sleep(0.2)

        self.current_act = [0, 0]


        # Relaunch mapping
        self.launch_launch()
        time.sleep(1.0)

        while True:
            obs, pose_dict = self.ros_get_obs()
            if obs is not None: break

        while True:
            obs, _ = self.ros_get_obs()
            if obs is not None: break
            rospy.loginfo_throttle(1, "Getting None observations. ")

        self.prev_target_dist = self.get_dist_to_target(pose_dict["position"], self.current_target)

        return obs

    def get_current_target_and_path(self):
        _, _, pose_dict = self.get_robot_pose()

        with self.path_lock:
            # If empty path or we didn't receive path messages yet
            if self.path_data is None or len(self.path_data.poses) == 0 or pose_dict is None:
                return self.current_target, self.current_path

            if not self.new_path:
                return self.current_target, self.current_path

            # Art
            if len(self.path_data.poses) == 1:
                self.current_path = self.path_data
            else:
                self.current_path = self.path_data

            while True:
                if self.current_target is not None and utilities.dist_between_pose_and_position(self.current_target, pose_dict["position"]) >= 0.3:
                    break
                if len(self.current_path.poses) > 0:
                    self.current_target = self.current_path.poses[0]
                    del self.current_path.poses[0]
                if len(self.current_path.poses) == 0 or self.current_target is not None:
                    break

        return self.current_target, self.current_path

    def launch_launch(self):
        self.launch = roslaunch.parent.ROSLaunchParent(self.launch_uuid, [(self.launch_path, ["info_output:=log"])], show_summary=True)
        self.launch.start()

    def kill_launch(self):
        try:
            self.launch.shutdown()
        except:
            print("Failed to shutdown launch, probably just the initial reset.")

    def demo(self):
        self.reset()

        while True:
            for i in range(self.config["max_steps"]):
                self.step(self.ActionTuple(*np.random.randn(4)))
            self.reset()

    def kill(self):
        pass

if __name__=="__main__":
    import yaml
    with open("configs/marv_obstacle_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env = MARVObstacleEnv(config)
    env.demo()
    #rospy.loginfo("Marv obstacle spinning..")
    #rospy.spin()


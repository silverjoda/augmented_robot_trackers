#!/usr/bin/env python
import os
import random
import time
import threading
import src.utilities as utilities
import rospy
from copy import deepcopy
import numpy as np
import ros_numpy
import rospy
import tf
from tf.transformations import quaternion_from_euler, quaternion_matrix, quaternion_from_matrix, euler_from_quaternion
import tf2_ros
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Vector3, Pose, PoseArray
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path, Odometry
#from rds_msgs.msg import K3
from scipy.spatial import KDTree
from std_srvs.srv import SetBool
from spot_msgs.srv import SetLocomotion
from spot_msgs.msg import MobilityParams
from spot_msgs.msg import BehaviorFaultState
from spot_msgs.msg import SystemFaultState
from spot_msgs.msg import BatteryStateArray
from spot_msgs.msg import PowerState
from spot_msgs.msg import LeaseArray
from spot_msgs.msg import EStopStateArray
from spot_msgs.srv import ClearBehaviorFault
from spot_msgs.msg import Feedback

import os
import threading
import time
from copy import deepcopy

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_srvs.srv import Trigger, TriggerResponse, SetBool, TriggerRequest

class SpotAssitant:
    def __init__(self, config):
        self.config = config
        self.init_ros(name=self.config["node_name"])

    def init_ros(self, name):
        rospy.init_node(name)
        rospy.loginfo("Starting spot assistant")

        self.ros_rate = rospy.Rate(self.config["ros_rate"])

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.config["tf_buffer_size"]))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.path_data = None
        self.trav_vis_pc_data = None
        self.trav_vis_pc_array = None
        self.trav_vis_kdtree = None

        self.path_lock = threading.Lock()
        self.trav_vis_pc_lock = threading.Lock()
        self.nav_cmd_vel_lock = threading.Lock()
        self.odometry_lock = threading.Lock()
        self.mobility_params_lock = threading.Lock()
        self.power_state_lock = threading.Lock()
        self.estop_lock = threading.Lock()
        self.leases_lock = threading.Lock()
        self.feedback_lock = threading.Lock()

        self.walking_dir = "FW"
        
        rospy.Subscriber(self.config["planner_path_topic"],
                         Path,
                         self._ros_path_callback, queue_size=1)

        rospy.Subscriber(self.config["traversability_visual_topic"],
                         PointCloud2,
                         self._ros_trav_vis_pc_callback, queue_size=1)

        # (Uncommented until tested)
        rospy.Subscriber("/nav/cmd_vel",
                         Twist,
                         self._ros_nav_cmd_vel_callback, queue_size=1)

        # Spot status subscribers
        rospy.Subscriber("/spot/status/behavior_faults",
                         BehaviorFaultState,
                         self._ros_behaviorfaultstate_callback, queue_size=1)

        rospy.Subscriber("/spot/status/system_faults",
                         SystemFaultState,
                         self._ros_systemfaultstate_callback, queue_size=1)

        rospy.Subscriber("/spot/status/power_state",
                         PowerState,
                         self._ros_powerstate_callback, queue_size=1)

        rospy.Subscriber("/spot/status/estop",
                         EStopStateArray,
                         self._ros_estop_callback, queue_size=1)

        rospy.Subscriber("/spot/status/leases",
                         LeaseArray,
                         self._ros_leases_callback, queue_size=1)

        rospy.Subscriber("/spot/status/feedback",
                         Feedback,
                         self._ros_feedback_callback, queue_size=1)

        rospy.Subscriber("/spot/status/mobility_params",
                         MobilityParams,
                         self._ros_mobility_params_callback, queue_size=1)

        self.path_semantic_pts_pub = rospy.Publisher(self.config["path_semantic_pts_topic"],
                                                 PointCloud2,
                                                 queue_size=1)

        self.forced_dir_pub = rospy.Publisher(self.config["force_dir_topic"],
                                               Vector3,
                                               queue_size=1)

        self.forced_dir_odom_pub = rospy.Publisher(self.config["force_dir_odom_topic"],
                                              Odometry,
                                              queue_size=1)

        self.slope_dirs_pub = rospy.Publisher(self.config["slopes_dirs_topic"],
                                            PoseArray,
                                            queue_size=1)

        self.spot_walking_mode_pub = rospy.Publisher(self.config["spot_walking_mode_topic"],
                                              String,
                                              queue_size=1)


        time.sleep(1)

    def _ros_behaviorfaultstate_callback(self, data):
        # Check for Behavior Faults, if any, just clear them, lol.
        if len(data.faults) > 0:
            for bf in data.faults:
                # Get ID of fault, and then clear it if possible
                if bf.cause == 1:
                    rospy.logwarn("Spot has fallen, attempting to self right in a few seconds")
                    time.sleep(2)
                    self.call_service("estop/release")
                    time.sleep(3)
                    self.call_service("power_on")
                    time.sleep(3)
                    self.call_service("sit")
                    time.sleep(3)
                    self.call_service("power_on")
                    time.sleep(5)
                    self.call_service("self_right")
                    time.sleep(15)
                    self.call_service("sit")
                    time.sleep(3)
                    self.call_service("self_right")
                    time.sleep(12)
                    self.call_service("stand")
                    time.sleep(3)

                with self.feedback_lock:
                    if hasattr(self, "feedback_data"):
                        t1 = time.time()
                        rospy.logwarn("Spot cannot get up, trying random shit...")
                        while not self.feedback_data.standing:
                            random_command = random.choice(["sit", "stand", "self_right", "estop/release", "power_on"])
                            time.sleep(1)
                            self.call_service(random_command)
                            if time.time() - t1 > 30:
                                break

                            self.call_service("power_off")
                            time.sleep(5)
                            self.call_service("power_on")
                            time.sleep(3)
                            self.call_service("estop/release")
                            time.sleep(1)
                            self.call_service("sit")
                            time.sleep(1)
                            self.call_service("stand")
                            time.sleep(1)

                if bf.status == 1:
                    self.clear_behavior_fault(bf.behavior_fault_id)

                # Caused by fall and not clearable
                if bf.cause == 1 and bf.status == 2:
                    # Decide what to do here if we can't self right. Probably restart
                    rospy.logerr_throttle(1, "Spot has fallen and cannot recover :(  ... might attempt to restart the system")

    def _ros_systemfaultstate_callback(self, data):
        if len(data.faults) > 0:
            for bf in data.faults:
                pass

    def _ros_powerstate_callback(self, data):
        with self.power_state_lock:
            self.power_state_data = data

    def _ros_estop_callback(self, data):
        with self.estop_lock:
            self.estop_data = data

    def _ros_feedback_callback(self, data):
        with self.feedback_lock:
            self.feedback_data = data

    def _ros_leases_callback(self, data):
        with self.leases_lock:
            self.leases_data = data
            for l in self.leases_data.resources:
                if l.lease_owner.client_name == '':
                    rospy.loginfo_throttle(1, "Spot assistant has noticed that the claim lease is not active, attempting to claim")
                    self.call_service("claim")

    def _ros_path_callback(self, data):
        with self.path_lock:
            self.path_data = data

    def _ros_nav_cmd_vel_callback(self, data):
        with self.nav_cmd_vel_lock:
            self.nav_cmd_vel_data = data

            # Check if soft estop. If active, release it
            with self.estop_lock:
                if hasattr(self, "estop_data"):
                    for esd in self.estop_data.estop_states:
                        if esd.name == "software_estop" and esd.state == 1:
                            self.call_service("estop/release")
                            rospy.loginfo("Calling estop")

            # Check if power state. If off, then power on
            with self.power_state_lock:
                if hasattr(self, "power_state_data"):
                    if self.power_state_data.motor_power_state == 1:
                        rospy.loginfo("Spot assistant has registered nav/cmd_vel, but motors are off. Attempting to power on")
                        self.call_service("power_on")
                        time.sleep(3)

            # If not standing and non zero cmd_vel, stand
            with self.feedback_lock:
                if hasattr(self, "feedback"):
                    if not self.feedback_data.standing and abs(self.nav_cmd_vel_data.linear.x) > 0.099 or abs(self.nav_cmd_vel_data.angular.z) > 0.099:
                        self.call_service("sit")
                        time.sleep(2)
                        self.call_service("stand")
                        time.sleep(1)

    def _ros_trav_vis_pc_callback(self, data):
        trav_vis_pc_array = self.make_array_from_trav_data(data)

        if len(trav_vis_pc_array) == 0:
            return

        trav_vis_kdtree = KDTree(trav_vis_pc_array[:, :3])

        with self.trav_vis_pc_lock:
            self.trav_vis_pc_data = data
            self.trav_vis_pc_array = trav_vis_pc_array
            self.trav_vis_kdtree = trav_vis_kdtree

    def _ros_mobility_params_callback(self, data):
        with self.mobility_params_lock:
            self.mobility_params_data = data

    def make_array_from_trav_data(self, data):
        pc = ros_numpy.numpify(data).ravel()
        pc = np.stack([pc[f] for f in ['x', 'y', 'z', 'trav', 'slopex', 'slopey', 'cost']]).T
        pc_filt_traversable = pc[pc[:, 3] < 3, :]
        n_pts = len(pc_filt_traversable)
        if n_pts > self.config["max_pc_points"]:
            decim_coeff = self.config["max_pc_points"] / float(n_pts)
            pc_filt_traversable = pc_filt_traversable[np.random.rand(n_pts) < decim_coeff, :]
        return pc_filt_traversable

    def calculate_path_trav_points(self, position, orientation, head_pos, tail_pos):
        # Copy path
        with self.path_lock:
            path = deepcopy(self.path_data)

        # Make KDtree from traversability map for better searching
        with self.trav_vis_pc_lock:
            # Return empty list if data not available yet
            if self.trav_vis_kdtree is None: return ([], [] ,[], [])

            # Find indeces of neighboring points of traversability map which lie on path waypoints
            locations = []
            locations.append([tail_pos[0], tail_pos[1], tail_pos[2]])
            locations.append([position.x, position.y, position.z])
            locations.append([head_pos[0], head_pos[1], head_pos[2]])

            if path is not None:
                for i in range(0, len(path.poses), self.config["path_lookahead_pose_skip"]):
                    p = path.poses[i]
                    x, y, z = p.pose.position.x, p.pose.position.y, p.pose.position.z

                    if self.dist_between_2_positions([position.x, position.y, position.z], [x,y,z]) > self.config["max_path_lookahead_dist"]:
                        break

                    locations.append([x,y,z])

            locations = np.array(locations)
            _, path_pt_indeces = self.trav_vis_kdtree.query(locations,
                                                       k=self.config["k_neighbors"],
                                                       #distance_upper_bound=self.config["kd_tree_max_rad"],
                                                       )

            # Use obtained waypoint neighborhood indeces to obtain statistics on those points
            mean_locs = []
            mean_slope_dirs = []
            mean_travs = []
            slopes = []
            for i in range(len(path_pt_indeces)):
                pt_indeces = path_pt_indeces[i]
                mean_loc = np.mean(self.trav_vis_pc_array[pt_indeces, :3], axis=0)
                mean_trav = np.mean(self.trav_vis_pc_array[pt_indeces, 3:4], axis=0)[0]
                slope_xy_array = self.trav_vis_pc_array[pt_indeces, 4:6]
                mean_slope_vec = np.mean(slope_xy_array, axis=0)

                slope = mean_slope_vec[np.argmax(np.abs(mean_slope_vec))]

                denom = np.sum(np.abs(mean_slope_vec))
                if denom > 0:
                    mean_slope_dir = mean_slope_vec / denom
                else:
                    mean_slope_dir = [0,0]

                if abs(slope) >= self.config["min_slope_for_dir_enforcement"]:
                    mean_locs.append(mean_loc)
                    mean_slope_dirs.append(mean_slope_dir)
                    slopes.append(slope)

                mean_travs.append(mean_trav)

            mean_locs_arr, mean_slope_dirs_arr, slopes_arr, mean_travs_arr = np.array(mean_locs), np.array(mean_slope_dirs), np.array(slopes), np.array(mean_travs)

        return mean_locs_arr, mean_slope_dirs_arr, slopes_arr, mean_travs_arr

    def get_pose(self):
        # Get pose using TF
        try:
            trans = self.tf_buffer.lookup_transform(self.config["root_frame"],
                                                    self.config["spot_root_frame"],
                                                    rospy.Time(0),
                                                    rospy.Duration(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn_throttle(1, "Spot assistant: Get_robot_pose TRANSFORMATION OLD, err: {}".format(err))
            return None, None, None, None

        trans_np = ros_numpy.numpify(trans.transform)
        head_pos = np.matmul(trans_np, np.array([0.4,0,0,1]))[:3]
        tail_pos = np.matmul(trans_np, np.array([-0.4,0,0,1]))[:3]

        return trans.transform.translation, trans.transform.rotation, head_pos, tail_pos

    def dist_between_2_positions(self, p1, p2):
        return np.sqrt(np.sum(np.square(np.array(p1) - np.array(p2))))

    def publish_dir(self, enforced_dir):
        msg = Vector3()
        if enforced_dir is None:
            msg.x, msg.y, msg.z = [-1, -1, -1]
        else:
            msg.x, msg.y = enforced_dir
            msg.z = 1
        self.forced_dir_pub.publish(msg)

    def publish_odom_dir(self, pos, enforced_dir):
        if enforced_dir is None:
            return

        orientation_quat = quaternion_from_euler(0, 0, np.arctan2(enforced_dir[1], enforced_dir[0]))
        msg = Odometry()
        msg.header.frame_id = self.config["root_frame"]
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = pos
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = orientation_quat
        self.forced_dir_odom_pub.publish(msg)

    def publish_stairs_pts(self, stairs_pts):
        if stairs_pts is None or self.trav_vis_pc_data is None or len(stairs_pts) == 0: return
        N = len(stairs_pts)
        stairs_pc = np.zeros(N, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
        ])

        pts = np.array(stairs_pts)
        stairs_pc['x'] = pts[:, 0]
        stairs_pc['y'] = pts[:, 1]
        stairs_pc['z'] = pts[:, 2]

        # Publish stairs points
        msg = ros_numpy.msgify(PointCloud2, stairs_pc)
        with self.trav_vis_pc_lock:
            msg.header.frame_id = self.trav_vis_pc_data.header.frame_id
        msg.header.stamp = rospy.Time.now()

        self.path_semantic_pts_pub.publish(msg)

    def publish_slope_dirs(self, mean_locs, xy_dirs):
        if mean_locs is None or len(mean_locs) == 0 or self.trav_vis_pc_data is None: return
        # Publish pose array
        pose_array_msg = PoseArray()
        with self.trav_vis_pc_lock:
            pose_array_msg.header.frame_id = self.trav_vis_pc_data.header.frame_id
        for i in range(len(xy_dirs)):
            pose_msg = Pose()
            # pose_msg.header.frame_id = self.trav_vis_pc_data.header.frame_id
            pose_msg.position.x, pose_msg.position.y, pose_msg.position.z = mean_locs[i, :]
            orientation_quat = quaternion_from_euler(0, 0, np.arctan2(xy_dirs[i][1], xy_dirs[i][0]))
            pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w = orientation_quat
            pose_array_msg.poses.append(pose_msg)
        self.slope_dirs_pub.publish(pose_array_msg)

    def enable_stairs_mode(self, cmd_bool):
        # Read current gait type first, if different, then set
        with self.mobility_params_lock:
            if hasattr(self, "mobility_params_data"):
                if self.mobility_params_data.stair_hint == cmd_bool:
                    return

        try:
            rospy.wait_for_service("/spot/stair_mode", 0.3)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logwarn("Spot assistant: Timeout while waiting for stairs_mode_srv: %s" % (e,))
            return

        try:
            stairs_proxy = rospy.ServiceProxy("/spot/stair_mode", SetBool)
            cmd_result = stairs_proxy(cmd_bool)
            rospy.loginfo("Spot assistant: Setting spot stairs enable to: {}".format(cmd_bool))
        except rospy.ServiceException as e:
            rospy.logwarn("Spot assistant: Setting stairs mode call failed: %s" % e)
            return False
        return cmd_result

    def set_spot_locomotion_type(self, gait_type):
        gait_num = self.config[gait_type]

        # Read current gait type first, if different, then set
        with self.mobility_params_lock:
            if hasattr(self, "mobility_params_data"):
                if self.mobility_params_data.locomotion_hint == gait_num:
                    return

        try:
            rospy.wait_for_service("/spot/locomotion_mode", timeout=0.3)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Spot assistant: Timeout while waiting for setting locomotion mode: %s" % (e,))
            return
        try:
            locom_type_proxy = rospy.ServiceProxy("/spot/locomotion_mode", SetLocomotion)
            cmd_result = locom_type_proxy(gait_num)
            rospy.loginfo("Setting spot locomotion type to: {}".format(gait_num))
        except rospy.ServiceException as e:
            rospy.logwarn("Setting spot locomotion type call failed: %s" % e)
            return False
        return cmd_result

    def clear_behavior_fault(self, behavior_fault_id):
        try:
            rospy.wait_for_service("/spot/clear_behavior_fault", 0.3)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logwarn("Spot assistant: Timeout while waiting for clear_behavior_fault service: %s" % (e,))
            return

        try:
            service_proxy = rospy.ServiceProxy("/spot/clear_behavior_fault", ClearBehaviorFault)
            service_result = service_proxy(behavior_fault_id)
            rospy.loginfo("Spot assistant: attempt to clear fault with id {} returned {}".format(behavior_fault_id, service_result))
        except rospy.ServiceException as e:
            rospy.logwarn("Spot assistant: Clearning behavior fault failed: %s" % e)
            return False
        return service_result

    def call_service(self, service_name):
        full_service_name = "/spot/{}".format(service_name)
        # Wait for last cmd_vel msg to have gone
        time.sleep(0.2)
        rospy.loginfo("Spot assistant: Registered {} attempt".format(service_name))

        try:
            rospy.wait_for_service(full_service_name, timeout=1)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Spot assistant: Timeout while waiting for {} service: {}".format(service_name, e))
            return False

        try:
            service_proxy = rospy.ServiceProxy(full_service_name, Trigger)
            res = service_proxy(TriggerRequest())
            rospy.loginfo("Spot assistant: Attempt to call: {} resulted in: {}".format(service_name, res))
        except rospy.ServiceException as e:
            rospy.logwarn("Spot assistant: Attempt to {} failed, err: {}".format(service_name, e))
            return False
        return res

    def loop_spot_assistant(self):
        while not rospy.is_shutdown():
            position, orientation, head_pos, tail_pos = self.get_pose()

            if position is None or orientation is None or head_pos is None or tail_pos is None:
                self.ros_rate.sleep()
                continue

            # Calculate waypoints which are on stairs and their slop dirs
            mean_locs, mean_slope_dirs, slopes, mean_travs = self.calculate_path_trav_points(position, orientation, head_pos, tail_pos)

            # === Decide on enforced dir
            enforced_dir = None
            enforced_dir_loc = None
            # Decide on what command to publish
            _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
            if len(mean_locs) > 0:
                for mean_loc, xy_dir, slope in zip(mean_locs, mean_slope_dirs, slopes):
                    if self.dist_between_2_positions([position.x, position.y, position.z], mean_loc) < self.config["dir_enforcement_dist_threshold"]:
                        if slope < 0:
                            self.walking_dir = "FW"
                        else:
                            self.walking_dir = "BW"
                        enforced_dir = xy_dir
                        enforced_dir_loc = mean_loc
                        break

            # Decide to set stairs mode if there is an enforced dir
            set_stairs = enforced_dir is not None

            # Decide on locomotion mode
            gait_type = "HINT_AUTO"
            if len(mean_travs) > 0:
                for mean_trav in mean_travs:
                    print(mean_trav)
                    difficult_obst = mean_trav > self.config["mean_trav_crawl_thresh"]
                    is_close = self.dist_between_2_positions([position.x, position.y, position.z], mean_trav) < self.config["crawl_enforcement_dist_threshold"]
                    if difficult_obst and is_close:
                        gait_type = "HINT_CRAWL"

            # Publish command
            if self.config["enable_stairs_mode_setting"]:
                self.enable_stairs_mode(set_stairs)

            if self.config["enable_locom_setting"]:
                self.set_spot_locomotion_type(gait_type)

            if self.config["enable_force_dir_setting"]:
                self.publish_dir(enforced_dir)

            self.publish_odom_dir(enforced_dir_loc, enforced_dir)
            self.publish_stairs_pts(mean_locs)
            self.publish_slope_dirs(mean_locs, mean_slope_dirs)

            self.ros_rate.sleep()

def main():
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/spot_assistant_config.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spot_assistant = SpotAssitant(config)
    spot_assistant.loop_spot_assistant()

if __name__=="__main__":
    main()

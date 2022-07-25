#!/usr/bin/env python

import time

import rospy
import tf
import tf2_ros
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Empty
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

class IcpPub:
    def __init__(self):
        node_name = "fake_icp_odom_publisher"
        rospy.init_node(node_name)
        self.ros_rate = rospy.Rate(30)

        rospy.loginfo("{} starting to publish fake_icp_odom.".format(node_name))

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        tf2_ros.TransformListener(self.tf_buffer)

        self.icp_odom_publisher = rospy.Publisher("icp_odom",
                                                  Odometry,
                                                  queue_size=1)

        rospy.Subscriber("X1/points_filtered",
                         PointCloud2,
                         self._ros_pc_cb, queue_size=1)

        time.sleep(0.3)

    def _ros_pc_cb(self, data):
        # Get transform
        try:
            trans = self.tf_buffer.lookup_transform("world",
                                                    "X1/base_link",
                                                    rospy.Time(0),
                                                    rospy.Duration(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn("Fake icp odom transform: TRANSFORMATION ERROR, err: {}".format(err))
            return None

        odom_msg = Odometry()
        odom_msg.pose.pose.position = trans.transform.translation
        odom_msg.pose.pose.orientation = trans.transform.rotation
        odom_msg.header.stamp = data.header.stamp
        odom_msg.header.frame_id = "world"
        self.icp_odom_publisher.publish(odom_msg)

def main():
    IcpPub()
    rospy.spin()

if __name__=="__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

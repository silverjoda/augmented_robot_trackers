#!/usr/bin/env python

import time

import rospy
import tf
import tf2_ros
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Empty

class ZRPpub:
    def __init__(self):
        node_name = "base_link_zrp_publisher"
        rospy.init_node(node_name)
        self.ros_rate = rospy.Rate(30)

        rospy.loginfo("{} starting to publish bl_zrp.".format(node_name))

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        tf2_ros.TransformListener(self.tf_buffer)

        self.bl_zpr_br = tf.TransformBroadcaster()

        time.sleep(0.1)

    def loop(self):
        while not rospy.is_shutdown():
            # Get pose
            (r, p, _), stamp = self.get_rpy(self.tf_buffer)
            self.bl_zpr_br.sendTransform((0, 0, -0.11),
                                    tf.transformations.quaternion_from_euler(0, -p, 0), #  (-r, -p, 0) for roll and pitch correction
                                    stamp,
                                    "X1/base_link_zrp",
                                    "X1/base_link")
            try:
                self.ros_rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def get_rpy(self, tf_buffer):
        # Get pose using TF
        try:
            trans = tf_buffer.lookup_transform("map",
                                               "X1/base_link",
                                               rospy.Time(0),
                                               rospy.Duration(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn_throttle(1, "BL_ZPR_PUBLISHER: TRANSFORMATION OLD err: {}".format(err))
            return (0,0,0), rospy.Time.now()

        quat = trans.transform.rotation
        return euler_from_quaternion((quat.x, quat.y, quat.z, quat.w)), trans.header.stamp

def main():
    zpr = ZRPpub()
    zpr.loop()

if __name__=="__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

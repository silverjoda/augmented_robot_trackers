import numpy as np
#from nav_msgs.msg import Path
#from sensor_msgs.msg import PointCloud2
import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_matrix
import threading

def get_robot_pose_dict(root_frame, base_link_frame, tf_buffer, time):
    # Get pose using TF
    try:
        trans = tf_buffer.lookup_transform(root_frame,
                                           base_link_frame,
                                           time)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
        rospy.logwarn("Get_robot_pose_dict: TRANSFORMATION ERROR, err: {}".format(err))
        return None

    # Translation
    pos = trans.transform.translation

    # Orientation
    quat = trans.transform.rotation
    roll, pitch, yaw = euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))
    matrix = quaternion_matrix((quat.x, quat.y, quat.z, quat.w))

    # Directional vectors
    x1, y1 = [np.cos(yaw), np.sin(yaw)]

    pose_dict = {"position": pos,
                 "quat": quat,
                 "matrix": matrix,
                 "euler": (roll, pitch, yaw),
                 "dir_vec": (x1, y1)}

    # Give results in quaterion, euler and vector form
    return pose_dict

def subscriber_factory(topic_name, topic_type):
    class RosSubscriber:
        def __init__(self):
            self.lock = threading.Lock()
            self.cb = None
            self.msg = None

    def msg_cb_wrapper(subscriber):
        def msg_cb(msg):
            with subscriber.lock:
                subscriber.msg = msg
        return msg_cb
    subscriber = RosSubscriber()
    rospy.Subscriber(topic_name,
                     topic_type,
                     msg_cb_wrapper(subscriber), queue_size=1)
    return subscriber
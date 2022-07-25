#!/bin/bash
rosbag record /camera_front/image_raw/compressed /camera_left/image_raw/compressed /camera_rear/image_raw/compressed /camera_right/image_raw/compressed /camera_up/image_raw/compressed /diagnostics /os_cloud_node/points_throttle /spot/depth/back/camera_info /spot/depth/back/image /spot/depth/frontleft/camera_info /spot/depth/frontleft/image /spot/depth/frontright/camera_info /spot/depth/frontright/image /spot/depth/left/camera_info /spot/depth/left/image /spot/depth/right/camera_info /spot/depth/right/image /spot/odometry /tf /tf_static /cmd_vel /joy


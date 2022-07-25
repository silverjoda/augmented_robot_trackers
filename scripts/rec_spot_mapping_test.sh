#!/bin/bash
rosbag record /spot/odometry /spot/cmd_vel /tf /tf_static /points /points_deskewed
# --lz4 # Use for compressed

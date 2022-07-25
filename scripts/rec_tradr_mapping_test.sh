#!/bin/bash
rosbag record /imu/odom /tf /tf_static /points /points_deskewed
# --lz4 # Use for compressed

#!/bin/bash
rosbag record /lidar_nuc /icp_odom /tf /tf_static /joy /imu/data /husky_velocity_controller/odom /odometry/filtered --lz4



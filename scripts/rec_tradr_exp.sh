#!/bin/bash
rosbag record -O ${1}.bag /imu/odom /tf /tf_static /points_filtered /art/debug/text_info /static_path_out /teleop/text_info /art/tradr_flipper_controller_state /rds/traversability_visual /art/debug/pc_bnds /art/feat_bbx_out /cmd_vel /imu/data

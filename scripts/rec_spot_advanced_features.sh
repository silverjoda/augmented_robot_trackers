#!/bin/bash
rosbag record -O spot_data_${1}.bag -a -x "(.*)rgbd(.*)|(.*)compressed(.*)|(.*)image(.*)|(.*)camera(.*)|(.*)raw(.*)|(.*)scan_point_cloud(.*)|(.*)os_cloud_node(.*)|/points_deskewed|/points|/points_slow"

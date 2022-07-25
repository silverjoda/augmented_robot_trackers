#!/bin/bash
rosbag record -O tradr_data_${1}.bag -a -x "(.*)rgbd(.*)|(.*)compressed(.*)|(.*)image(.*)|(.*)camera(.*)|(.*)raw(.*)|(.*)scan_point_cloud(.*)|(.*)os_cloud_node(.*)|/points_deskewed|/points|/dynamic_point_cloud|/points_slow"

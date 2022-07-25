#!/bin/bash
rosbag record -O marv_data_${1}.bag -a -x "(.*)rgbd(.*)|(.*)rds(.*)|(.*)compressed(.*)|(.*)image(.*)|(.*)camera(.*)|(.*)raw(.*)"

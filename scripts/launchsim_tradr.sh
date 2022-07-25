#!/bin/bash
SUBT_ENDLESS_MISSION=1 SUBT_ROBOT_TEAM=tradr_ta ~/cras_subt/src/subt_virtual/scripts/run_sim worldName:=cave_circuit_practice_01 enableGroundTruth:=false &
SUBT_ROBOT_TEAM=tradr_ta SUBT_USE_SINGULARITY=0 ~/cras_subt/src/subt_virtual/scripts/run_bridge_all &

sleep 15

rosservice call /X1/front_rgbd/set_rate 0.01
rosservice call /X1/rear_rgbd/set_rate 0.01
rosservice call /X1/left_rgbd/set_rate 0.01
rosservice call /X1/right_rgbd/set_rate 0.01

roslaunch aloam_velodyne sys_ta.launch

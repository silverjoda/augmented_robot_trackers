#!/bin/bash

if [ -z $1 ]
 then 
    worldName=cave_circuit_practice_01
else
    worldName=$1
fi

SUBT_ENDLESS_MISSION=1 SUBT_ROBOT_TEAM=1marv-rds ~/cras_subt/src/subt_virtual/scripts/run_sim worldName:=${worldName} enableGroundTruth:=true headless:=false &
SUBT_ROBOT_TEAM=1marv-rds SUBT_USE_SINGULARITY=0 ~/cras_subt/src/subt_virtual/scripts/run_bridge_all &



#IGN_TRANSPORT_TOPIC_STATISTICS=1 ign service -s /world/${worldName}/set_pose --reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --timeout 2000 --req 'name: "X1", position: {x: -0, y: -0, z: 1}'

#roslaunch augmented_robot_trackers sim_marv_filters.launch

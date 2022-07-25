#!/bin/bash

rm ${HOME}/subt_ws/install/share/ctu_cras_norlab_marv_sensor_config_1/worlds/example.sdf
ln -s ${HOME}/cras_subt/src/augmented_robot_trackers/src/envs/sdf/$1 ${HOME}/subt_ws/install/share/ctu_cras_norlab_marv_sensor_config_1/worlds/example.sdf

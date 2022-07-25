#!/bin/bash

rm ${HOME}/singularity_imgs/marv/ctu_cras_norlab_marv_sensor_config_1/worlds/example.sdf
ln -s ${HOME}/cras_subt/src/augmented_robot_trackers/src/envs/sdf/$1 ${HOME}/singularity_imgs/marv/ctu_cras_norlab_marv_sensor_config_1/worlds/example.sdf

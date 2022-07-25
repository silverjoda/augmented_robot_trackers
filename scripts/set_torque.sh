#!/bin/bash
rostopic pub /X1/flippers_cmd_max_torque/front_left std_msgs/Float64 "data: $@" --once &
rostopic pub /X1/flippers_cmd_max_torque/front_right std_msgs/Float64 "data: $@" --once &
rostopic pub /X1/flippers_cmd_max_torque/rear_left std_msgs/Float64 "data: $@" --once &
rostopic pub /X1/flippers_cmd_max_torque/rear_right std_msgs/Float64 "data: $@" --once &


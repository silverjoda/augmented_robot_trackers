#!/bin/bash
rosbag record /spot/cmd_vel /spot/odometry /spot/odometry/twist /spot/status/battery_states /spot/status/behavior_faults /spot/status/estop /spot/status/feedback /spot/status/feet /spot/status/leases /spot/status/metrics /spot/status/mobility_params /spot/status/power_state /spot/status/system_faults /spot/status/wifi /spot/trajectory/cancel /spot/trajectory/feedback /spot/trajectory/goal /spot/trajectory/result /spot/trajectory/status /tf /tf_static /rds/traversability_visual /rds/path /rds/debug_track /icp_odom /nav/cmd_vel


# --lz4 # Use for compressed





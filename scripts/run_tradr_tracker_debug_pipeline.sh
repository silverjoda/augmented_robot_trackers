rosbag play tradr_data_.bag --clock &
roslaunch augmented_robot_trackers tradr_tracker_debug_utilities.launch & 
rosrun augmented_robot_trackers marv_tracker.py


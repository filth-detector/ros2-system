#!/bin/bash
export ROS_DOMAIN_ID=1
source /opt/ros/jazzy/setup.bash
source install/setup.bash

(sleep 6; gz model --list) &
ros2 launch robot_bringup gazebo_launch.py "$@"

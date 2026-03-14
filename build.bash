#!/bin/bash
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
rm -rf build/ install/ log/
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -Wno-dev --parallel-workers $(nproc)
source install/setup.bash   

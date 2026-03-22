export ROS_DOMAIN_ID=1
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run filth_detector filth_detector --ros-args -p model_path:="../model_tests/segformer.onnx"

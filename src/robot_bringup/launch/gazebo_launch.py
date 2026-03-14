"""
Usage::

    ros2 launch robot_bringup gazebo_launch.py
    ros2 launch robot_bringup gazebo_launch.py world:=/path/to/custom.sdf
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PythonExpression,
    Command,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Build the Gazebo simulation launch with Nav2 and SLAM."""

    # ── Package paths ───────────────────────────────────────────────
    desc_share = get_package_share_directory('robot_description')
    bringup_share = get_package_share_directory('robot_bringup')
    pointcloud_to_laserscan_config = os.path.join(bringup_share, 'config', 'pointcloud_to_laserscan_config.yaml')

    # ── Launch arguments ────────────────────────────────────────────
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(desc_share, 'worlds', 'factory.sdf'),
        description='Path to the Gazebo SDF world file.',
    )
    world_path = LaunchConfiguration('world')

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode (server only, no GUI).',
    )
    headless = LaunchConfiguration('headless')

    # ── Environment ─────────────────────────────────────────────────
    # Let Gazebo find models/worlds installed in ROS share directories
    gz_resource_env = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(desc_share, 'worlds') + ':' +
        os.path.dirname(desc_share),
    )

    # ── Gazebo simulator ────────────────────────────────────────────
    # Build gz_args: always -r (run); add -s --headless-rendering
    # when headless=true so Gazebo runs without a GUI window.
    gz_render_args = PythonExpression([
        "'-r -s --headless-rendering ' if '",
        headless,
        "' == 'true' else '-r '",
    ])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py',
            )
        ),
        launch_arguments={
            'gz_args': [gz_render_args, world_path],
        }.items(),
    )

    # ── Robot description (xacro with use_sim:=true) ────────────────
    xacro_file = os.path.join(desc_share, 'urdf', 'robot.xacro')
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file, ' use_sim:=true']),
        value_type=str,
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
        output='screen',
    )

    # ── Spawn robot into Gazebo ─────────────────────────────────────
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'my_robot',
            '-topic', '/robot_description',
            '-x', '0.0',
            '-y', '-2.0',
            '-z', '0.2',
        ],
        output='screen',
    )

    # ── Gazebo ↔ ROS bridge ─────────────────────────────────────────
    bridge_config = os.path.join(bringup_share, 'config', 'gz_bridge_config.yaml')

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': bridge_config,
            'use_sim_time': True,
        }],
        output='screen',
    )

    # ── 3D point cloud → 2D laser scan ──────────────────────────────
    # plio_to_laserScan = Node(
    #     package='pointcloud_to_laserscan',
    #     executable='pointcloud_to_laserscan_node',
    #     name='pointcloud_to_laserscan',
    #     output='screen',
    #     parameters=[
    #         pointcloud_to_laserscan_config,
    #         {
    #             'use_sim_time': True,
    #         }
    #     ],
    #     remappings=[
    #         ('cloud_in', '/scan/points'),
    #         ('scan', '/scan'),
    #     ],
    # )

    # ── SLAM Toolbox ────────────────────────────────────────────────
    # slam_config = os.path.join(bringup_share, 'config', 'slam_toolbox_config.yaml')

    # slam = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(
    #             get_package_share_directory('slam_toolbox'),
    #             'launch', 'online_async_launch.py',
    #         )
    #     ),
    #     launch_arguments={
    #         'use_sim_time': 'true',
    #         'slam_params_file': slam_config,
    #     }.items(),
    # )

    # ── Nav2 ────────────────────────────────────────────────────────
    # nav2 = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(bringup_share, 'launch', 'bringup_launch.py')
    #     ),
    #     launch_arguments={
    #         'use_sim_time': 'true',
    #         'use_localization': 'False',  # SLAM Toolbox handles map→odom
    #     }.items(),
    # )

    # ── RViz ────────────────────────────────────────────────────────
    rviz_config = os.path.join(bringup_share, 'config', 'rviz_config.rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # ── Static TF: Gazebo scoped frame → URDF frame ────────────────
    # Gazebo Harmonic scopes sensor frames as model/link/sensor, e.g.
    # "my_robot/base_link/lidar".  Publish an identity transform so
    # pointcloud_to_laserscan (and any other node) can look up the
    # transform chain: my_robot/base_link/lidar → lidar_link → base_link.
    # gz_lidar_frame_fix = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     arguments=[
    #         '--frame-id', 'lidar_link',
    #         '--child-frame-id', 'my_robot/base_link/lidar',
    #         '--x', '0', '--y', '0', '--z', '0',
    #         '--roll', '0', '--pitch', '0', '--yaw', '0',
    #     ],
    #     parameters=[{'use_sim_time': True}],
    #     output='screen',
    # )

    gz_camera_frame_fix = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--frame-id', 'camera_link',
            '--child-frame-id', 'my_robot/camera_link/camera',
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # ── Joystick control ───────────────────────────────────────────
    teleop_config = os.path.join(bringup_share, 'config', 'teleop_config.yaml')

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{'deadzone': 0.1, 'autorepeat_rate': 20.0, 'use_sim_time': True}],
        output='screen',
    )

    teleop_joy_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy',
        parameters=[teleop_config, {'use_sim_time': True}],
        output='screen',
    )

    # ── Assemble ────────────────────────────────────────────────────
    # return LaunchDescription([
    #     world_arg,
    #     headless_arg,
    #     gz_resource_env,
    #     gazebo,
    #     robot_state_publisher,
    #     spawn_robot,
    #     gz_bridge,
    #     gz_lidar_frame_fix,
    #     gz_camera_frame_fix,
    #     plio_to_laserScan,
    #     slam,
    #     nav2,
    #     rviz,
    #     joy_node,
    #     teleop_joy_node,
    # ])
    
    return LaunchDescription([
        world_arg,
        headless_arg,
        gz_resource_env,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        gz_bridge,
        gz_camera_frame_fix,
        rviz,
        joy_node,
        teleop_joy_node,
    ])

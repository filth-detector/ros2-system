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
    # ── Package paths ───────────────────────────────────────────────
    desc_share = get_package_share_directory('robot_description')
    bringup_share = get_package_share_directory('robot_bringup')

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

    # ── Assemble ────────────────────────────────────────────────────
    return LaunchDescription([
        world_arg,
        headless_arg,
        gz_resource_env,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        gz_bridge,
        gz_camera_frame_fix,
    ])

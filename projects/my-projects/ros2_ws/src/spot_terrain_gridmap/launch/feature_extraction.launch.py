"""
Launch file for the **feature_extraction** node.

Usage:
    ros2 launch spot_terrain_gridmap feature_extraction.launch.py

Override the inference device from CLI:
    ros2 launch spot_terrain_gridmap feature_extraction.launch.py device:=cpu
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("spot_terrain_gridmap")
    default_params = os.path.join(
        pkg_share, "config", "feature_extraction_params.yaml"
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to the YAML parameter file.",
    )

    device_arg = DeclareLaunchArgument(
        "device",
        default_value="cuda",
        description="Inference device for DINOv2: 'cuda' or 'cpu'.",
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Isaac Sim) clock if true",
    )

    node = Node(
        package="spot_terrain_gridmap",
        executable="feature_extraction",
        name="feature_extraction",
        output="screen",
        emulate_tty=True,
        parameters=[
            LaunchConfiguration("params_file"),
            {"device": LaunchConfiguration("device")},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    return LaunchDescription([
        params_file_arg,
        device_arg,
        use_sim_time_arg,
        node,
    ])

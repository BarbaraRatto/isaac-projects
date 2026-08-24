import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('spot_energy_estimation')
    params_file = os.path.join(pkg_share, 'config', 'energy_params.yaml')

    energy_node = Node(
        package='spot_energy_estimation',
        executable='energy_estimation_node',
        name='energy_estimation_node',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([
        energy_node,
    ])

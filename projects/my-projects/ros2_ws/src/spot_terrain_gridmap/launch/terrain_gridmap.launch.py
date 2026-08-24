"""
Launch file per il nodo terrain_feature_node.

Uso:
    ros2 launch spot_terrain_gridmap terrain_gridmap.launch.py

Uso con override di un parametro da riga di comando:
    ros2 launch spot_terrain_gridmap terrain_gridmap.launch.py device:=cpu

IMPORTANTE: questo nodo richiede PyTorch/transformers installati nel venv
dedicato (vedi README.md). Va lanciato con l'interprete Python del venv
attivo (source ~/venvs/dino_env/bin/activate prima di lanciare, oppure
tramite lo script di attivazione automatica se configurato - vedi README).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('spot_terrain_gridmap')
    default_params_file = os.path.join(pkg_share, 'config', 'terrain_gridmap_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Percorso al file YAML dei parametri del nodo.',
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description="Device di inferenza per DINOv2: 'cuda' o 'cpu'.",
    )

    terrain_feature_node = Node(
        package='spot_terrain_gridmap',
        executable='terrain_feature_node',
        name='terrain_feature_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {'device': LaunchConfiguration('device')},
        ],
    )

    return LaunchDescription([
        params_file_arg,
        device_arg,
        terrain_feature_node,
    ])

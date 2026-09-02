#!/usr/bin/env python3
"""
rl_training.launch.py

Launch file per la fase di TRAINING.

Avvia:
  1. energy_costmap_node  — comprime la gridmap DINOv2 (384→16+1 layer)
  2. energy_estimation_node — già in esecuzione di solito, non duplicato
  
NON avvia:
  - NAV2 (non necessario durante il training — il cmd_vel è generato dall'env)
  - Isaac Sim (deve essere già in esecuzione separatamente)

Uso:
  ros2 launch spot_rl_controller rl_training.launch.py

Poi, in un altro terminale:
  python3 -m spot_rl_controller.train_rl --config config/rl_params.yaml
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("spot_rl_controller")
    config_file = os.path.join(pkg_share, "config", "rl_params.yaml")

    # ── Argomenti configurabili ────────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Usa il tempo di simulazione di Isaac Sim",
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    # ── Nodo: energy_costmap ───────────────────────────────────────────────
    energy_costmap_node = Node(
        package="spot_rl_controller",
        executable="energy_costmap_node",
        name="energy_costmap_node",
        output="screen",
        parameters=[
            config_file,
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            # Rimappature topic (da config/rl_params.yaml):
            # Input
            ("/terrain_gridmap",            "/terrain_gridmap"),
            ("/energy/current_consumption", "/energy/current_consumption"),
            # Output
            ("/energy_costmap_tensor",      "/energy_costmap_tensor"),
            ("/energy_costmap_debug",       "/energy_costmap_debug"),
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        energy_costmap_node,
    ])

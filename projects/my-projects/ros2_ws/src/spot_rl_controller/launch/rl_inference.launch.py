#!/usr/bin/env python3
"""
rl_inference.launch.py

Launch file per la fase di INFERENCE (robot in moto con modello addestrato).

Avvia:
  1. energy_costmap_node   — comprime la gridmap DINOv2
  2. rl_controller_node    — carica il modello PPO e pubblica /cmd_vel

NON avvia:
  - NAV2 controller server (sostituito dall'rl_controller_node)
  - NAV2 planner + bt_navigator: rimangono attivi per il path planning globale
    (avviali separatamente con nav2_bringup.launch.py se necessario)

Uso:
  ros2 launch spot_rl_controller rl_inference.launch.py \
      model_path:=/path/to/models/best_model

Poi invia un goal da RViz (2D Nav Goal) o:
  ros2 topic pub /goal_pose geometry_msgs/PoseStamped "{...}"
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
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="models/best_model",
        description="Percorso al modello SB3 addestrato (senza .zip)",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    model_path   = LaunchConfiguration("model_path")

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
    )

    # ── Nodo: rl_controller (inference) ───────────────────────────────────
    rl_controller_node = Node(
        package="spot_rl_controller",
        executable="rl_controller_node",
        name="rl_controller_node",
        output="screen",
        parameters=[
            config_file,
            {
                "use_sim_time": use_sim_time,
                "model_path":   model_path,
                "config_path":  config_file,
            },
        ],
    )

    # ── Relay /cmd_vel → /cmd_vel/smooth (verso CHAMP nel container) ──────
    relay_node = Node(
        package="topic_tools",
        executable="relay",
        name="cmd_vel_rl_relay",
        output="screen",
        arguments=["/cmd_vel", "/cmd_vel/smooth"],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription([
        use_sim_time_arg,
        model_path_arg,
        energy_costmap_node,
        rl_controller_node,
        relay_node,
    ])

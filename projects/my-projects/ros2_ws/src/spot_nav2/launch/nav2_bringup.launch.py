import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('spot_nav2')
    params_file = os.path.join(pkg_share, 'config', 'nav2_params.yaml')

    lifecycle_nodes = ['controller_server', 'planner_server', 'bt_navigator']
    
    # Definiamo il parametro sim_time da passare a tutti i nodi
    sim_time_param = {'use_sim_time': True}

    # Definizione del nodo relay per inoltrare i comandi di velocità
    relay_node = Node(
        package='topic_tools',
        executable='relay',
        name='cmd_vel_relay',
        output='screen',
        arguments=['/cmd_vel', '/cmd_vel/smooth'],
        parameters=[sim_time_param]
    )

    # Nodo per il TF statico (ATTENZIONE: Usalo solo se non hai corretto Isaac Sim!)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_base_link_to_base',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base'],
        parameters=[sim_time_param]
    )

    return LaunchDescription([
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            # Aggiunto sim_time_param in coda ai parametri
            parameters=[params_file, sim_time_param],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params_file, sim_time_param],
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params_file, sim_time_param],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'autostart': True,
                'node_names': lifecycle_nodes,
            }],
        ),
        # Aggiunta del nodo relay alla lista di esecuzione
        relay_node,
        static_tf_node, # Aggiunto qui alla fine
    ])

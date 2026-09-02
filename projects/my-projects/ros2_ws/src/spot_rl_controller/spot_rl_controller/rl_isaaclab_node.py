#!/usr/bin/env python3
"""
rl_isaaclab_node.py

Nodo ROS2 per eseguire inferenza del modello PPO addestrato in Isaac Lab.
Prende Odometria e Goal (PoseStamped), calcola le finte feature DINOv2
basate sulle coordinate (come nell'addestramento), normalizza l'osservazione
e produce /cmd_vel per CHAMP.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
import math
import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Parametri copiati dall'ambiente Isaac Lab (SpotEnergyEnv)
TILE_SIZE_X = 2.0
TILE_SIZE_Y = 2.0
N_TERRAIN_ROWS = 8
TERRAIN_ENERGY_COST = np.array([
    0.10, 0.40, 0.45, 0.70, 0.45, 0.60, 0.80, 0.55
], dtype=np.float32)

VX_MAX = 1.0
VY_MAX = 0.6
OMEGA_MAX = 1.0


class RLIsaacLabNode(Node):
    def __init__(self):
        super().__init__('rl_isaaclab_node')
        
        self.declare_parameter('model_path', 'models/final_model_isaaclab')
        self.declare_parameter('freq_hz', 20.0)
        self.declare_parameter('goal_tol', 0.5)
        
        # Se il path è relativo, usa la cartella corrente (che è ros2_ws se hai lanciato da lì)
        model_p = self.get_parameter('model_path').value
        if not model_p.startswith('/'):
            model_p = os.path.abspath(model_p)

        self.model_path = model_p
        self.freq_hz = self.get_parameter('freq_hz').value
        self.goal_tol = self.get_parameter('goal_tol').value
        
        # Path di zip e pkl
        zip_path = self.model_path + '.zip'
        pkl_path = self.model_path + '_vecnorm.pkl'
        
        if not os.path.exists(zip_path):
            self.get_logger().error(f"Modello non trovato: {zip_path}")
            return
            
        self.get_logger().info(f"Caricamento modello PPO da {zip_path}...")
        self.model = PPO.load(zip_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        self.vec_norm = None
        if os.path.exists(pkl_path):
            self.get_logger().info(f"Caricamento VecNormalize da {pkl_path}...")
            import pickle
            with open(pkl_path, 'rb') as f:
                self.vec_norm = pickle.load(f)
            self.vec_norm.training = False
        else:
            self.get_logger().warn("Nessun file _vecnorm.pkl trovato. Le osservazioni non verranno normalizzate!")
            
        # Stato Robot
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_vx = 0.0
        self.robot_vy = 0.0
        self.robot_omega = 0.0
        
        # Stato Goal
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0
        self.goal_received = False
        
        # Subscriber & Publisher
        self.sub_odom = self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self._on_goal, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel/smooth', 1)
        
        # Timer ciclo di controllo
        self.timer = self.create_timer(1.0 / self.freq_hz, self._control_loop)
        self.get_logger().info("RL IsaacLab Node (DINOv2 Fake Features) inizializzato e pronto!")
        
    def _on_odom(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.robot_vx = msg.twist.twist.linear.x
        self.robot_vy = msg.twist.twist.linear.y
        self.robot_omega = msg.twist.twist.angular.z
        
    def _on_goal(self, msg: PoseStamped):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.goal_received = True
        self.get_logger().info(f"Nuovo goal: ({self.goal_x:.2f}, {self.goal_y:.2f})")
        
    def _control_loop(self):
        if not self.goal_received:
            return
            
        dist = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        if dist < self.goal_tol:
            self.get_logger().info(f"Goal raggiunto (dist={dist:.2f}m)!", throttle_duration_sec=2.0)
            cmd = Twist()
            self.pub_cmd.publish(cmd)  # Ferma il robot
            return
            
        # 1. Costruisci Gridmap (2500) con vera feature 2D finta (come in training a SCACCHIERA)
        # Calcoliamo l'esatta coordinata X e Y nel mondo per le 50x50 celle della griglia locale (da -2.5m a +2.5m)
        grid_x = np.linspace(-2.45, 2.45, 50)
        grid_y = np.linspace(-2.45, 2.45, 50)
        # np.meshgrid("ij") crea le coordinate dove la prima dim (righe) è X e la seconda (colonne) è Y
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y, indexing='ij')
        
        world_x = self.robot_x + mesh_x
        world_y = self.robot_y + mesh_y
        
        tile_x = np.floor(world_x / TILE_SIZE_X).astype(int)
        tile_y = np.floor(world_y / TILE_SIZE_Y).astype(int)
        
        # Campo Minato
        is_obstacle = ((tile_x % 2) == 1) & ((tile_y % 2) == 0)
        tile_idx = np.where(is_obstacle, 6, 0)
        
        costs_2d = TERRAIN_ENERGY_COST[tile_idx]
        gridmap = costs_2d.flatten().astype(np.float32)
        
        # 2. Vettore goal (4)
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        
        # In Isaac Lab calcoliamo dtheta = 0 perche' il goal e' un punto
        goal_vec = np.array([dx, dy, math.cos(0.0), math.sin(0.0)], dtype=np.float32)
        
        # 3. Energy vec (3) - in inferenza assumiamo valori normali / bassi
        # O potremmo mappare i consumi veri, ma per ora il robot è focalizzato sulla feature visiva
        energy_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # 4. Robot vel (3) normalizzata sui max dell'ambiente
        vx_norm = np.clip(self.robot_vx / VX_MAX, -1.0, 1.0)
        vy_norm = np.clip(self.robot_vy / VY_MAX, -1.0, 1.0)
        omega_norm = np.clip(self.robot_omega / OMEGA_MAX, -1.0, 1.0)
        robot_vel = np.array([vx_norm, vy_norm, omega_norm], dtype=np.float32)
        
        # 5. Concatenazione finale (1D, 2510 elementi)
        obs_array = np.concatenate([gridmap, goal_vec, energy_vec, robot_vel])
        obs_array = obs_array.reshape(1, -1)  # SB3 richiede (batch_size, obs_dim)
        
        # 6. Normalizzazione (Fondamentale!)
        if self.vec_norm is not None:
            obs_array = self.vec_norm.normalize_obs(obs_array)
            
        # 7. Predizione
        action, _ = self.model.predict(obs_array, deterministic=True)
        
        # 8. Decodifica azione e pubblicazione
        # action e' (1, 3) e in range [-1, 1]
        cmd = Twist()
        cmd.linear.x = float(action[0][0]) * VX_MAX
        cmd.linear.y = float(action[0][1]) * VY_MAX
        cmd.angular.z = float(action[0][2]) * OMEGA_MAX
        
        self.pub_cmd.publish(cmd)
        
def main(args=None):
    rclpy.init(args=args)
    node = RLIsaacLabNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

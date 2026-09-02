#!/usr/bin/env python3
"""
spot_env.py

Gymnasium Environment per il robot Spot — FALLBACK via ROS2.

Questo file è un'alternativa semplificata a spot_isaaclab_env.py:
  - Usalo per DEBUG o test rapidi senza Isaac Lab
  - Per il TRAINING vero, usa spot_isaaclab_env.py (Isaac Lab, molto più veloce)
  - Per INFERENCE sul robot reale, usa rl_controller_node.py

Come funziona:
  Si connette al sistema ROS2 attivo e:
  - Osserva: gridmap compressa (energy_costmap_node) + energia + odometria + goal
  - Agisce:  pubblica /cmd_vel (vx, vy, omega)
  - Reward:  progresso verso goal - consumo energetico - penalità temporale

Observation space (flat vector 2510-d, identico a spot_isaaclab_env):
  [0:2500]   gridmap energetica 50×50 (flatten)  — in [0, 1]
  [2500:2504] goal_vec (dx, dy, cos_dtheta, sin_dtheta)
  [2504:2507] energy_vec (P_inst_norm, P_avg_norm, E_cum_norm)
  [2507:2510] robot_vel (vx_norm, vy_norm, omega_norm)

Action space:
  (vx_norm, vy_norm, omega_norm) ∈ [-1, 1]³  — denormalizzato in step()
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

from energy_msgs.msg import EnergyEstimate


# ===========================================================================
# Costanti di default
# ===========================================================================
DEFAULT_VX_MAX    = 0.5    # [m/s]
DEFAULT_VY_MAX    = 0.3    # [m/s]
DEFAULT_OMEGA_MAX = 1.0    # [rad/s]
DEFAULT_N_ROWS    = 50
DEFAULT_N_COLS    = 50
DEFAULT_MAX_STEPS = 500
DEFAULT_GOAL_TOL  = 0.15   # [m]
DEFAULT_P_MAX     = 200.0  # [W]
OBS_DIM           = 2510   # 50×50 + 4 + 3 + 3


class _SpotROSNode(Node):
    """
    Nodo ROS2 interno: gestisce subscriber/publisher per l'env Gymnasium.
    """

    def __init__(self, config: dict) -> None:
        super().__init__("spot_rl_env_node")

        topics = config.get("topics", {})
        self._costmap_topic = topics.get("costmap_out",  "/energy_costmap_tensor")
        self._energy_topic  = topics.get("energy_in",    "/energy/current_consumption")
        self._odom_topic    = topics.get("odom_in",      "/odom")
        self._goal_topic    = topics.get("goal_in",      "/goal_pose")
        self._cmd_vel_topic = topics.get("cmd_vel_out",  "/cmd_vel")

        self._obs_lock = threading.Lock()

        # Gridmap compressa (1, 50, 50) — solo il layer di costo energetico
        self._cost_grid: np.ndarray | None = None
        self._n_rows = int(config.get("gridmap", {}).get("size_x", 5.0) /
                           config.get("gridmap", {}).get("resolution", 0.1))
        self._n_cols = int(config.get("gridmap", {}).get("size_y", 5.0) /
                           config.get("gridmap", {}).get("resolution", 0.1))

        # Segnali energetici
        self._inst_power: float = 0.0
        self._avg_power:  float = 0.0
        self._cum_energy: float = 0.0
        self._p_max = float(config.get("reward", {})
                            .get("p_max_normalization", DEFAULT_P_MAX))

        # Odometria
        self._robot_x:     float = 0.0
        self._robot_y:     float = 0.0
        self._robot_yaw:   float = 0.0
        self._robot_vx:    float = 0.0
        self._robot_vy:    float = 0.0
        self._robot_omega: float = 0.0

        # Goal
        self._goal_x:      float = 5.0
        self._goal_y:      float = 0.0
        self._goal_yaw:    float = 0.0
        self._goal_received = False

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        self.create_subscription(Float32MultiArray, self._costmap_topic, self._on_costmap, 1)
        self.create_subscription(EnergyEstimate, self._energy_topic, self._on_energy, qos_be)
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, qos_be)
        self.create_subscription(PoseStamped, self._goal_topic, self._on_goal, 10)

        self._pub_cmd  = self.create_publisher(Twist,       self._cmd_vel_topic, 1)
        self._pub_goal = self.create_publisher(PoseStamped, self._goal_topic,    10)

    # ── Callback gridmap ──────────────────────────────────────────────────
    def _on_costmap(self, msg: Float32MultiArray) -> None:
        """
        Legge il tensor pubblicato da energy_costmap_node.
        Usa solo l'ultimo layer (indice -1): il layer di costo energetico.
        """
        dims = msg.layout.dim
        if len(dims) < 3:
            return
        n_layers = dims[0].size
        n_rows   = dims[1].size
        n_cols   = dims[2].size
        data = np.array(msg.data, dtype=np.float32).reshape(n_layers, n_rows, n_cols)
        with self._obs_lock:
            # Layer -1: costo energetico per cella (l'unico che usiamo)
            self._cost_grid = data[-1]  # (n_rows, n_cols)
            self._n_rows = n_rows
            self._n_cols = n_cols

    # ── Callback energia ──────────────────────────────────────────────────
    def _on_energy(self, msg: EnergyEstimate) -> None:
        with self._obs_lock:
            self._inst_power = msg.instantaneous_power
            self._avg_power  = msg.average_power
            self._cum_energy = msg.cumulative_energy

    # ── Callback odometria ────────────────────────────────────────────────
    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        with self._obs_lock:
            self._robot_x     = msg.pose.pose.position.x
            self._robot_y     = msg.pose.pose.position.y
            self._robot_yaw   = math.atan2(siny, cosy)
            self._robot_vx    = msg.twist.twist.linear.x
            self._robot_vy    = msg.twist.twist.linear.y
            self._robot_omega = msg.twist.twist.angular.z

    # ── Callback goal ─────────────────────────────────────────────────────
    def _on_goal(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        with self._obs_lock:
            self._goal_x       = msg.pose.position.x
            self._goal_y       = msg.pose.position.y
            self._goal_yaw     = math.atan2(siny, cosy)
            self._goal_received = True

    # ── Azioni ────────────────────────────────────────────────────────────
    def publish_cmd_vel(self, vx: float, vy: float, omega: float) -> None:
        twist = Twist()
        twist.linear.x  = float(vx)
        twist.linear.y  = float(vy)
        twist.angular.z = float(omega)
        self._pub_cmd.publish(twist)

    def stop_robot(self) -> None:
        self.publish_cmd_vel(0.0, 0.0, 0.0)

    def set_goal(self, gx: float, gy: float) -> None:
        msg = PoseStamped()
        msg.header.frame_id = "odom"
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.pose.position.x = float(gx)
        msg.pose.position.y = float(gy)
        msg.pose.orientation.w = 1.0
        self._pub_goal.publish(msg)
        with self._obs_lock:
            self._goal_x       = float(gx)
            self._goal_y       = float(gy)
            self._goal_yaw     = 0.0
            self._goal_received = True

    # ── Snapshot ──────────────────────────────────────────────────────────
    def get_snapshot(self) -> dict:
        with self._obs_lock:
            return {
                "cost_grid":   self._cost_grid.copy() if self._cost_grid is not None else None,
                "inst_power":  self._inst_power,
                "avg_power":   self._avg_power,
                "cum_energy":  self._cum_energy,
                "robot_x":     self._robot_x,
                "robot_y":     self._robot_y,
                "robot_yaw":   self._robot_yaw,
                "robot_vx":    self._robot_vx,
                "robot_vy":    self._robot_vy,
                "robot_omega": self._robot_omega,
                "goal_x":      self._goal_x,
                "goal_y":      self._goal_y,
                "goal_yaw":    self._goal_yaw,
                "n_rows":      self._n_rows,
                "n_cols":      self._n_cols,
                "p_max":       self._p_max,
            }


# ===========================================================================
# Gymnasium Environment (fallback ROS2)
# ===========================================================================
class SpotEnv(gym.Env):
    """
    Env Gymnasium che legge dati reali da ROS2 (Isaac Sim in esecuzione).

    NOTA: usa spot_isaaclab_env.py per il training vero (Isaac Lab).
    Questo env è utile per debug, test veloci e come riferimento.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        if config is None:
            config = {}

        action_cfg  = config.get("action_space", {})
        reward_cfg  = config.get("reward", {})
        episode_cfg = config.get("episode", {})
        gridmap_cfg = config.get("gridmap", {})

        self._vx_max    = float(action_cfg.get("vx_max",    DEFAULT_VX_MAX))
        self._vy_max    = float(action_cfg.get("vy_max",    DEFAULT_VY_MAX))
        self._omega_max = float(action_cfg.get("omega_max", DEFAULT_OMEGA_MAX))

        self._lambda_avg  = float(reward_cfg.get("lambda_avg_power",  0.01))
        self._lambda_inst = float(reward_cfg.get("lambda_inst_power", 0.003))
        self._lambda_time = float(reward_cfg.get("lambda_time",       0.005))
        self._goal_bonus  = float(reward_cfg.get("goal_bonus",        10.0))
        self._coll_pen    = float(reward_cfg.get("collision_penalty",  5.0))
        self._p_max       = float(reward_cfg.get("p_max_normalization", DEFAULT_P_MAX))

        self._max_steps      = int(episode_cfg.get("max_steps",           DEFAULT_MAX_STEPS))
        self._goal_tol       = float(episode_cfg.get("goal_tolerance_m",  DEFAULT_GOAL_TOL))
        self._stuck_timeout  = int(episode_cfg.get("stuck_timeout_steps", 50))
        self._stuck_min_prog = float(episode_cfg.get("stuck_min_progress_m", 0.05))

        self._n_rows = int(gridmap_cfg.get("size_x", 5.0) / gridmap_cfg.get("resolution", 0.1))
        self._n_cols = int(gridmap_cfg.get("size_y", 5.0) / gridmap_cfg.get("resolution", 0.1))

        # ── Spazio azioni: (vx, vy, omega) normalizzato in [-1, 1] ─────────
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # ── Spazio osservazioni: flat vector 2510-d ───────────────────────
        # Identico a spot_isaaclab_env per compatibilità del modello
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # Stato episodio
        self._step_count   = 0
        self._prev_dist    = 0.0
        self._best_dist    = 0.0
        self._stuck_steps  = 0
        self._ep_e_start   = 0.0

        self._goal_candidates = [
            (3.0, 0.0), (3.0, 2.0), (3.0, -2.0),
            (5.0, 0.0), (5.0, 2.0), (5.0, -2.0),
            (2.0, 3.0), (4.0, 3.0), (4.0, -3.0),
        ]
        self._rng = np.random.default_rng()

        # ── Nodo ROS2 interno ─────────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init()
        self._ros_node = _SpotROSNode(config)
        self._spin_thread = threading.Thread(
            target=self._spin_loop, daemon=True
        )
        self._spin_thread.start()
        self.get_logger = self._ros_node.get_logger

    def _spin_loop(self) -> None:
        while rclpy.ok():
            try:
                rclpy.spin_once(self._ros_node, timeout_sec=0.01)
            except Exception:
                break

    # ======================================================================
    # reset()
    # ======================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._ros_node.stop_robot()
        time.sleep(0.2)

        gx, gy = (options["goal"] if options and "goal" in options
                  else self._rng.choice(self._goal_candidates))
        self._ros_node.set_goal(float(gx), float(gy))

        obs = self._wait_valid_obs(timeout_s=5.0)
        snap = self._ros_node.get_snapshot()

        self._step_count  = 0
        self._prev_dist   = self._dist_to_goal(snap)
        self._best_dist   = self._prev_dist
        self._stuck_steps = 0
        self._ep_e_start  = snap["cum_energy"]

        return obs, {"goal": (gx, gy), "initial_dist": self._prev_dist}

    # ======================================================================
    # step()
    # ======================================================================
    def step(self, action):
        vx    = float(np.clip(action[0], -1.0, 1.0)) * self._vx_max
        vy    = float(np.clip(action[1], -1.0, 1.0)) * self._vy_max
        omega = float(np.clip(action[2], -1.0, 1.0)) * self._omega_max
        self._ros_node.publish_cmd_vel(vx, vy, omega)

        time.sleep(1.0 / 20.0)   # attesa a 20 Hz

        snap    = self._ros_node.get_snapshot()
        obs     = self._build_obs(snap)
        reward, info = self._compute_reward(snap)

        dist       = self._dist_to_goal(snap)
        terminated = dist < self._goal_tol
        if terminated:
            reward += self._goal_bonus
            self._ros_node.stop_robot()
            info["success"] = True

        # Stuck detection
        if dist < self._best_dist - self._stuck_min_prog:
            self._best_dist   = dist
            self._stuck_steps = 0
        else:
            self._stuck_steps += 1
        if self._stuck_steps >= self._stuck_timeout and not terminated:
            reward    -= self._coll_pen
            terminated = True
            info["stuck"] = True

        self._prev_dist  = dist
        self._step_count += 1
        truncated = (self._step_count >= self._max_steps) and not terminated

        info.update({
            "dist_to_goal":  dist,
            "avg_power":     snap["avg_power"],
            "cum_energy_ep": snap["cum_energy"] - self._ep_e_start,
            "step":          self._step_count,
        })
        return obs, reward, terminated, truncated, info

    # ======================================================================
    # Helpers
    # ======================================================================
    def _compute_reward(self, snap):
        dist     = self._dist_to_goal(snap)
        progress = self._prev_dist - dist
        r_prog   = progress / max(self._goal_tol, 0.01)
        r_energy = (self._lambda_avg  * snap["avg_power"]  / self._p_max
                    + self._lambda_inst * snap["inst_power"] / self._p_max)
        r_time   = self._lambda_time
        return float(r_prog - r_energy - r_time), {
            "r_progress": r_prog, "r_energy": r_energy, "r_time": r_time,
        }

    def _build_obs(self, snap) -> np.ndarray:
        """
        Costruisce il vettore flat 2510-d:
          [gridmap 50×50 flatten | goal 4 | energy 3 | vel 3]
        """
        n_rows = self._n_rows
        n_cols = self._n_cols

        # Gridmap: costo energetico per cella, in [0,1]
        if snap["cost_grid"] is not None:
            cg = snap["cost_grid"]
            if cg.shape != (n_rows, n_cols):
                cg = np.zeros((n_rows, n_cols), dtype=np.float32)
            gridmap_flat = np.nan_to_num(cg, nan=0.0).flatten()
        else:
            gridmap_flat = np.zeros(n_rows * n_cols, dtype=np.float32)

        dx     = snap["goal_x"] - snap["robot_x"]
        dy     = snap["goal_y"] - snap["robot_y"]
        dtheta = snap["goal_yaw"] - snap["robot_yaw"]
        goal_vec = np.array([dx, dy, math.cos(dtheta), math.sin(dtheta)], dtype=np.float32)

        energy_vec = np.array([
            np.clip(snap["inst_power"] / self._p_max, 0.0, 1.0),
            np.clip(snap["avg_power"]  / self._p_max, 0.0, 1.0),
            np.clip(snap["cum_energy"] / (self._p_max * self._max_steps * 0.05), 0.0, 1.0),
        ], dtype=np.float32)

        robot_vel = np.array([
            np.clip(snap["robot_vx"]    / self._vx_max,    -1.0, 1.0),
            np.clip(snap["robot_vy"]    / self._vy_max,    -1.0, 1.0),
            np.clip(snap["robot_omega"] / self._omega_max, -1.0, 1.0),
        ], dtype=np.float32)

        return np.concatenate([gridmap_flat, goal_vec, energy_vec, robot_vel]).astype(np.float32)

    def _dist_to_goal(self, snap):
        dx = snap["goal_x"] - snap["robot_x"]
        dy = snap["goal_y"] - snap["robot_y"]
        return math.sqrt(dx*dx + dy*dy)

    def _wait_valid_obs(self, timeout_s=5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snap = self._ros_node.get_snapshot()
            if snap["cost_grid"] is not None:
                return self._build_obs(snap)
            time.sleep(0.1)
        return self._build_obs(self._ros_node.get_snapshot())

    def close(self):
        self._ros_node.stop_robot()
        self._ros_node.destroy_node()
        super().close()

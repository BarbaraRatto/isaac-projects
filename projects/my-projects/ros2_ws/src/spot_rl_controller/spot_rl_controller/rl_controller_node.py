#!/usr/bin/env python3
"""
rl_controller_node.py

Nodo ROS2 di inference: carica il modello PPO addestrato e produce /cmd_vel
in tempo reale, agendo come sostituto del NAV2 controller server.

Flusso:
  /energy_costmap_tensor + /energy/current_consumption + /odom + /goal_pose
        |
        v
  SpotFeaturesExtractor + PPO policy
        |
        v
  /cmd_vel  →  CHAMP  →  /joint_trajectory  →  Isaac Sim / robot reale

Modalità reattiva (opzionale):
  Se P_avg supera la soglia configurata, la policy viene rieseguita anche
  senza attendere il tick di controllo standard, permettendo di reagire
  immediatamente a un cambio di terreno non previsto.

Uso:
  ros2 run spot_rl_controller rl_controller_node \
      --ros-args -p model_path:=models/best_model \
                 -p config_path:=config/rl_params.yaml
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

from energy_msgs.msg import EnergyEstimate

# Aggiungi il pacchetto al path se necessario
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


class RLControllerNode(Node):
    """
    Nodo di inference che carica il modello SB3 e pubblica /cmd_vel
    alla frequenza configurata, con modalità reattiva opzionale.
    """

    def __init__(self) -> None:
        super().__init__("rl_controller_node")

        # ── Dichiarazione parametri ROS2 ───────────────────────────────────
        self.declare_parameter("model_path", "models/best_model")
        self.declare_parameter("vecnorm_path", "")   # opzionale
        self.declare_parameter("config_path", "config/rl_params.yaml")

        model_path  = self.get_parameter("model_path").value
        vecnorm_path = self.get_parameter("vecnorm_path").value
        config_path = self.get_parameter("config_path").value

        # ── Caricamento configurazione ─────────────────────────────────────
        self._config = self._load_config(config_path)
        topics   = self._config.get("topics",   {})
        reactive = self._config.get("reactive",  {})
        ctrl     = self._config.get("control",   {})
        action_c = self._config.get("action_space", {})
        gridmap_c = self._config.get("gridmap", {})

        self._vx_max    = float(action_c.get("vx_max",    0.5))
        self._vy_max    = float(action_c.get("vy_max",    0.3))
        self._omega_max = float(action_c.get("omega_max", 1.0))
        self._p_max     = float(self._config.get("reward", {})
                                .get("p_max_normalization", 200.0))
        self._n_pca     = int(gridmap_c.get("n_pca_components", 16))
        self._n_layers  = self._n_pca + 1
        self._n_rows    = int(gridmap_c.get("size_x", 5.0) / gridmap_c.get("resolution", 0.1))
        self._n_cols    = int(gridmap_c.get("size_y", 5.0) / gridmap_c.get("resolution", 0.1))
        self._max_steps = int(self._config.get("episode", {}).get("max_steps", 500))
        self._goal_tol  = float(self._config.get("episode", {}).get("goal_tolerance_m", 0.15))

        self._freq_hz         = float(ctrl.get("frequency_hz", 20.0))
        self._reactive_enabled = bool(reactive.get("enabled", True))
        self._power_threshold  = float(reactive.get("power_spike_threshold_w", 50.0))
        self._reactive_cooldown = int(reactive.get("cooldown_steps", 10))

        # ── Caricamento modello SB3 ────────────────────────────────────────
        from stable_baselines3 import PPO

        if not os.path.exists(model_path + ".zip"):
            self.get_logger().error(f"Modello non trovato: {model_path}.zip")
            raise FileNotFoundError(f"SB3 model not found: {model_path}.zip")

        self.get_logger().info(f"Caricamento modello PPO da: {model_path}")
        self._model = PPO.load(model_path, device="cuda")
        self.get_logger().info("Modello caricato ✓")

        # ── Stato osservativo ──────────────────────────────────────────────
        self._lock = threading.Lock()
        self._gridmap:     np.ndarray | None = None
        self._inst_power:  float = 0.0
        self._avg_power:   float = 0.0
        self._cum_energy:  float = 0.0
        self._robot_x:     float = 0.0
        self._robot_y:     float = 0.0
        self._robot_yaw:   float = 0.0
        self._robot_vx:    float = 0.0
        self._robot_vy:    float = 0.0
        self._robot_omega: float = 0.0
        self._goal_x:      float = 5.0
        self._goal_y:      float = 0.0
        self._goal_yaw:    float = 0.0
        self._goal_received = False

        # Storico potenza media per rilevamento spike
        self._prev_avg_power: float = 0.0
        self._reactive_cooldown_ctr: int = 0

        # ── QoS ───────────────────────────────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscriber ────────────────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray,
            topics.get("costmap_out", "/energy_costmap_tensor"),
            self._on_costmap, 1,
        )
        self.create_subscription(
            EnergyEstimate,
            topics.get("energy_in", "/energy/current_consumption"),
            self._on_energy, qos_be,
        )
        self.create_subscription(
            Odometry,
            topics.get("odom_in", "/odom"),
            self._on_odom, qos_be,
        )
        self.create_subscription(
            PoseStamped,
            topics.get("goal_in", "/goal_pose"),
            self._on_goal, 10,
        )

        # ── Publisher ─────────────────────────────────────────────────────
        self._pub_cmd = self.create_publisher(
            Twist, topics.get("cmd_vel_out", "/cmd_vel"), 1,
        )

        # ── Timer di controllo (ciclo principale) ─────────────────────────
        period = 1.0 / self._freq_hz
        self._ctrl_timer = self.create_timer(period, self._control_loop)

        self.get_logger().info(
            f"rl_controller_node pronto — "
            f"frequenza: {self._freq_hz} Hz, "
            f"modalità reattiva: {'ON' if self._reactive_enabled else 'OFF'}"
        )

    # ======================================================================
    # Callback subscriber
    # ======================================================================
    def _on_costmap(self, msg: Float32MultiArray) -> None:
        dims = msg.layout.dim
        if len(dims) < 3:
            return
        n_layers = dims[0].size
        n_rows   = dims[1].size
        n_cols   = dims[2].size
        data = np.array(msg.data, dtype=np.float32).reshape(
            n_layers, n_rows, n_cols
        )
        with self._lock:
            self._gridmap = data

    def _on_energy(self, msg: EnergyEstimate) -> None:
        with self._lock:
            self._inst_power = msg.instantaneous_power
            self._avg_power  = msg.average_power
            self._cum_energy = msg.cumulative_energy

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        with self._lock:
            self._robot_x     = msg.pose.pose.position.x
            self._robot_y     = msg.pose.pose.position.y
            self._robot_yaw   = yaw
            self._robot_vx    = msg.twist.twist.linear.x
            self._robot_vy    = msg.twist.twist.linear.y
            self._robot_omega = msg.twist.twist.angular.z

    def _on_goal(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        with self._lock:
            self._goal_x       = msg.pose.position.x
            self._goal_y       = msg.pose.position.y
            self._goal_yaw     = yaw
            self._goal_received = True
        self.get_logger().info(
            f"Nuovo goal ricevuto: ({msg.pose.position.x:.2f}, "
            f"{msg.pose.position.y:.2f})"
        )

    # ======================================================================
    # Ciclo di controllo principale
    # ======================================================================
    def _control_loop(self) -> None:
        """
        Eseguito ad ogni tick del timer (20 Hz).
        Costruisce l'obs, predice l'azione, pubblica /cmd_vel.
        """
        with self._lock:
            if not self._goal_received:
                return   # nessun goal ancora: aspetta

            # Snapshot dello stato corrente
            snap = {
                "gridmap":     self._gridmap,
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
            }
            avg_power = self._avg_power

        # ── Controllo raggiungimento goal ──────────────────────────────────
        dist = self._dist_to_goal(snap)
        if dist < self._goal_tol:
            self._publish_cmd(0.0, 0.0, 0.0)
            self.get_logger().info(
                f"Goal raggiunto (dist={dist:.3f}m). "
                "In attesa del prossimo goal..."
            )
            with self._lock:
                self._goal_received = False
            return

        # ── Rilevamento spike di potenza (modalità reattiva) ───────────────
        if self._reactive_enabled and self._reactive_cooldown_ctr == 0:
            power_increase = avg_power - self._prev_avg_power
            if power_increase > self._power_threshold:
                self.get_logger().warn(
                    f"Spike energetico rilevato: +{power_increase:.1f} W "
                    f"(P_avg: {self._prev_avg_power:.1f}→{avg_power:.1f} W). "
                    "Ricalcolo percorso..."
                )
                self._reactive_cooldown_ctr = self._reactive_cooldown
        else:
            self._reactive_cooldown_ctr = max(0, self._reactive_cooldown_ctr - 1)
        self._prev_avg_power = avg_power

        # ── Costruzione observation ────────────────────────────────────────
        obs = self._build_obs(snap)

        # ── Predizione azione PPO ─────────────────────────────────────────
        action, _ = self._model.predict(obs, deterministic=True)

        # ── Denormalizzazione e pubblicazione /cmd_vel ─────────────────────
        vx    = float(np.clip(action[0], -1.0, 1.0)) * self._vx_max
        vy    = float(np.clip(action[1], -1.0, 1.0)) * self._vy_max
        omega = float(np.clip(action[2], -1.0, 1.0)) * self._omega_max

        self._publish_cmd(vx, vy, omega)

        self.get_logger().debug(
            f"cmd_vel: vx={vx:.2f} vy={vy:.2f} ω={omega:.2f} | "
            f"dist={dist:.2f}m P_avg={avg_power:.1f}W",
            throttle_duration_sec=0.5,
        )

    # ======================================================================
    # Helper: observation builder (speculare a SpotEnv._build_obs)
    # ======================================================================
    def _build_obs(self, snap: dict) -> dict:
        """
        Costruisce il dizionario di osservazione per la policy.
        Deve essere identico a SpotEnv._build_obs per coerenza.
        """
        n_layers = self._n_layers
        n_rows   = self._n_rows
        n_cols   = self._n_cols

        if snap["gridmap"] is not None:
            gm = snap["gridmap"]
            if gm.shape != (n_layers, n_rows, n_cols):
                gm = np.zeros((n_layers, n_rows, n_cols), dtype=np.float32)
            gridmap = np.nan_to_num(gm, nan=0.0)
        else:
            gridmap = np.zeros((n_layers, n_rows, n_cols), dtype=np.float32)

        dx     = snap["goal_x"] - snap["robot_x"]
        dy     = snap["goal_y"] - snap["robot_y"]
        dtheta = snap["goal_yaw"] - snap["robot_yaw"]
        goal_vec = np.array(
            [dx, dy, math.cos(dtheta), math.sin(dtheta)], dtype=np.float32
        )

        energy_vec = np.array([
            float(np.clip(snap["inst_power"] / self._p_max, 0.0, 1.0)),
            float(np.clip(snap["avg_power"]  / self._p_max, 0.0, 1.0)),
            float(np.clip(snap["cum_energy"] / (self._p_max * self._max_steps * 0.05), 0.0, 1.0)),
        ], dtype=np.float32)

        robot_vel = np.array([
            float(np.clip(snap["robot_vx"]    / self._vx_max,    -1.0, 1.0)),
            float(np.clip(snap["robot_vy"]    / self._vy_max,    -1.0, 1.0)),
            float(np.clip(snap["robot_omega"] / self._omega_max, -1.0, 1.0)),
        ], dtype=np.float32)

        return {
            "gridmap":    gridmap.astype(np.float32),
            "goal_vec":   goal_vec,
            "energy_vec": energy_vec,
            "robot_vel":  robot_vel,
        }

    # ======================================================================
    # Helper: distanza al goal
    # ======================================================================
    def _dist_to_goal(self, snap: dict) -> float:
        dx = snap["goal_x"] - snap["robot_x"]
        dy = snap["goal_y"] - snap["robot_y"]
        return math.sqrt(dx * dx + dy * dy)

    # ======================================================================
    # Helper: pubblicazione /cmd_vel
    # ======================================================================
    def _publish_cmd(self, vx: float, vy: float, omega: float) -> None:
        twist = Twist()
        twist.linear.x  = float(vx)
        twist.linear.y  = float(vy)
        twist.angular.z = float(omega)
        self._pub_cmd.publish(twist)

    # ======================================================================
    # Helper: caricamento config
    # ======================================================================
    def _load_config(self, config_path: str) -> dict:
        # Cerca il file config in posizioni standard se non specificato
        candidates = [
            Path(config_path),
            Path(__file__).parent.parent / "config" / "rl_params.yaml",
            Path("config/rl_params.yaml"),
        ]
        for c in candidates:
            if c.exists():
                with open(c, "r") as f:
                    cfg = yaml.safe_load(f)
                self.get_logger().info(f"Config caricata da: {c}")
                return cfg or {}
        self.get_logger().warn(
            f"Config non trovata in '{config_path}', usando default."
        )
        return {}


# ===========================================================================
# Entry point
# ===========================================================================
def main(args=None) -> None:
    rclpy.init(args=args)
    node = RLControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

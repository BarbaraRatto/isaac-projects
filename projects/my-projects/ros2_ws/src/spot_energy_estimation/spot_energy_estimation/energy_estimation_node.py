#!/usr/bin/env python3
"""
Nodo di stima del consumo energetico del robot quadrupede.

Sottoscrive /joint_states e calcola, per ciascun giunto attuato:

    P_j(t) = |tau_j(t) * theta_dot_j(t)|

sommando poi sui giunti per ottenere la potenza istantanea totale.
Integra nel tempo (metodo trapezoidale) per ottenere l'energia cumulata,
e mantiene una media mobile della potenza su una finestra temporale
configurabile per ridurre il rumore.

Pubblica il risultato su /energy/current_consumption come messaggio
custom energy_msgs/msg/EnergyEstimate.
"""

from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from energy_msgs.msg import EnergyEstimate


class EnergyEstimationNode(Node):

    def __init__(self):
        super().__init__('energy_estimation_node')

        # --- Parametri configurabili ---
        self.declare_parameter('moving_average_window', 0.5)
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('energy_topic', '/energy/current_consumption')
        self.declare_parameter('frame_id', 'base_link')

        self._window_s = self.get_parameter(
            'moving_average_window').get_parameter_value().double_value
        joint_states_topic = self.get_parameter(
            'joint_states_topic').get_parameter_value().string_value
        energy_topic = self.get_parameter(
            'energy_topic').get_parameter_value().string_value
        self._frame_id = self.get_parameter(
            'frame_id').get_parameter_value().string_value

        # --- Stato interno ---
        self._cumulative_energy = 0.0   # [J]
        self._last_stamp_s = None       # timestamp precedente [s], per l'integrazione
        self._last_power = 0.0          # potenza istantanea precedente [W], per il trapezio

        # Buffer (timestamp, potenza) per la media mobile a finestra temporale
        self._power_history = deque()

        # --- Publisher / Subscriber ---
        self._pub = self.create_publisher(EnergyEstimate, energy_topic, 10)
        self._sub = self.create_subscription(
            JointState, joint_states_topic, self._joint_states_callback, 10)

        self.get_logger().info(
            f"Energy estimation node avviato. "
            f"Input: '{joint_states_topic}', Output: '{energy_topic}', "
            f"finestra media mobile: {self._window_s} s"
        )

    def _joint_states_callback(self, msg: JointState):
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Isaac Sim potrebbe non popolare 'effort' per qualche pubblicazione
        # transitoria: se le liste non sono coerenti in lunghezza, scartiamo
        # il messaggio invece di crashare.
        n = len(msg.name)
        if len(msg.velocity) != n or len(msg.effort) != n:
            self.get_logger().warn(
                'Lunghezze incoerenti in /joint_states (name/velocity/effort); '
                'messaggio scartato.',
                throttle_duration_sec=5.0
            )
            return

        # --- Potenza per giunto e potenza totale istantanea ---
        joint_power = [abs(tau * vel) for tau, vel in zip(msg.effort, msg.velocity)]
        instantaneous_power = sum(joint_power)

        # --- Integrazione trapezoidale per l'energia cumulata ---
        if self._last_stamp_s is not None:
            dt = stamp_s - self._last_stamp_s
            # Protezione contro dt negativo/nullo (es. riavvio della sim,
            # primo messaggio dopo un salto temporale, o messaggi fuori ordine)
            if 0.0 < dt < 1.0:
                self._cumulative_energy += 0.5 * (self._last_power + instantaneous_power) * dt

        self._last_stamp_s = stamp_s
        self._last_power = instantaneous_power

        # --- Media mobile a finestra temporale ---
        self._power_history.append((stamp_s, instantaneous_power))
        while self._power_history and (stamp_s - self._power_history[0][0]) > self._window_s:
            self._power_history.popleft()
        average_power = sum(p for _, p in self._power_history) / len(self._power_history)

        # --- Costruzione e pubblicazione del messaggio ---
        out = EnergyEstimate()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._frame_id
        out.instantaneous_power = instantaneous_power
        out.average_power = average_power
        out.cumulative_energy = self._cumulative_energy
        out.joint_names = list(msg.name)
        out.joint_power = joint_power

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = EnergyEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

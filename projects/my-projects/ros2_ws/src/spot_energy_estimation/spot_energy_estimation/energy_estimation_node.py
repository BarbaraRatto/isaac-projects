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


from std_msgs.msg import Float32

class EnergyEstimationNode(Node):

    def __init__(self):
        super().__init__('energy_estimation_node')

        # --- Parametri configurabili ---
        self.declare_parameter('moving_average_window', 5.0)            # changed from 0.5 s to 5.0 s
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
        
        # --- Variabili per il Benchmark on-demand ---
        self._benchmark_active = False
        self._benchmark_end_time = 0.0
        self._benchmark_energy = 0.0
        self._benchmark_duration = 0.0

        # --- Publisher / Subscriber ---
        self._pub = self.create_publisher(EnergyEstimate, energy_topic, 10)
        self._sub = self.create_subscription(
            JointState, joint_states_topic, self._joint_states_callback, 10)
            
        self._benchmark_sub = self.create_subscription(
            Float32, '/energy/start_benchmark', self._benchmark_cb, 10)

        self.get_logger().info(
            f"Energy estimation node avviato. "
            f"Input: '{joint_states_topic}', Output: '{energy_topic}', "
            f"finestra media mobile: {self._window_s} s"
        )
        
    def _benchmark_cb(self, msg: Float32):
        if self._last_stamp_s is None:
            self.get_logger().warn("Nessun dato da /joint_states. Avvia la simulazione prima del benchmark!")
            return
        if msg.data <= 0:
            self.get_logger().warn("La durata del benchmark deve essere > 0 s.")
            return
            
        self._benchmark_duration = float(msg.data)
        self._benchmark_end_time = self._last_stamp_s + self._benchmark_duration
        self._benchmark_energy = 0.0
        self._benchmark_active = True
        self.get_logger().info(f"\n---> BENCHMARK AVVIATO per {self._benchmark_duration} secondi <---")

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
            # Protezione contro dt negativo/nullo
            if 0.0 < dt < 1.0:
                step_energy = 0.5 * (self._last_power + instantaneous_power) * dt
                self._cumulative_energy += step_energy
                
                # Calcolo per il benchmark on-demand
                if self._benchmark_active:
                    self._benchmark_energy += step_energy
                    if stamp_s >= self._benchmark_end_time:
                        avg_power = self._benchmark_energy / self._benchmark_duration
                        self.get_logger().info(
                            f"\n======================================================\n"
                            f" BENCHMARK COMPLETATO!\n"
                            f" Durata: {self._benchmark_duration} s\n"
                            f" Energia Consumata: {self._benchmark_energy:.2f} Joules\n"
                            f" Potenza Media: {avg_power:.2f} Watt\n"
                            f"======================================================\n"
                        )
                        self._benchmark_active = False

        self._last_stamp_s = stamp_s
        self._last_power = instantaneous_power

        # --- Media mobile a finestra temporale ---
        self._power_history.append((stamp_s, instantaneous_power))
        while self._power_history and (stamp_s - self._power_history[0][0]) > self._window_s:
            self._power_history.popleft()
        average_power = sum(p for _, p in self._power_history) / len(self._power_history)

        # --- Costruzione e pubblicazione del messaggio ---
        if not hasattr(self, '_last_publish_s'):
            self._last_publish_s = stamp_s
            
        # Pubblichiamo solo ogni _window_s secondi per non intasare il terminale
        if (stamp_s - self._last_publish_s) >= self._window_s:
            out = EnergyEstimate()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = self._frame_id
            out.instantaneous_power = instantaneous_power
            out.average_power = average_power
            out.cumulative_energy = self._cumulative_energy
            out.joint_names = list(msg.name)
            out.joint_power = joint_power
    
            self._pub.publish(out)
            self._last_publish_s = stamp_s


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

import os

# Fail-safe: if not explicitly chosen otherwise,
# ROS 2 communicates only on the local machine.
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

print(
    "ROS 2 network mode:",
    "LOCALHOST" if os.environ["ROS_LOCALHOST_ONLY"] == "1" else "NETWORK",
)

import sys
import shlex
import subprocess
import time
from pathlib import Path

dir_path = Path(__file__).resolve().parent
sys.path.append(str(dir_path / ".."))

ros_ws = dir_path / "msgs_ws"
setup_bash = ros_ws / "install" / "setup.bash"

if not setup_bash.exists():
    print("Building the msgs first...")
    subprocess.run(["colcon", "build"], cwd=ros_ws, check=True)

if os.environ.get("QUADRUPED_PYMPC_ROS2_SOURCED") != "1":
    print("Sourcing ROS2 workspace and restarting script...")
    cmd = (
        f"source {shlex.quote(str(setup_bash))} && "
        "export QUADRUPED_PYMPC_ROS2_SOURCED=1 && "
        f"exec {shlex.quote(sys.executable)} "
        + " ".join(shlex.quote(arg) for arg in [str(Path(__file__).resolve()), *sys.argv[1:]])
    )
    os.execv("/bin/bash", ["bash", "-c", cmd])

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

# Importa i messaggi custom del tuo controllore
from dls2_interface.msg import BaseState, BlindState, ControlSignal, TrajectoryGenerator
import numpy as np
from scipy.spatial.transform import Rotation

from quadruped_pympc import config as cfg


MPC_JOINT_NAMES = [
    'fl_hx', 'fl_hy', 'fl_kn',
    'fr_hx', 'fr_hy', 'fr_kn',
    'hl_hx', 'hl_hy', 'hl_kn',
    'hr_hx', 'hr_hy', 'hr_kn',
]

# Native DoF order reported by the Isaac Sim Spot articulation.  Publishing in
# this order is safe both when Articulation Controller uses jointNames and when
# it applies the arrays directly by index.
ISAAC_JOINT_NAMES = [
    'fl_hx', 'fr_hx', 'hl_hx', 'hr_hx',
    'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy',
    'fl_kn', 'fr_kn', 'hl_kn', 'hr_kn',
]
MPC_TO_ISAAC = np.array(
    [MPC_JOINT_NAMES.index(name) for name in ISAAC_JOINT_NAMES], dtype=int
)

# Limits authored in the Isaac Sim Spot USD (hip, thigh, knee), in Nm.
ISAAC_EFFORT_LIMITS = np.tile([45.0, 45.0, 115.0], 4)
STANDUP_EFFORT_LIMITS = np.tile(
    np.asarray(
        cfg.simulation_params.get('standup_joint_effort_limits', [10.0, 40.0, 65.0]),
        dtype=float,
    ),
    4,
)


def expand_joint_values(value, name):
    """Return scalar, one-leg, or full-robot configuration as 12 values."""
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == 1:
        return np.full(12, values.item())
    if values.size == 3:
        return np.tile(values, 4)
    if values.size == 12:
        return values.copy()
    raise ValueError(f"{name} must contain 1, 3, or 12 values")


NOMINAL_POSITION_GAINS = expand_joint_values(
    cfg.simulation_params['impedence_joint_position_gain'],
    'impedence_joint_position_gain',
)
STANDUP_POSITION_GAINS = expand_joint_values(
    cfg.simulation_params.get('standup_joint_position_gain', [12.0, 30.0, 30.0]),
    'standup_joint_position_gain',
)

class IsaacRos2Translator(Node):
    def __init__(self):
        super().__init__('isaac_ros2_translator')

        self.declare_parameter('odom_is_relative', True)
        self.declare_parameter(
            'initial_base_height',
            float(cfg.simulation_params.get('isaac_odom_origin_height', cfg.hip_height)),
        )
        self.declare_parameter('command_mode', 'external_pd')
        self.declare_parameter('effort_limit_scale', 0.9)

        self.odom_is_relative = bool(self.get_parameter('odom_is_relative').value)
        self.initial_base_height = float(self.get_parameter('initial_base_height').value)
        self.command_mode = str(self.get_parameter('command_mode').value)
        self.effort_limit_scale = float(self.get_parameter('effort_limit_scale').value)
        if self.command_mode not in ('external_pd', 'isaac_pd'):
            raise ValueError("command_mode must be 'external_pd' or 'isaac_pd'")

        if self.odom_is_relative:
            self.get_logger().info(
                f"Relative odometry: adding chassis origin height "
                f"{self.initial_base_height:.6f} m"
            )

        self.joint_mapping_validated = False
        self.command_joint_names = ISAAC_JOINT_NAMES[:]
        self.mpc_to_command = MPC_TO_ISAAC.copy()

        # ---------------------------------------------------------
        # 1. DA ISAAC SIM -> AL CONTROLLORE
        # ---------------------------------------------------------
        # Ascolta i messaggi standard da Isaac Sim (Nomi aggiornati in base a ros2 topic list)
        self.sub_isaac_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.sub_isaac_joints = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)

        # Pubblica i messaggi custom per il controllore
        self.pub_base_state = self.create_publisher(BaseState, '/base_state', 10)
        self.pub_blind_state = self.create_publisher(BlindState, '/blind_state', 10)

        # ---------------------------------------------------------
        # 3. CONTATTI DELLE ZAMPE
        # ---------------------------------------------------------
        self.contact_forces = [0.0, 0.0, 0.0, 0.0]
        self.sub_fl_contact = self.create_subscription(Float32, '/fl/contact', lambda msg: self.contact_callback(msg, 0), 10)
        self.sub_fr_contact = self.create_subscription(Float32, '/fr/contact', lambda msg: self.contact_callback(msg, 1), 10)
        self.sub_hl_contact = self.create_subscription(Float32, '/hl/contact', lambda msg: self.contact_callback(msg, 2), 10)
        self.sub_hr_contact = self.create_subscription(Float32, '/hr/contact', lambda msg: self.contact_callback(msg, 3), 10)

        # ---------------------------------------------------------
        # 4. DAL CONTROLLORE -> AD ISAAC SIM
        # ---------------------------------------------------------
        # Variabili per salvare la posizione e velocità desiderate dal trajectory generator
        self.desired_positions = []
        self.desired_velocities = []
        self.kp_gains = []
        self.kd_gains = []
        self.desired_timestamp = None
        self.last_trajectory_rx_monotonic = None
        self.last_fail_safe_warning = 0.0
        self.current_positions = []
        self.current_velocities = []

        self.sub_trajectory = self.create_subscription(
            TrajectoryGenerator, '/trajectory_generator', self.trajectory_callback, 10)

        # Ascolta il segnale di controllo custom
        self.sub_control = self.create_subscription(
            ControlSignal, '/control_signal', self.control_callback, 10)
        
        # Pubblica i comandi in formato standard per Isaac Sim (Nodo Articulation Controller)
        self.pub_isaac_commands = self.create_publisher(JointState, '/isaac/joint_command', 10)

        if self.command_mode == 'external_pd':
            self.get_logger().warning(
                "external_pd: Isaac Sim must use effort control with every joint drive "
                "stiffness/damping set to zero; connect only effortCommand."
            )
        else:
            self.get_logger().warning(
                "isaac_pd: connect positionCommand, velocityCommand and effortCommand, "
                "and set the Isaac joint-drive gains to kp=10, kd=2."
            )


    def trajectory_callback(self, msg: TrajectoryGenerator):
        # Salva le posizioni e velocità target calcolate dall'MPC
        self.desired_positions = list(msg.joints_position)
        self.desired_velocities = list(msg.joints_velocity)
        self.kp_gains = list(msg.kp)
        self.kd_gains = list(msg.kd)
        self.desired_timestamp = float(msg.timestamp)
        self.last_trajectory_rx_monotonic = time.monotonic()

    def publish_zero_effort(self, reason: str):
        """Actively clear Isaac's last effort command when input is unsafe."""
        joint_cmd_msg = JointState()
        joint_cmd_msg.name = self.command_joint_names
        joint_cmd_msg.effort = [0.0] * 12
        self.pub_isaac_commands.publish(joint_cmd_msg)

        now = time.monotonic()
        if now - self.last_fail_safe_warning >= 1.0:
            self.get_logger().warning(f"Fail-safe: zero effort ({reason})")
            self.last_fail_safe_warning = now


    def odom_callback(self, msg: Odometry):
        # Converte Odometry (standard) in BaseState (custom)
        base_msg = BaseState()
        odom_z = msg.pose.pose.position.z
        if self.odom_is_relative:
            # IsaacComputeOdometry reports displacement from the pose at timeline
            # start, where Spot is spawned standing at initial_base_height.
            # Do not zero against the first message received: the translator may
            # start after the drives were disabled and the robot already fell.
            odom_z = odom_z + self.initial_base_height

        base_msg.timestamp = (
            float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        )
        base_msg.pose.position = [msg.pose.pose.position.x, msg.pose.pose.position.y, odom_z]
        base_msg.pose.orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                                     msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        
        # La twist di ROS /odom è nel child_frame (body frame).
        # Il controllore mette la velocità lineare in MuJoCo qvel[0:3] che si aspetta il WORLD frame.
        # Convertiamo: v_world = R_body_to_world @ v_body
        quat_xyzw = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        R_body_to_world = Rotation.from_quat(quat_xyzw).as_matrix()
        
        lin_vel_body = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        lin_vel_world = R_body_to_world @ lin_vel_body
        base_msg.velocity.linear = lin_vel_world.tolist()
        
        # La velocità angolare ROS è già in body frame, che è ciò che
        # MuJoCo si aspetta per qvel[3:6]. Non serve conversione.
        base_msg.velocity.angular = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        
        self.pub_base_state.publish(base_msg)

    def contact_callback(self, msg: Float32, leg_idx: int):
        # Aggiorna la forza percepita per la zampa specifica
        self.contact_forces[leg_idx] = msg.data

    def joint_states_callback(self, msg: JointState):
        # Il controllore si aspetta i giunti in ordine: FL, FR, HL, HR (3 per zampa)
        # Ma Isaac Sim potrebbe pubblicarli in un ordine diverso (es. tutti gli hx, poi hy, poi kn)
        # Creiamo un dizionario temporaneo per mappare i nomi ai valori
        pos_dict = dict(zip(msg.name, msg.position))
        vel_dict = dict(zip(msg.name, msg.velocity))

        missing_position = [name for name in MPC_JOINT_NAMES if name not in pos_dict]
        missing_velocity = [name for name in MPC_JOINT_NAMES if name not in vel_dict]
        if missing_position or missing_velocity:
            self.get_logger().error(
                "Rejecting /joint_states: missing positions "
                f"{missing_position}, missing velocities {missing_velocity}. "
                f"Received names: {list(msg.name)}"
            )
            return

        ordered_positions = [pos_dict[name] for name in MPC_JOINT_NAMES]
        ordered_velocities = [vel_dict[name] for name in MPC_JOINT_NAMES]

        if not self.joint_mapping_validated:
            # Mirror the actual articulation order on the command topic.  This
            # also covers graphs where jointNames is not wired and Isaac uses
            # the array indices directly.
            command_joint_names = [
                name for name in msg.name if name in MPC_JOINT_NAMES
            ]
            if len(command_joint_names) == 12:
                self.command_joint_names = command_joint_names
                self.mpc_to_command = np.array([
                    MPC_JOINT_NAMES.index(name)
                    for name in self.command_joint_names
                ], dtype=int)
            self.get_logger().info(
                "Validated joint mapping. Isaac command order: "
                + ', '.join(self.command_joint_names)
            )
            self.joint_mapping_validated = True

        # Converte JointState (standard) in BlindState (custom)
        blind_msg = BlindState()
        blind_msg.timestamp = (
            float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        )
        blind_msg.joints_name = MPC_JOINT_NAMES
        blind_msg.joints_position = ordered_positions
        blind_msg.joints_velocity = ordered_velocities
        
        # Salva lo stato corrente per il calcolo PD nel control_callback
        self.current_positions = ordered_positions[:]
        self.current_velocities = ordered_velocities[:]
        
        # Converte la forza dei sensori in stato di contatto (Soglia: 20)
        blind_msg.feet_contact = [force > 20.0 for force in self.contact_forces]
        
        self.pub_blind_state.publish(blind_msg)

    def control_callback(self, msg: ControlSignal):
        # Converte ControlSignal (custom) in JointState (standard) per comandare Isaac Sim
        joint_cmd_msg = JointState()
        
        joint_cmd_msg.name = self.command_joint_names
        
        # Torque feedforward dal controllore MPC
        torques = np.array(msg.torques, dtype=np.float64)
        
        state_complete = (
            len(self.desired_positions) == 12
            and len(self.desired_velocities) == 12
            and len(self.current_positions) == 12
            and len(self.current_velocities) == 12
            and len(self.kp_gains) == 12
            and len(self.kd_gains) == 12
        )

        if len(torques) != 12 or not np.all(np.isfinite(torques)):
            self.publish_zero_effort("invalid torque command")
            return

        trajectory_age = (
            time.monotonic() - self.last_trajectory_rx_monotonic
            if self.last_trajectory_rx_monotonic is not None
            else float('inf')
        )
        timestamps_match = (
            self.desired_timestamp is not None
            and abs(self.desired_timestamp - float(msg.timestamp)) <= 1e-6
        )

        if not state_complete:
            self.publish_zero_effort("incomplete state or trajectory")
            return

        trajectory_values = np.concatenate([
            self.desired_positions,
            self.desired_velocities,
            self.kp_gains,
            self.kd_gains,
            self.current_positions,
            self.current_velocities,
        ])
        if not np.all(np.isfinite(trajectory_values)):
            self.publish_zero_effort("non-finite state or trajectory")
            return

        if trajectory_age > 0.1 or not timestamps_match:
            self.publish_zero_effort("stale or mismatched trajectory")
            return

        if self.command_mode == 'external_pd':
            
            q_des = np.array(self.desired_positions)
            q_cur = np.array(self.current_positions)
            qd_des = np.array(self.desired_velocities)
            qd_cur = np.array(self.current_velocities)
            kp = np.array(self.kp_gains)
            kd = np.array(self.kd_gains)
            
            torques = torques + kp * (q_des - q_cur) + kd * (qd_des - qd_cur)

        elif self.command_mode == 'isaac_pd':
            joint_cmd_msg.position = np.asarray(self.desired_positions)[self.mpc_to_command].tolist()
            joint_cmd_msg.velocity = np.asarray(self.desired_velocities)[self.mpc_to_command].tolist()

        # Infer the same stand-up -> MPC blend already encoded in the received
        # kp values and use it to interpolate the effort limits.  The previous
        # binary selection released the hx cap from 9 to 40.5 Nm in one sample
        # when kp reached its nominal value, exactly after "goUp completed".
        kp_now = np.asarray(self.kp_gains, dtype=float)
        gain_span = STANDUP_POSITION_GAINS - NOMINAL_POSITION_GAINS
        changing_gains = np.abs(gain_span) > 1e-9
        if np.any(changing_gains):
            blend_components = (
                (STANDUP_POSITION_GAINS[changing_gains] - kp_now[changing_gains])
                / gain_span[changing_gains]
            )
            support_blend = float(
                np.clip(np.median(blend_components), 0.0, 1.0)
            )
        else:
            support_blend = 1.0
        effort_limits = (
            (1.0 - support_blend) * STANDUP_EFFORT_LIMITS
            + support_blend * ISAAC_EFFORT_LIMITS
        ) * self.effort_limit_scale
        effort_mpc_order = np.clip(torques, -effort_limits, effort_limits)
        joint_cmd_msg.effort = effort_mpc_order[self.mpc_to_command].tolist()
        
        self.pub_isaac_commands.publish(joint_cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacRos2Translator()
    print("Nodo Traduttore avviato! In attesa di messaggi tra Isaac Sim e il Controllore...")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import os
import sys
import numpy as np
import shlex
import subprocess
from pathlib import Path

# ==============================================================================
# Setup ROS 2 Environment and Custom Messages
# ==============================================================================
# Fail-safe: if not explicitly chosen otherwise, communicate only locally.
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

# This path assumes the script is run from inside the my-projects/isaacsim folder
# Adjust if necessary
controller_dir = Path(__file__).resolve().parent.parent / "iit_controller" / "Quadruped-PyMPC"
ros_ws = controller_dir / "ros2" / "msgs_ws"
setup_bash = ros_ws / "install" / "setup.bash"

if not setup_bash.exists():
    print(f"ERROR: Cannot find ROS2 workspace setup at {setup_bash}. Please build msgs first.")
    sys.exit(1)

# Ensure ROS2 environment is sourced before importing rclpy inside Isaac Sim
if os.environ.get("QUADRUPED_PYMPC_ROS2_SOURCED") != "1":
    print("Sourcing ROS2 workspace and restarting script...")
    cmd = (
        f"source {shlex.quote(str(setup_bash))} && "
        "export QUADRUPED_PYMPC_ROS2_SOURCED=1 && "
        f"exec {shlex.quote(sys.executable)} "
        + " ".join(shlex.quote(arg) for arg in [str(Path(__file__).resolve()), *sys.argv[1:]])
    )
    os.execv("/bin/bash", ["bash", "-c", cmd])

# ==============================================================================
# Isaac Sim Initialization
# ==============================================================================
from isaacsim import SimulationApp
# Start Isaac Sim Simulation Application
simulation_app = SimulationApp({"headless": False})

# ==============================================================================
# ROS 2 and Isaac Sim Imports (Must be done after SimulationApp starts)
# ==============================================================================
import rclpy
from rclpy.node import Node
from dls2_interface.msg import BaseState, BlindState, ControlSignal

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation

# ==============================================================================
# Bridge Node Definition
# ==============================================================================
class IsaacROS2BridgeNode(Node):
    def __init__(self, robot_articulation: Articulation):
        super().__init__('isaac_ros2_bridge_node')
        
        self.robot = robot_articulation
        
        # Publishers
        self.publisher_base_state = self.create_publisher(BaseState, "/base_state", 1)
        self.publisher_blind_state = self.create_publisher(BlindState, "/blind_state", 1)
        
        # Subscriber
        self.subscriber_control_signal = self.create_subscription(
            ControlSignal, "/control_signal", self.control_callback, 1)
        
        # Keep track of latest torques
        self.desired_torques = np.zeros(12)  # Assuming 12 DoF quadruped

    def control_callback(self, msg):
        """Called whenever a new control signal is received from the MPC."""
        # msg.torques dovrebbero essere 12 valori (FL, FR, RL, RR)
        self.desired_torques = np.array(msg.torques)

    def publish_states(self):
        """Called every simulation step to publish robot states to ROS 2."""
        if not self.robot.initialized:
            return

        # 1. Get Base State
        base_pos, base_quat = self.robot.get_world_pose()
        # Isaac returns quat as [w, x, y, z]. If your controller expects [x, y, z, w], you must reorder:
        base_lin_vel = self.robot.get_linear_velocity()
        base_ang_vel = self.robot.get_angular_velocity()
        
        base_msg = BaseState()
        base_msg.pose.position = base_pos.tolist()
        base_msg.pose.orientation = base_quat.tolist() # Check quaternion format!
        base_msg.velocity.linear = base_lin_vel.tolist()
        base_msg.velocity.angular = base_ang_vel.tolist()
        
        self.publisher_base_state.publish(base_msg)

        # 2. Get Joint States
        # Assicurati che l'ordine dei giunti corrisponda a quello che si aspetta il controllore!
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        
        blind_msg = BlindState()
        blind_msg.joints_position = joint_positions.tolist()
        blind_msg.joints_velocity = joint_velocities.tolist()
        
        self.publisher_blind_state.publish(blind_msg)

    def apply_control(self):
        """Called every simulation step to apply torques to Isaac Sim."""
        if not self.robot.initialized:
            return
        # Applica le coppie ai giunti. Anche qui, l'ordine è cruciale.
        self.robot.set_joint_efforts(self.desired_torques)

# ==============================================================================
# Main Execution Loop
# ==============================================================================
def main():
    rclpy.init()

    # Create World
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    # TODO: Inserisci qui il path corretto per l'USD del tuo robot
    robot_usd_path = "/home/students/work/barbara/isaac-projects/projects/my-projects/Graph.usd" # Cambia con l'USD reale
    
    # Aggiungi il robot alla scena come Articulation
    from omni.isaac.core.utils.stage import add_reference_to_stage
    add_reference_to_stage(usd_path=robot_usd_path, prim_path="/World/Robot")
    
    robot = Articulation(prim_path="/World/Robot", name="quadruped")
    world.scene.add(robot)
    
    world.reset()
    
    # Inizializza il nodo ROS 2 Bridge
    bridge_node = IsaacROS2BridgeNode(robot)

    # Simulation loop
    while simulation_app.is_running():
        # Spin ROS 2 to process callbacks
        rclpy.spin_once(bridge_node, timeout_sec=0.0)
        
        # Apply the latest received torques
        bridge_node.apply_control()
        
        # Step simulation physics
        world.step(render=True)
        
        # Publish new state
        bridge_node.publish_states()

    # Cleanup
    bridge_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == '__main__':
    main()

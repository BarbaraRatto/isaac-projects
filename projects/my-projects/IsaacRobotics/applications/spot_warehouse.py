import os
import sys

# Isaac Sim 5.1 uses Python 3.11; system ROS Humble has Python 3.10 bindings.
# Make the bundled ROS 2 bridge visible before SimulationApp starts.
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ["PYTHONPATH"] = os.pathsep.join(
    path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if "/opt/ros" not in path
)
sys.path = [path for path in sys.path if "/opt/ros" not in path]

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
from pathlib import Path
import omni.appwindow
import omni.graph.core as og
import omni.usd
import cv2

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from spot_policy import SpotFlatTerrainPolicy, SpotArmFlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.utils.extensions import enable_extension
enable_extension('isaacsim.ros2.bridge')
from spot_camera import attach_zed_camera, create_zed_ros_graph, publish_zed_static_tf

# Import di ROS2 integrato in Isaac Sim (DEVE avvenire dopo l'enable_extension)
import rclpy
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
from builtin_interfaces.msg import Time

class SpotRunner(object):
    def __init__(self, physics_dt, render_dt) -> None:
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        # Spawn il nostro nuovo terreno randomico invece della warehouse
        prim = define_prim("/World/Warehouse", "Xform")
        asset_path = "/home/students/work/barbara/isaac-projects/projects/my-projects/terrain_generator/real_terrains.usd"
        prim.GetReferences().AddReference(asset_path)

        # Aggiungiamo la luce di default (Default Light Rig)
        from isaacsim.core.utils.prims import create_prim
        create_prim(
            "/World/defaultLight",
            "DistantLight",
            attributes={"inputs:intensity": 3000.0, "inputs:angle": 1.0}
        )

        # -------------------------------------------------------------

        # SPOT WITH THE ARM
        """
        BASE_DIR = Path(__file__).resolve().parent.parent
        policy_path = os.path.join(BASE_DIR, "policies/spot_arm/models", "spot_arm_policy.pt")
        policy_params_path = os.path.join(BASE_DIR, "policies/spot_arm/params", "env.yaml")
        usd_path = os.path.join(BASE_DIR, "assets", "spot_arm.usd")

        self._spot = SpotArmFlatTerrainPolicy(
        """    
        # -------------------------------------------------------------

        # SPOT WITHOUT THE ARM
        BASE_DIR = Path(__file__).resolve().parent.parent
        policy_path = os.path.join(BASE_DIR, "policies/spot/models/spot_policy.pt")
        policy_params_path = os.path.join(BASE_DIR, "policies/spot/params", "env.yaml")
        usd_path = os.path.join(BASE_DIR, "assets", "spot.usd")

        self._spot = SpotFlatTerrainPolicy(
        # -------------------------------------------------------------
            prim_path="/World/Spot",
            name="Spot",
            usd_path=usd_path,
            policy_path=policy_path,
            policy_params_path=policy_params_path,
            position=np.array([0, 9, 1.1]),
        )

        # La ZED e i publisher ROS sono ricreati ad ogni avvio dello script.
        self._zed_camera_path = attach_zed_camera(
            omni.usd.get_context().get_stage(), assets_root_path, "/World/Spot"
        )

        self._base_command = np.zeros(3)
        self._input_keyboard_mapping = {
            "NUMPAD_8": [1.0, 0.0, 0.0], "UP": [1.0, 0.0, 0.0],
            "NUMPAD_2": [-1.0, 0.0, 0.0], "DOWN": [-1.0, 0.0, 0.0],
            "NUMPAD_6": [0.0, -1.0, 0.0], "RIGHT": [0.0, -1.0, 0.0],
            "NUMPAD_4": [0.0, 1.0, 0.0], "LEFT": [0.0, 1.0, 0.0],
            "NUMPAD_7": [0.0, 0.0, 1.0], "N": [0.0, 0.0, 1.0],
            "NUMPAD_9": [0.0, 0.0, -1.0], "M": [0.0, 0.0, -1.0],
        }

        self.needs_reset = False
        self.first_step = True
        self._zed_graph_needs_rearm = False

    def setup(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._sub_keyboard_event
        )
        self._world.add_physics_callback("spot_forward", callback_fn=self.on_physics_step)

        # Inizializziamo rclpy per pubblicare manualmente i JointState su ROS2
        try:
            if not rclpy.ok():
                rclpy.init()
            self.ros_node = rclpy.create_node('spot_standalone_js_publisher')
            self.js_pub = self.ros_node.create_publisher(JointState, '/joint_states', 10)
            
            # --- AGGIUNTE PER NAV2 ---
            self.odom_pub = self.ros_node.create_publisher(Odometry, '/odom', 10)
            self.tf_pub = self.ros_node.create_publisher(TFMessage, '/tf', 10)
            self.clock_pub = self.ros_node.create_publisher(Clock, '/clock', 10)
            self.cmd_vel_sub = self.ros_node.create_subscription(Twist, '/cmd_vel', self._cmd_vel_callback, 10)
            self._zed_tf_broadcaster = publish_zed_static_tf(self.ros_node)

            print("[ROS2] Nodi creati con successo (JointStates, Odom, TF, Clock, CmdVel, ZED_X).")
        except Exception as e:
            carb.log_error(f"Errore nella creazione del nodo ROS2: {e}")

    def _cmd_vel_callback(self, msg: Twist):
        # Mappa i comandi Twist di Nav2 su _base_command [vx, vy, wz]
        self._base_command[0] = msg.linear.x
        self._base_command[1] = msg.linear.y
        self._base_command[2] = msg.angular.z

    def on_physics_step(self, step_size) -> None:
        if self.first_step:
            self._spot.initialize()
            self.first_step = False
        elif self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
        else:
            self._spot.forward(step_size, self._base_command)
            
            # Pubblichiamo i JointState su ROS2 manualmente
            if hasattr(self, 'js_pub') and self.js_pub is not None:
                # Creazione del timestamp di simulazione per tutti i messaggi
                sim_time = self._world.current_time
                sec = int(sim_time)
                nanosec = int((sim_time - sec) * 1e9)
                stamp = Time(sec=sec, nanosec=nanosec)
                
                # --- PUBLISH CLOCK ---
                clock_msg = Clock()
                clock_msg.clock = stamp
                self.clock_pub.publish(clock_msg)
                
                # --- RECUPERO STATO ROBOT ---
                pos, quat = self._spot.robot.get_world_pose()
                lin_vel = self._spot.robot.get_linear_velocity()
                ang_vel = self._spot.robot.get_angular_velocity()
                
                if pos is not None and quat is not None:
                    # --- PUBLISH ODOMETRY ---
                    odom_msg = Odometry()
                    odom_msg.header.stamp = stamp
                    odom_msg.header.frame_id = "odom"
                    odom_msg.child_frame_id = "base_link"
                    odom_msg.pose.pose.position.x = float(pos[0])
                    odom_msg.pose.pose.position.y = float(pos[1])
                    odom_msg.pose.pose.position.z = float(pos[2])
                    # In Isaac Sim quat è [w, x, y, z], ma anche il msg ROS2 si aspetta campi w, x, y, z separati
                    odom_msg.pose.pose.orientation.w = float(quat[0])
                    odom_msg.pose.pose.orientation.x = float(quat[1])
                    odom_msg.pose.pose.orientation.y = float(quat[2])
                    odom_msg.pose.pose.orientation.z = float(quat[3])
                    odom_msg.twist.twist.linear.x = float(lin_vel[0])
                    odom_msg.twist.twist.linear.y = float(lin_vel[1])
                    odom_msg.twist.twist.linear.z = float(lin_vel[2])
                    odom_msg.twist.twist.angular.x = float(ang_vel[0])
                    odom_msg.twist.twist.angular.y = float(ang_vel[1])
                    odom_msg.twist.twist.angular.z = float(ang_vel[2])
                    self.odom_pub.publish(odom_msg)
                    
                    # --- PUBLISH TF ---
                    t = TransformStamped()
                    t.header.stamp = stamp
                    t.header.frame_id = "odom"
                    t.child_frame_id = "base_link"
                    t.transform.translation.x = float(pos[0])
                    t.transform.translation.y = float(pos[1])
                    t.transform.translation.z = float(pos[2])
                    t.transform.rotation.w = float(quat[0])
                    t.transform.rotation.x = float(quat[1])
                    t.transform.rotation.y = float(quat[2])
                    t.transform.rotation.z = float(quat[3])
                    tf_msg = TFMessage()
                    tf_msg.transforms = [t]
                    self.tf_pub.publish(tf_msg)

                # --- PUBLISH JOINT STATES ---
                # Usiamo get_measured_joint_efforts() invece di get_applied_joint_efforts()
                # perché il controller RL invia comandi di *posizione*. 
                tau = self._spot.robot.get_measured_joint_efforts()
                theta_dot = self._spot.robot.get_joint_velocities()
                j_pos = self._spot.robot.get_joint_positions()
                
                if tau is not None and theta_dot is not None and j_pos is not None:
                    msg = JointState()
                    msg.header.stamp = stamp
                    msg.header.frame_id = "base_link"
                    msg.name = getattr(self._spot.robot, 'dof_names', [f"joint_{i}" for i in range(len(j_pos))])
                    msg.position = j_pos.tolist()
                    msg.velocity = theta_dot.tolist()
                    msg.effort = tau.tolist()
                    self.js_pub.publish(msg)

    def run(self) -> None:
        while simulation_app.is_running():
            self._world.step(render=True)
            if hasattr(self, 'ros_node') and rclpy.ok():
                rclpy.spin_once(self.ros_node, timeout_sec=0.0)
            if self._world.is_stopped():
                self.needs_reset = True
                self._zed_graph_needs_rearm = True
            elif self._zed_graph_needs_rearm:
                # Camera helpers detach their ROS writers on Stop.
                # Re-evaluate the existing graph once after Play to attach them again.
                graph = og.get_graph_by_path(self._zed_ros_graph_path)
                if graph is None:
                    raise RuntimeError(f"ZED ROS graph not found: {self._zed_ros_graph_path}")
                og.Controller.evaluate_sync(graph)
                self._zed_graph_needs_rearm = False
                print("[ROS2] ZED camera publishers rearmed after Play.")
        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True


def main():
    physics_dt = 1 / 200.0
    render_dt = 1 / 60.0

    runner = SpotRunner(physics_dt=physics_dt, render_dt=render_dt)
    simulation_app.update()
    runner._world.reset()
    simulation_app.update()
    runner.setup()
    # World.reset() sends a STOP event, which detaches ROS camera writers.
    # Create the graph only after that reset so its publishers stay attached.
    runner._zed_ros_graph_path = create_zed_ros_graph(runner._zed_camera_path)
    simulation_app.update()
    runner.run()
    simulation_app.close()


if __name__ == "__main__":
    main()

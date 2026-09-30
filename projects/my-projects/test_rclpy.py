from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import sys
sys.path = [p for p in sys.path if '/opt/ros' not in p]
from omni.isaac.core.utils.extensions import enable_extension
enable_extension('isaacsim.ros2.bridge')
import rclpy
print("RCLPY SUCCESS")
app.close()

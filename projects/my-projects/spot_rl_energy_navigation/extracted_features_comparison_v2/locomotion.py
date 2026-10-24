"""Local copy of the frozen Spot joint controller; no navigation RL training."""
import torch
from common import local_path


class SpotController:
    def __init__(self, robot, device):
        self.robot = robot
        self.model = torch.jit.load(str(local_path("assets/robot/locomotion.pt")), map_location=device).eval()
        self.previous_action = torch.zeros((1, 12), device=device)
        self.action = torch.zeros_like(self.previous_action)
        self.counter = 0

    def reset(self):
        self.previous_action = torch.zeros_like(self.previous_action)
        self.action = torch.zeros_like(self.action)
        self.counter = 0

    def apply(self, command):
        if self.counter % 10 == 0:
            data = self.robot.data
            obs = torch.cat((data.root_lin_vel_b, data.root_ang_vel_b, data.projected_gravity_b,
                             command, data.joint_pos-data.default_joint_pos, data.joint_vel,
                             self.previous_action), dim=-1)
            if obs.shape != (1, 48):
                raise ValueError(f"Controller expects 48 inputs: {obs.shape}")
            with torch.inference_mode():
                self.action = self.model(obs.float()).reshape(1, 12)
            self.previous_action = self.action.clone()
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos + .2*self.action)
        self.counter += 1

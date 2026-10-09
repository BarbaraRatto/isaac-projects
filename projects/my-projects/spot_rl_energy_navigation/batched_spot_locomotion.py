"""Batched frozen IsaacRobotics Spot velocity controller."""

import torch

from rl_config import POLICY_WEIGHTS


class BatchedSpotVelocityController:
    def __init__(self, robot, num_envs: int, device: str):
        self.robot = robot
        self.num_envs = num_envs
        self.device = device
        self.model = torch.jit.load(str(POLICY_WEIGHTS), map_location=device).eval()
        self.previous_action = torch.zeros((num_envs, 12), device=device)
        self.action = torch.zeros_like(self.previous_action)
        self.counter = 0
        self.decimation = 10
        self.action_scale = 0.2

    def reset(self, env_ids) -> None:
        # The JIT output was created under torch.inference_mode(). Replace it
        # out of place: in-place writes to inference tensors are forbidden.
        keep = torch.ones((self.num_envs, 1), device=self.device)
        keep[env_ids] = 0.0
        self.previous_action = self.previous_action * keep
        self.action = self.action * keep

    def apply(self, command: torch.Tensor) -> None:
        if self.counter % self.decimation == 0:
            data = self.robot.data
            observation = torch.cat((
                data.root_lin_vel_b, data.root_ang_vel_b, data.projected_gravity_b,
                command, data.joint_pos - data.default_joint_pos,
                data.joint_vel, self.previous_action,
            ), dim=-1)
            if observation.shape != (self.num_envs, 48):
                raise RuntimeError(f"Unexpected controller input {observation.shape}")
            with torch.inference_mode():
                self.action = self.model(observation.float()).reshape(self.num_envs, 12)
            self.previous_action = self.action.clone()
        target = self.robot.data.default_joint_pos + self.action * self.action_scale
        self.robot.set_joint_position_target(target)
        self.counter += 1

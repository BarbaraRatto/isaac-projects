"""Isaac Lab wrapper for IsaacRobotics' pretrained Spot velocity controller."""

import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab_assets import SPOT_CFG

from rl_config import POLICY_WEIGHTS, ROBOT_USD, SPAWN_POSITION


class SpotVelocityController:
    """Use the original 48-value observation, JIT weights and position action."""

    def __init__(self, robot: Articulation, device: str):
        self.robot = robot
        self.device = device
        self.model = torch.jit.load(str(POLICY_WEIGHTS), map_location=device).eval()
        self.previous_action = torch.zeros((1, 12), device=device)
        self.action = torch.zeros((1, 12), device=device)
        self.counter = 0
        self.decimation = 10
        self.action_scale = 0.2

    def reset(self) -> None:
        """Forget the previous trial's action before the next placement."""
        # The model output is an inference tensor and cannot be changed in place.
        self.previous_action = torch.zeros((1, 12), device=self.device)
        self.action = torch.zeros((1, 12), device=self.device)
        self.counter = 0

    def forward(self, command: torch.Tensor) -> None:
        if self.counter % self.decimation == 0:
            data = self.robot.data
            observation = torch.cat(
                (
                    data.root_lin_vel_b,
                    data.root_ang_vel_b,
                    data.projected_gravity_b,
                    command,
                    data.joint_pos - data.default_joint_pos,
                    data.joint_vel,
                    self.previous_action,
                ),
                dim=-1,
            )
            if observation.shape != (1, 48):
                raise RuntimeError(f"Unexpected Spot policy observation shape: {observation.shape}")
            with torch.inference_mode():
                self.action = self.model(observation.float()).reshape(1, 12)
            self.previous_action = self.action.clone()
        target = self.robot.data.default_joint_pos + self.action * self.action_scale
        self.robot.set_joint_position_target(target)
        self.robot.write_data_to_sim()
        self.counter += 1


def create_spot() -> Articulation:
    """Spawn the project's Spot USD with PD drives matching its IsaacRobotics policy."""
    cfg = SPOT_CFG.copy()
    cfg.prim_path = "/World/Spot"
    cfg.spawn.usd_path = str(ROBOT_USD)
    cfg.init_state.pos = SPAWN_POSITION
    cfg.actuators = {
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[".*_h[xy]"], effort_limit_sim=45.0,
            stiffness=60.0, damping=1.5,
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=[".*_kn"], effort_limit_sim=115.0,
            stiffness=60.0, damping=1.5,
        ),
    }
    return Articulation(cfg)


def reset_spot(robot: Articulation, position: tuple[float, float, float] | None = None) -> None:
    """Restore the joint state and place Spot at a world-space position."""
    root_state = robot.data.default_root_state.clone()
    if position is not None:
        root_state[0, :3] = torch.tensor(position, device=root_state.device, dtype=root_state.dtype)
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()

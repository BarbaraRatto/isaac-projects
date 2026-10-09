"""Single-robot Gymnasium task simulated with Isaac Lab.

The learned policy sends (vx, vy, yaw rate). The frozen IsaacRobotics policy
continues to turn those commands into joint position targets. No terrain label,
energy cost, ROS 2, camera, or DINO feature is present in the observation.
"""

import re

import gymnasium as gym
import numpy as np
import omni.usd
import torch
from gymnasium import spaces
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils

from locomotion import SpotVelocityController, create_spot, reset_spot
from navigation_config import NAV_CFG, NavigationConfig
from scene import build_scene


VIEW_CAMERA_PATH = "/World/NavigationCamera"
GOAL_MARKER_PATH = "/World/NavigationGoal"


class SpotNavigationEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, simulation_app, device: str = "cuda:0",
                 render: bool = False, cfg: NavigationConfig = NAV_CFG,
                 camera_factory=None, scene_factory=build_scene):
        super().__init__()
        self.app = simulation_app
        self.cfg = cfg
        self.render_enabled = render
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=cfg.physics_dt, device=device)
        )
        scene_factory()
        self.cell_low, self.cell_high = self._find_asphalt_cell()
        self.goal_marker = None
        if render:
            stage = omni.usd.get_context().get_stage()
            UsdGeom.Camera.Define(stage, VIEW_CAMERA_PATH)
            self.goal_marker = UsdGeom.Sphere.Define(stage, GOAL_MARKER_PATH)
            self.goal_marker.CreateRadiusAttr(0.12)
            self.goal_marker.CreateDisplayColorAttr([Gf.Vec3f(0.0, 1.0, 0.0)])

        self.robot = create_spot()
        # Isaac Lab sensors are created before the first simulation reset.
        self.sensor_camera = camera_factory() if camera_factory is not None else None
        self.controller = SpotVelocityController(self.robot, self.sim.device)
        self.sim.reset()
        self.foot_ids = [i for i, name in enumerate(self.robot.body_names)
                         if name.endswith("_foot")]
        if len(self.foot_ids) != 4:
            raise RuntimeError(f"Expected four Spot feet, found {self.foot_ids}")

        if render:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = VIEW_CAMERA_PATH
            center_x = (self.cell_low[0] + self.cell_high[0]) / 2.0
            center_y = (self.cell_low[1] + self.cell_high[1]) / 2.0
            self.sim.set_camera_view(
                (center_x + 1.0, center_y + 5.0, 3.5),
                (center_x, center_y, 0.0),
                camera_prim_path=VIEW_CAMERA_PATH,
            )

        # Three bounded policy actions become a velocity command for Spot.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # Goal in body frame (2), base linear velocity (2), yaw rate (1),
        # projected gravity (3), and previous velocity command (3).
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)
        self.goal_w = torch.zeros(3, device=self.sim.device)
        self.previous_command = torch.zeros(3, device=self.sim.device)
        self.previous_power_w = 0.0
        self.previous_distance_m = 0.0
        self.episode_energy_j = 0.0
        self.elapsed_steps = 0

    def _find_asphalt_cell(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        stage = omni.usd.get_context().get_stage()
        cells = []
        for prim in stage.Traverse():
            match = re.fullmatch(r"Cell_(\d+)_(\d+)_(.+)", prim.GetName())
            if match and match.group(3) == self.cfg.terrain_name:
                cells.append((int(match.group(1)), int(match.group(2)), prim))
        if not cells:
            raise RuntimeError(f"No {self.cfg.terrain_name} cell in terrain USD")
        prim = min(cells, key=lambda cell: (cell[0], cell[1]))[2]
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        box = bounds.ComputeWorldBound(prim).ComputeAlignedBox()
        return tuple(map(float, box.GetMin())), tuple(map(float, box.GetMax()))

    def _physics_step(self, command: torch.Tensor, render_sensor: bool = False) -> None:
        if not self.app.is_running():
            raise RuntimeError("Isaac Sim was closed during the RL episode")
        self.controller.forward(command.view(1, 3))
        self.sim.step(render=self.render_enabled or render_sensor)
        self.robot.update(self.cfg.physics_dt)

    def _measured_power(self) -> float:
        effort = self.robot.root_physx_view.get_dof_projected_joint_forces()
        speed = self.robot.data.joint_vel
        if effort.shape != speed.shape or speed.shape != (1, self.robot.num_joints):
            raise RuntimeError("Measured joint efforts and velocities do not match")
        power = torch.sum(torch.abs(effort * speed))
        if not bool(torch.isfinite(power)):
            raise RuntimeError("Non-finite measured mechanical power")
        return float(power.item())

    def _goal_distance(self) -> float:
        delta = self.goal_w[:2] - self.robot.data.root_pos_w[0, :2]
        return float(torch.linalg.vector_norm(delta).item())

    def _observation(self) -> np.ndarray:
        data = self.robot.data
        delta_w = self.goal_w - data.root_pos_w[0]
        delta_b = quat_apply_inverse(yaw_quat(data.root_quat_w), delta_w.view(1, 3))[0]
        obs = torch.cat((
            delta_b[:2] / 2.0,
            data.root_lin_vel_b[0, :2],
            data.root_ang_vel_b[0, 2:3],
            data.projected_gravity_b[0],
            self.previous_command / torch.tensor(
                (self.cfg.max_forward_m_s, self.cfg.max_lateral_m_s,
                 self.cfg.max_yaw_rad_s), device=self.sim.device
            ),
        ))
        result = obs.detach().cpu().numpy().astype(np.float32)
        if result.shape != self.observation_space.shape or not np.isfinite(result).all():
            raise RuntimeError("Invalid navigation observation")
        return result

    def _is_unsafe(self) -> bool:
        data = self.robot.data
        x, y, z = map(float, data.root_pos_w[0, :3])
        low, high = self.cell_low, self.cell_high
        if z < high[2] + self.cfg.minimum_body_height_m:
            return True
        if float(data.projected_gravity_b[0, 2]) > -0.5:
            return True
        margin = self.cfg.cell_margin_m
        if not (low[0] + margin <= x <= high[0] - margin
                and low[1] + margin <= y <= high[1] - margin):
            return True
        feet = data.body_pos_w[0, self.foot_ids, :2]
        foot_margin = 0.05
        return bool(((feet[:, 0] < low[0] + foot_margin)
                     | (feet[:, 0] > high[0] - foot_margin)
                     | (feet[:, 1] < low[1] + foot_margin)
                     | (feet[:, 1] > high[1] - foot_margin)).any())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        low, high = self.cell_low, self.cell_high
        center_y = (low[1] + high[1]) / 2.0
        start_y = center_y + float(self.np_random.uniform(-0.15, 0.15))
        goal_y = center_y + float(self.np_random.uniform(-0.5, 0.5))
        start = (low[0] + self.cfg.start_inset_m, start_y,
                 high[2] + self.cfg.spawn_height_m)
        self.goal_w[:] = torch.tensor(
            (high[0] - self.cfg.goal_inset_m, goal_y, high[2]),
            device=self.sim.device,
        )
        if self.goal_marker is not None:
            UsdGeom.XformCommonAPI(self.goal_marker.GetPrim()).SetTranslate(
                Gf.Vec3d(float(self.goal_w[0]), goal_y, high[2] + 0.15)
            )
        reset_spot(self.robot, start)
        self.controller.reset()
        self.robot.update(self.cfg.physics_dt)
        zero = torch.zeros(3, device=self.sim.device)
        for _ in range(round(self.cfg.settle_seconds / self.cfg.physics_dt)):
            self._physics_step(zero)
        self.previous_command.zero_()
        self.previous_power_w = self._measured_power()
        self.previous_distance_m = self._goal_distance()
        self.episode_energy_j = 0.0
        self.elapsed_steps = 0
        if self._is_unsafe():
            raise RuntimeError("Spot fell or left the asphalt cell during reset")
        return self._observation(), {"goal_distance_m": self.previous_distance_m}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"Invalid velocity action: {action}")
        a = np.clip(action, -1.0, 1.0)
        command = torch.tensor((
            self.cfg.max_forward_m_s * float(a[0]),
            self.cfg.max_lateral_m_s * float(a[1]),
            self.cfg.max_yaw_rad_s * float(a[2]),
        ), device=self.sim.device)
        energy_j = 0.0
        unsafe = False
        for physics_step in range(self.cfg.physics_steps_per_action):
            self._physics_step(
                command,
                render_sensor=(self.sensor_camera is not None
                               and physics_step == self.cfg.physics_steps_per_action - 1),
            )
            power_w = self._measured_power()
            energy_j += 0.5 * (self.previous_power_w + power_w) * self.cfg.physics_dt
            self.previous_power_w = power_w
            if self._is_unsafe():
                unsafe = True
                break

        self.previous_command = command
        self.elapsed_steps += 1
        self.episode_energy_j += energy_j
        distance_m = self._goal_distance()
        progress_m = self.previous_distance_m - distance_m
        self.previous_distance_m = distance_m
        success = distance_m <= self.cfg.goal_radius_m and not unsafe
        terminated = bool(success or unsafe)
        truncated = bool(self.elapsed_steps >= self.cfg.max_episode_steps and not terminated)
        reward = (self.cfg.progress_reward_per_m * progress_m
                  - self.cfg.energy_penalty_per_j * energy_j
                  - self.cfg.time_penalty_per_step)
        if success:
            reward += self.cfg.success_bonus
        if unsafe:
            reward -= self.cfg.failure_penalty
        info = {
            "is_success": success,
            "goal_distance_m": distance_m,
            "final_distance_m": distance_m,
            "step_energy_j": energy_j,
            "episode_energy_j": self.episode_energy_j,
            "termination_reason": "goal" if success else "unsafe" if unsafe else
                                  "timeout" if truncated else "running",
        }
        if self.sensor_camera is not None:
            self.sensor_camera.update(
                self.cfg.physics_steps_per_action * self.cfg.physics_dt,
                force_recompute=True,
            )
        return self._observation(), float(reward), terminated, truncated, info

    def close(self):
        self.sim.clear_instance()

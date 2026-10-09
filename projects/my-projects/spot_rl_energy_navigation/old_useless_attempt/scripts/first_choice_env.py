"""First two-choice PPO task: visual features, goal, mechanical work, cmd_vel."""

import math
import sys
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from first_choice_scene import build_first_choice_scene
from locomotion import reset_spot
from navigation_config import NAV_CFG
from navigation_env import SpotNavigationEnv
from rl_config import PHYSICS_DT
from visual_features import (FeatureField, LOCAL_SIZE, VisualObservation,
                             camera_ground_points, camera_patches, visible_mask)
from zed_sensor import create_body_zed, sync_zed_pose


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "spot_terrain_gridmap"))
FIRST_CHOICE_CFG = replace(
    NAV_CFG,
    max_episode_seconds=40.0,
    goal_radius_m=0.55,
    success_bonus=20.0,
    failure_penalty=10.0,
    time_penalty_per_step=0.03,
)


class FirstChoiceEnv(SpotNavigationEnv):
    def __init__(self, simulation_app, atlas_file: Path, *, online: bool = False,
                 cached_visible: bool = False, augment: bool = False,
                 device: str = "cuda:0", render: bool = False):
        self.online = online
        self.cached_visible = cached_visible
        self.augment = augment
        super().__init__(
            simulation_app, device=device, render=render, cfg=FIRST_CHOICE_CFG,
            camera_factory=(lambda: create_body_zed(depth=True))
            if online or cached_visible else None,
            scene_factory=build_first_choice_scene,
        )
        self.visual = VisualObservation(atlas_file)
        self.dino = None
        if online:
            from spot_terrain_gridmap.dino_feature_extractor import DinoFeatureExtractor
            self.dino = DinoFeatureExtractor(
                model_name="facebook/dinov2-small", device="cuda", input_size=518
            )
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(11 + LOCAL_SIZE,), dtype=np.float32)
        self.max_y = 0.0
        self.path_length_m = 0.0
        self.previous_xy = None
        self.visual_valid = 0

    def _physics_step(self, command: torch.Tensor, render_sensor: bool = False) -> None:
        if render_sensor:
            sync_zed_pose(self.sensor_camera, self.robot)
        super()._physics_step(command, render_sensor=render_sensor)

    def _is_unsafe(self) -> bool:
        data = self.robot.data
        x, y, z = (float(v) for v in data.root_pos_w[0, :3])
        if z < self.cfg.minimum_body_height_m:
            return True
        if float(data.projected_gravity_b[0, 2]) > -0.5:
            return True
        if x < -0.35 or x > 18.35 or y < -1.8 or y > 4.8:
            return True
        if x < 5.7 and y > 1.65:
            return True
        return False

    def _observation(self) -> np.ndarray:
        data = self.robot.data
        delta_w = self.goal_w - data.root_pos_w[0]
        delta_b = quat_apply_inverse(yaw_quat(data.root_quat_w), delta_w.view(1, 3))[0]
        normalizers = torch.tensor(
            (self.cfg.max_forward_m_s, self.cfg.max_lateral_m_s, self.cfg.max_yaw_rad_s),
            device=self.sim.device,
        )
        state = torch.cat((
            delta_b[:2] / 10.0,
            data.root_lin_vel_b[0, :2],
            data.root_ang_vel_b[0, 2:3],
            data.projected_gravity_b[0],
            self.previous_command / normalizers,
        )).detach().cpu().numpy().astype(np.float32)
        xy = data.root_pos_w[0, :2].detach().cpu().numpy()
        quat = data.root_quat_w[0]
        yaw = math.atan2(
            2.0 * float(quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * float(quat[2] ** 2 + quat[3] ** 2),
        )
        if self.online:
            points, features = camera_patches(self.sensor_camera, self.dino)
            field = FeatureField()
            field.insert(points, features)
            raw, mask = field.finish()
            local = self.visual.local(xy, yaw, self.visual.compress(raw), mask)
        elif self.cached_visible:
            points, _ = camera_ground_points(self.sensor_camera)
            visible = visible_mask(points) & self.visual.valid
            local = self.visual.local(xy, yaw, self.visual.atlas_small, visible)
            if self.augment:
                cells = local.reshape(-1, 13)
                keep = self.np_random.random(len(cells)) > 0.10
                cells[~keep] = 0.0
                noise = self.np_random.normal(0.0, 0.55, (len(cells), 12))
                cells[:, :12] += (noise * cells[:, -1, None]).astype(np.float32)
                local = cells.reshape(-1)
        else:
            local = self.visual.local(xy, yaw)
        self.visual_valid = int(local.reshape(-1, 13)[:, -1].sum())
        obs = np.concatenate((state, local)).astype(np.float32)
        if obs.shape != self.observation_space.shape or not np.isfinite(obs).all():
            raise RuntimeError(f"Invalid first-choice observation: {obs.shape}")
        return obs

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        start_y = float(self.np_random.uniform(-0.12, 0.12))
        start = (1.5, start_y, 0.75)
        self.goal_w[:] = torch.tensor((16.5, 0.0, 0.0), device=self.sim.device)
        reset_spot(self.robot, start)
        self.controller.reset()
        self.robot.update(PHYSICS_DT)
        if self.sensor_camera is not None:
            self.sensor_camera.reset()
        zero = torch.zeros(3, device=self.sim.device)
        settle_steps = round(self.cfg.settle_seconds / PHYSICS_DT)
        for i in range(settle_steps):
            self._physics_step(zero, render_sensor=self.sensor_camera is not None and i == settle_steps - 1)
        if self.sensor_camera is not None:
            self.sensor_camera.update(self.cfg.settle_seconds, force_recompute=True)
        self.previous_command.zero_()
        self.previous_power_w = self._measured_power()
        self.previous_distance_m = self._goal_distance()
        self.episode_energy_j = 0.0
        self.elapsed_steps = 0
        self.max_y = float(self.robot.data.root_pos_w[0, 1])
        self.previous_xy = self.robot.data.root_pos_w[0, :2].clone()
        self.path_length_m = 0.0
        if self._is_unsafe():
            raise RuntimeError("Spot fell or left the course during reset")
        obs = self._observation()
        return obs, {"goal_distance_m": self.previous_distance_m,
                     "visual_valid": self.visual_valid}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        xy = self.robot.data.root_pos_w[0, :2]
        self.path_length_m += float(torch.linalg.vector_norm(xy - self.previous_xy).item())
        self.previous_xy = xy.clone()
        self.max_y = max(self.max_y, float(xy[1]))
        info["route"] = "bypass" if self.max_y > 1.55 else "direct"
        info["path_length_m"] = self.path_length_m
        info["visual_valid"] = self.visual_valid
        return obs, reward, terminated, truncated, info

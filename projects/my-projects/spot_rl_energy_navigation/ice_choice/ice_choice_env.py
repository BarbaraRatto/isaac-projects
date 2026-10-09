"""Same original-USD navigation task with cached or live DINO observations."""

import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from ice_choice.ice_choice_config import ICE_GEOMETRY, ICE_NAV_CFG
from ice_visual_features import (FEATURE_DIM, FeatureField, VISUAL_SIZE, VisualObservation,
                                 camera_ground_points, mask_from_points)
from locomotion import reset_spot
from navigation_env import SpotNavigationEnv, VIEW_CAMERA_PATH
from rl_config import PHYSICS_DT
from scene import build_scene
from zed_sensor import create_body_zed, sync_zed_pose


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "spot_terrain_gridmap"))


class IceChoiceEnv(SpotNavigationEnv):
    def __init__(self, simulation_app, atlas_file: Path, mode: str, *,
                 device: str = "cuda:0", render: bool = False):
        if mode not in ("cached", "online"):
            raise ValueError("mode must be cached or online")
        self.mode = mode
        self.ever_ice_contact = False
        self.path_length_m = 0.0
        self.previous_xy = None
        self.last_visual_ms = 0.0
        self.last_dino_ms = 0.0
        self.dino_load_seconds = 0.0
        # Both modes create and render the SAME RGB-D camera every action.
        super().__init__(
            simulation_app, device=device, render=render, cfg=ICE_NAV_CFG,
            camera_factory=lambda: create_body_zed(depth=True),
            scene_factory=build_scene,
        )
        if render:
            self.sim.set_camera_view(
                (27.0, 15.0, 12.0), (21.0, 6.0, 0.0),
                camera_prim_path=VIEW_CAMERA_PATH,
            )
        self.visual = VisualObservation(atlas_file)
        self.dino = None
        if mode == "online":
            from spot_terrain_gridmap.dino_feature_extractor import DinoFeatureExtractor
            started = time.perf_counter()
            self.dino = DinoFeatureExtractor(
                model_name="facebook/dinov2-small", device="cuda", input_size=518
            )
            self.dino_load_seconds = time.perf_counter() - started
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(11 + VISUAL_SIZE,), dtype=np.float32
        )

    def _physics_step(self, command: torch.Tensor, render_sensor: bool = False) -> None:
        if render_sensor:
            sync_zed_pose(self.sensor_camera, self.robot)
        super()._physics_step(command, render_sensor=render_sensor)
        feet = self.robot.data.body_pos_w[0, self.foot_ids, :2]
        ice = ICE_GEOMETRY
        inside_x = (feet[:, 0] >= ice.ice_x_bounds[0]) & (feet[:, 0] <= ice.ice_x_bounds[1])
        inside_y = (feet[:, 1] >= ice.ice_y_bounds[0]) & (feet[:, 1] <= ice.ice_y_bounds[1])
        self.ever_ice_contact |= bool((inside_x & inside_y).any())

    def _is_unsafe(self) -> bool:
        data = self.robot.data
        x, y, z = map(float, data.root_pos_w[0, :3])
        bounds = ICE_GEOMETRY
        return (z < self.cfg.minimum_body_height_m
                or float(data.projected_gravity_b[0, 2]) > -0.5
                or not bounds.world_x_bounds[0] <= x <= bounds.world_x_bounds[1]
                or not bounds.world_y_bounds[0] <= y <= bounds.world_y_bounds[1])

    def _observation(self) -> np.ndarray:
        data = self.robot.data
        delta_w = self.goal_w - data.root_pos_w[0]
        delta_b = quat_apply_inverse(yaw_quat(data.root_quat_w), delta_w.view(1, 3))[0]
        maxima = torch.tensor(
            (self.cfg.max_forward_m_s, self.cfg.max_lateral_m_s, self.cfg.max_yaw_rad_s),
            device=self.sim.device,
        )
        state = torch.cat((
            delta_b[:2] / 10.0,
            data.root_lin_vel_b[0, :2], data.root_ang_vel_b[0, 2:3],
            data.projected_gravity_b[0], self.previous_command / maxima,
        )).detach().cpu().numpy().astype(np.float32)
        xy = data.root_pos_w[0, :2].detach().cpu().numpy()
        q = data.root_quat_w[0]
        yaw = math.atan2(2.0 * float(q[0] * q[3] + q[1] * q[2]),
                         1.0 - 2.0 * float(q[2] ** 2 + q[3] ** 2))

        started = time.perf_counter()
        self.last_dino_ms = 0.0
        if self.mode == "online":
            rgb = self.sensor_camera.data.output["rgb"][0].cpu().numpy().copy()
            dino_started = time.perf_counter()
            features = self.dino.extract(rgb).reshape(-1, FEATURE_DIM)
            self.last_dino_ms = (time.perf_counter() - dino_started) * 1000.0
            points, accepted = camera_ground_points(self.sensor_camera)
            field = FeatureField()
            field.insert(points, features[accepted])
            raw, valid = field.finish()
            local = self.visual.local(xy, yaw, self.visual.compress(raw), valid)
        else:
            points, _ = camera_ground_points(self.sensor_camera)
            visible = mask_from_points(points) & self.visual.valid
            local = self.visual.local(xy, yaw, self.visual.atlas_small, visible)
        self.last_visual_ms = (time.perf_counter() - started) * 1000.0
        obs = np.concatenate((state, local)).astype(np.float32)
        if obs.shape != self.observation_space.shape or not np.isfinite(obs).all():
            raise RuntimeError(f"Invalid ice-choice observation: {obs.shape}")
        self.visual_valid = int(local.reshape(-1, 33)[:, -1].sum())
        return obs

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        jitter = self.np_random.uniform(-ICE_GEOMETRY.start_jitter_m,
                                        ICE_GEOMETRY.start_jitter_m, size=2)
        x, y = np.asarray(ICE_GEOMETRY.start_xy) + jitter
        self.goal_w[:] = torch.tensor((*ICE_GEOMETRY.goal_xy, 0.0), device=self.sim.device)
        if self.goal_marker is not None:
            from pxr import Gf, UsdGeom
            UsdGeom.XformCommonAPI(self.goal_marker.GetPrim()).SetTranslate(
                Gf.Vec3d(*ICE_GEOMETRY.goal_xy, 0.15)
            )
        reset_spot(self.robot, (float(x), float(y), self.cfg.spawn_height_m))
        state = self.robot.data.default_root_state.clone()
        state[0, :3] = torch.tensor((x, y, self.cfg.spawn_height_m), device=self.sim.device)
        half = ICE_GEOMETRY.start_yaw_rad / 2.0
        state[0, 3:7] = torch.tensor((math.cos(half), 0.0, 0.0, math.sin(half)),
                                    device=self.sim.device)
        self.robot.write_root_state_to_sim(state)
        self.controller.reset()
        self.robot.update(PHYSICS_DT)
        self.sensor_camera.reset()
        self.ever_ice_contact = False
        zero = torch.zeros(3, device=self.sim.device)
        settle_steps = round(self.cfg.settle_seconds / PHYSICS_DT)
        for i in range(settle_steps):
            self._physics_step(zero, render_sensor=i == settle_steps - 1)
        self.sensor_camera.update(self.cfg.settle_seconds, force_recompute=True)
        self.ever_ice_contact = False  # Landing is excluded from the episode.
        self.previous_command.zero_()
        self.previous_power_w = self._measured_power()
        self.previous_distance_m = self._goal_distance()
        self.episode_energy_j = 0.0
        self.elapsed_steps = 0
        self.path_length_m = 0.0
        self.previous_xy = self.robot.data.root_pos_w[0, :2].clone()
        if self._is_unsafe():
            raise RuntimeError("Spot fell or left the terrain while settling at START")
        obs = self._observation()
        self.visual_count_sum = self.visual_valid
        self.visual_count_steps = 1
        return obs, {"goal_distance_m": self.previous_distance_m,
                     "visual_valid": self.visual_valid}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        xy = self.robot.data.root_pos_w[0, :2]
        self.path_length_m += float(torch.linalg.vector_norm(xy - self.previous_xy).item())
        self.previous_xy = xy.clone()
        self.visual_count_sum += self.visual_valid
        self.visual_count_steps += 1
        info.update(
            mean_visual_valid=self.visual_count_sum / self.visual_count_steps,
            route="ice_contact" if self.ever_ice_contact else "avoided_ice",
            ice_contact=self.ever_ice_contact,
            path_length_m=self.path_length_m,
            visual_valid=self.visual_valid,
            visual_ms=self.last_visual_ms,
            dino_ms=self.last_dino_ms,
        )
        return obs, reward, terminated, truncated, info

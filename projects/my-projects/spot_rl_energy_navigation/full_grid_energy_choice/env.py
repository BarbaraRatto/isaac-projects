"""Energy-aware goal navigation over the complete original terrain USD.

One shared PPO controls body velocity. The frozen IsaacRobotics locomotion
policy controls the joints. DINO features are computed live from each ZED X.
"""

import math
from pathlib import Path

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, yaw_quat

from full_grid_energy_choice.config import GEOMETRY, NAV_CFG_FULL, FullGridGeometry
from navigation_config import NavigationConfig
from full_grid_energy_choice.visual_features import VISUAL_SIZE
from batched_spot_locomotion import BatchedSpotVelocityController
from full_grid_energy_choice.scene import FullGridChoiceCfg, create_source_terrain
from full_grid_energy_choice.vision import BatchedDino


class FullGridChoiceEnv(DirectRLEnv):
    cfg: FullGridChoiceCfg

    def __init__(self, cfg: FullGridChoiceCfg, atlas_file: Path | None, dino_batch_size: int = 4,
                 geometry: FullGridGeometry | None = None,
                 nav_cfg: NavigationConfig | None = None,
                 vision_type: type[BatchedDino] = BatchedDino, **kwargs):
        self.measure_only = atlas_file is None
        super().__init__(cfg, **kwargs)
        self.nav = nav_cfg if nav_cfg is not None else NAV_CFG_FULL
        self.geometry = geometry if geometry is not None else GEOMETRY
        self.controller = BatchedSpotVelocityController(self.robot, self.num_envs, self.device)
        self.vision = (None if self.measure_only else
                       vision_type(atlas_file, self.device, batch_size=dino_batch_size))
        self.foot_ids = [i for i, name in enumerate(self.robot.body_names) if name.endswith("_foot")]
        if len(self.foot_ids) != 4:
            raise RuntimeError(f"Expected four Spot feet, got {self.foot_ids}")
        self.goal_w = self.scene.env_origins + torch.tensor(
            (*self.geometry.goal_xy, 0.0), device=self.device
        )
        self.previous_command = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_distance = torch.zeros(self.num_envs, device=self.device)
        self.previous_power = torch.zeros(self.num_envs, device=self.device)
        self.step_energy = torch.zeros(self.num_envs, device=self.device)
        self.episode_energy = torch.zeros(self.num_envs, device=self.device)
        self.path_length = torch.zeros(self.num_envs, device=self.device)
        self.previous_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.rocks_entry = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.asphalt_entry = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.obstacle_entry = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.visual_valid = torch.zeros(self.num_envs, device=self.device)
        self.last_visual_ms = 0.0
        self.last_dino_ms = 0.0
        self._settle_once()

    def _setup_scene(self) -> None:
        create_source_terrain()
        self.robot = Articulation(self.cfg.robot_cfg)
        self.camera = None if self.measure_only else TiledCamera(self.cfg.camera_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["spot"] = self.robot
        if self.camera is not None:
            self.scene.sensors["zed_x"] = self.camera

    def _settle_once(self) -> None:
        zero = torch.zeros((self.num_envs, 3), device=self.device)
        for _ in range(round(self.nav.settle_seconds / self.physics_dt)):
            self.controller.apply(zero)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        self.settled_root = self.robot.data.root_state_w.clone()
        local_height = self.settled_root[:, 2] - self.scene.env_origins[:, 2]
        upright = self.robot.data.projected_gravity_b[:, 2] < -0.5
        if bool(((local_height < self.nav.minimum_body_height_m) | ~upright).any()):
            raise RuntimeError(
                f"Spot fell during settling at the new ramp start: "
                f"heights={local_height.tolist()} upright={upright.tolist()}"
            )
        self.settled_joint_pos = self.robot.data.joint_pos.clone()
        self.settled_joint_vel = self.robot.data.joint_vel.clone()
        # Keep the configured start/yaw while retaining the settled height and leg pose.
        self.settled_root[:, :2] = (
            self.scene.env_origins[:, :2] + torch.tensor(self.geometry.start_xy, device=self.device)
        )
        half = self.geometry.start_yaw_rad / 2.0
        self.settled_root[:, 3:7] = torch.tensor(
            (math.cos(half), 0.0, 0.0, math.sin(half)), device=self.device
        )
        self.settled_root[:, 7:] = 0.0
        self.controller.counter = 0
        self.controller.reset(torch.arange(self.num_envs, device=self.device))
        if self.measure_only:
            print(f"[GUIDED] settled_height_m={self.settled_root[:, 2].tolist()} "
                  "camera=off DINO=off", flush=True)
        else:
            print(f"[PARALLEL] settled_height_m={self.settled_root[:, 2].tolist()} "
                  f"camera={self.num_envs}x640x360 DINO=one frozen model "
                  f"microbatch={self.vision.batch_size}", flush=True)

    def _measured_power(self) -> torch.Tensor:
        effort = self.robot.root_physx_view.get_dof_projected_joint_forces()
        speed = self.robot.data.joint_vel
        if effort.shape != speed.shape or effort.shape[0] != self.num_envs:
            raise RuntimeError(f"Measured joint effort shape {effort.shape} != speed {speed.shape}")
        power = (effort * speed).abs().sum(dim=1)
        if not bool(torch.isfinite(power).all()):
            raise RuntimeError("Non-finite measured mechanical power")
        return power

    def _distance(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.goal_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)

    def _foot_over(self, x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> torch.Tensor:
        feet = self.robot.data.body_pos_w[:, self.foot_ids, :2] - self.scene.env_origins[:, None, :2]
        inside = ((feet[..., 0] >= x_bounds[0]) & (feet[..., 0] <= x_bounds[1])
                  & (feet[..., 1] >= y_bounds[0]) & (feet[..., 1] <= y_bounds[1]))
        return inside.any(dim=1)

    def _track_terrain(self) -> None:
        self.rocks_entry |= self._foot_over(self.geometry.rocks_x_bounds, self.geometry.rocks_y_bounds)
        self.asphalt_entry |= self._foot_over(self.geometry.asphalt_x_bounds, self.geometry.asphalt_y_bounds)
        self.obstacle_entry |= self._foot_over(self.geometry.obstacle_x_bounds, self.geometry.obstacle_y_bounds)

    def _sync_cameras(self) -> None:
        if self.camera is None:
            return
        root = self.robot.data.root_pos_w
        rot = self.robot.data.root_quat_w
        offset = torch.tensor((0.42, 0.0, 0.07), device=self.device).expand(self.num_envs, -1)
        half = math.radians(7.5)  # quaternion half-angle: actual downward pitch is 15 degrees
        pitch = torch.tensor((math.cos(half), 0.0, math.sin(half), 0.0), device=self.device)
        self.camera.set_world_poses(
            positions=root + quat_apply(rot, offset),
            orientations=quat_mul(rot, pitch.expand(self.num_envs, -1)),
            convention="world",
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 3):
            raise ValueError(f"Expected {(self.num_envs, 3)} actions, got {actions.shape}")
        action = actions.clamp(-1.0, 1.0)
        maxima = torch.tensor((self.nav.max_forward_m_s, self.nav.max_lateral_m_s,
                               self.nav.max_yaw_rad_s), device=self.device)
        self.command = action * maxima
        self.step_energy.zero_()
        self.previous_power = self._measured_power()

    def _apply_action(self) -> None:
        # The measured power here belongs to the preceding physics step.
        if self.controller.counter % self.cfg.decimation != 0:
            power = self._measured_power()
            self.step_energy += 0.5 * (self.previous_power + power) * self.physics_dt
            self.previous_power = power
        self._track_terrain()
        self.controller.apply(self.command)
        # The renderer runs once per RL action, after the last physics substep.
        if self.controller.counter % self.cfg.decimation == 0:
            self._sync_cameras()

    def _get_dones(self):
        self._track_terrain()
        pos = self.robot.data.root_pos_w - self.scene.env_origins
        gravity = self.robot.data.projected_gravity_b[:, 2]
        unsafe = ((pos[:, 2] < self.nav.minimum_body_height_m) | (gravity > -0.5)
                  | (pos[:, 0] < self.geometry.world_x_bounds[0])
                  | (pos[:, 0] > self.geometry.world_x_bounds[1])
                  | (pos[:, 1] < self.geometry.world_y_bounds[0])
                  | (pos[:, 1] > self.geometry.world_y_bounds[1]))
        self.success = (self._distance() <= self.nav.goal_radius_m) & ~unsafe
        self.unsafe = unsafe
        timeout = self.episode_length_buf >= self.max_episode_length
        return self.success | unsafe, timeout & ~(self.success | unsafe)

    def _get_rewards(self) -> torch.Tensor:
        power = self._measured_power()
        self.step_energy += 0.5 * (self.previous_power + power) * self.physics_dt
        self.episode_energy += self.step_energy
        distance = self._distance()
        progress = self.previous_distance - distance
        self.previous_distance = distance
        xy = self.robot.data.root_pos_w[:, :2]
        self.path_length += torch.linalg.vector_norm(xy - self.previous_xy, dim=1)
        self.previous_xy = xy.clone()
        reward = (self.nav.progress_reward_per_m * progress
                  - self.nav.energy_penalty_per_j * self.step_energy
                  - self.nav.time_penalty_per_step
                  + self.nav.success_bonus * self.success.float()
                  - self.nav.failure_penalty * self.unsafe.float())
        self.previous_command = self.command.clone()
        self.extras = {
            # SB3 computes numpy.mean over this field; give it CPU booleans.
            "is_success": self.success.detach().cpu().tolist(),
            "episode_energy_j": self.episode_energy.clone(),
            "path_length_m": self.path_length.clone(),
            "rocks_entry": self.rocks_entry.clone(),
            "obstacle_entry": self.obstacle_entry.clone(),
            "asphalt_entry": self.asphalt_entry.clone(),
            "final_distance_m": distance.clone(),
            "visual_ms": torch.full((self.num_envs,), self.last_visual_ms / self.num_envs,
                                    device=self.device),
            "dino_ms": torch.full((self.num_envs,), self.last_dino_ms / self.num_envs,
                                  device=self.device),
        }
        return reward

    def _get_observations(self) -> dict:
        data = self.robot.data
        delta = self.goal_w - data.root_pos_w
        delta_body = quat_apply_inverse(yaw_quat(data.root_quat_w), delta)
        maxima = torch.tensor((self.nav.max_forward_m_s, self.nav.max_lateral_m_s,
                               self.nav.max_yaw_rad_s), device=self.device)
        state = torch.cat((
            delta_body[:, :2] / 10.0, data.root_lin_vel_b[:, :2],
            data.root_ang_vel_b[:, 2:3], data.projected_gravity_b,
            self.previous_command / maxima,
        ), dim=1)
        q = data.root_quat_w
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1 - 2 * (q[:, 2].square() + q[:, 3].square()))
        if self.vision is None:
            # Guided energy comparison uses the same physics and reward, but
            # no camera, DINO, PCA atlas or visual input is needed.
            visual = torch.zeros((self.num_envs, VISUAL_SIZE), device=self.device)
            self.visual_valid.zero_()
            self.last_visual_ms = self.last_dino_ms = 0.0
        else:
            visual, self.visual_valid = self.vision.extract(
                self.camera, self.scene.env_origins, data.root_pos_w[:, :2], yaw
            )
            self.last_visual_ms = self.vision.last_visual_ms
            self.last_dino_ms = self.vision.last_dino_ms
        obs = torch.cat((state, visual), dim=1).float()
        if obs.shape != (self.num_envs, 11 + VISUAL_SIZE) or not bool(torch.isfinite(obs).all()):
            raise RuntimeError(f"Invalid parallel observation {obs.shape}")
        return {"policy": obs}

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        count = len(env_ids)
        # Independent small position jitter, same local start/goal and yaw.
        state = self.settled_root[env_ids].clone()
        jitter = (2 * torch.rand((count, 2), device=self.device) - 1) * self.geometry.start_jitter_m
        state[:, :2] += jitter
        self.robot.write_root_state_to_sim(state, env_ids)
        self.robot.write_joint_state_to_sim(
            self.settled_joint_pos[env_ids], self.settled_joint_vel[env_ids], env_ids=env_ids
        )
        self.robot.reset(env_ids)
        self.robot.update(self.physics_dt)
        self.controller.reset(env_ids)
        self.previous_command[env_ids] = 0.0
        self.previous_distance[env_ids] = torch.linalg.vector_norm(
            self.goal_w[env_ids, :2] - state[:, :2], dim=1
        )
        self.previous_power[env_ids] = 0.0
        self.step_energy[env_ids] = 0.0
        self.episode_energy[env_ids] = 0.0
        self.path_length[env_ids] = 0.0
        self.previous_xy[env_ids] = state[:, :2]
        self.rocks_entry[env_ids] = False
        self.obstacle_entry[env_ids] = False
        self.asphalt_entry[env_ids] = False
        self._sync_cameras()

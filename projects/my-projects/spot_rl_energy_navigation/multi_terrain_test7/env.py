"""Test7 environment: local copy of test4 physics/reward/reset, raw DINO, two actions."""

import math

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, yaw_quat

from multi_terrain_test7.config import NAV_CFG, OBSERVATION_SIZE, TERRAIN_LABELS
from multi_terrain_test7.layouts import Condition
from multi_terrain_test7.scene import Test7Cfg, setup_scene
from multi_terrain_test7.vision import RawBatchedDino
from multi_terrain_test7.actions import navigation_commands
from multi_terrain_test7.locomotion import BatchedSpotVelocityController
from multi_terrain_test7.terrain import read_terrain_codes


class Test7Env(DirectRLEnv):
    cfg: Test7Cfg

    def __init__(self, cfg: Test7Cfg, conditions: tuple[Condition, ...],
                 dino_batch_size: int = 4, **kwargs):
        if len(conditions) != cfg.scene.num_envs:
            raise ValueError("Every environment needs exactly one condition")
        if tuple(item.env_id for item in conditions) != tuple(range(len(conditions))):
            raise ValueError("Conditions must be ordered by env_id")
        self.conditions = conditions
        super().__init__(cfg, **kwargs)
        self.nav = NAV_CFG
        self.geometry = conditions[0].case.geometry
        self.controller = BatchedSpotVelocityController(self.robot, self.num_envs, self.device)
        self.vision = RawBatchedDino(self.device, batch_size=dino_batch_size)
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
        self.visual_valid = torch.zeros(self.num_envs, device=self.device)
        self.last_visual_ms = self.last_dino_ms = 0.0
        self.terrain_codes = torch.tensor(
            [read_terrain_codes(item.case) for item in conditions],
            dtype=torch.long, device=self.device,
        )
        self.terrain_entries = torch.zeros((self.num_envs, len(TERRAIN_LABELS)),
                                           dtype=torch.bool, device=self.device)
        self._settle_once()
        starts = torch.tensor([item.start_xy for item in conditions], device=self.device)
        goals = torch.tensor([item.goal_xy for item in conditions], device=self.device)
        self.settled_root[:, :2] = self.scene.env_origins[:, :2] + starts
        yaw = torch.atan2(goals[:, 1] - starts[:, 1], goals[:, 0] - starts[:, 0])
        self.settled_root[:, 3] = torch.cos(yaw / 2)
        self.settled_root[:, 4:6] = 0.0
        self.settled_root[:, 6] = torch.sin(yaw / 2)
        self.goal_w[:, :2] = self.scene.env_origins[:, :2] + goals
        self.goal_w[:, 2] = self.scene.env_origins[:, 2]

    def _setup_scene(self) -> None:
        setup_scene(self)

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
                f"Spot fell during initial settling: "
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
        print(f"[TEST7] settled_height_m={self.settled_root[:, 2].tolist()} "
              f"camera={self.num_envs}x640x360 DINO=raw384 frozen "
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

    def _track_terrain(self) -> None:
        feet = (self.robot.data.body_pos_w[:, self.foot_ids, :2]
                - self.scene.env_origins[:, None, :2])
        col = torch.floor((feet[..., 0] + 1.0) / 2.0).long()
        row = torch.floor((feet[..., 1] + 1.0) / 2.0).long()
        inside = (row >= 0) & (row < 3) & (col >= 0) & (col < 3)
        env_id = torch.arange(self.num_envs, device=self.device)[:, None]
        codes = self.terrain_codes[env_id, row.clamp(0, 2), col.clamp(0, 2)]
        codes = torch.where(inside, codes, -1)
        for code in range(len(TERRAIN_LABELS)):
            self.terrain_entries[:, code] |= (codes == code).any(dim=1)

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
        if actions.shape != (self.num_envs, 2):
            raise ValueError(f"Expected {(self.num_envs, 2)} actions, got {actions.shape}")
        self.command = navigation_commands(actions)
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
        self.fallen = (pos[:, 2] < self.nav.minimum_body_height_m) | (gravity > -0.5)
        self.out_of_bounds = ((pos[:, 0] < self.geometry.world_x_bounds[0])
                  | (pos[:, 0] > self.geometry.world_x_bounds[1])
                  | (pos[:, 1] < self.geometry.world_y_bounds[0])
                  | (pos[:, 1] > self.geometry.world_y_bounds[1]))
        unsafe = self.fallen | self.out_of_bounds
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
            "fallen": self.fallen.clone(),
            "out_of_bounds": self.out_of_bounds.clone(),
            "episode_energy_j": self.episode_energy.clone(),
            "path_length_m": self.path_length.clone(),
            "final_distance_m": distance.clone(),
            "visual_ms": torch.full((self.num_envs,), self.last_visual_ms / self.num_envs,
                                    device=self.device),
            "dino_ms": torch.full((self.num_envs,), self.last_dino_ms / self.num_envs,
                                  device=self.device),
        }
        body_xy = xy - self.scene.env_origins[:, :2]
        self.extras["body_x_m"] = body_xy[:, 0].clone()
        self.extras["body_y_m"] = body_xy[:, 1].clone()
        self.extras["command_vx_m_s"] = self.command[:, 0].clone()
        self.extras["command_vy_m_s"] = self.command[:, 1].clone()
        self.extras["command_wz_rad_s"] = self.command[:, 2].clone()
        for code, label in enumerate(TERRAIN_LABELS):
            self.extras[f"{label}_entry"] = self.terrain_entries[:, code].clone()
        return reward

    def _get_observations(self) -> dict:
        data = self.robot.data
        delta = self.goal_w - data.root_pos_w
        delta_body = quat_apply_inverse(yaw_quat(data.root_quat_w), delta)
        # Retain the 11 test4 state values. The previous lateral command is
        # always zero; divide that slot by 1 rather than the zero lateral limit.
        maxima = torch.tensor((self.nav.max_forward_m_s, 1.0,
                               self.nav.max_yaw_rad_s), device=self.device)
        state = torch.cat((
            delta_body[:, :2] / 10.0, data.root_lin_vel_b[:, :2],
            data.root_ang_vel_b[:, 2:3], data.projected_gravity_b,
            self.previous_command / maxima,
        ), dim=1)
        q = data.root_quat_w
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1 - 2 * (q[:, 2].square() + q[:, 3].square()))
        visual, self.visual_valid = self.vision.extract(
            self.camera, self.scene.env_origins, data.root_pos_w[:, :2], yaw
        )
        self.last_visual_ms = self.vision.last_visual_ms
        self.last_dino_ms = self.vision.last_dino_ms
        obs = torch.cat((state, visual), dim=1).float()
        if obs.shape != (self.num_envs, OBSERVATION_SIZE) or not bool(torch.isfinite(obs).all()):
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
        self.terrain_entries[env_ids] = False
        self._sync_cameras()

"""Evaluation-only map adapter and logging; training files stay unchanged."""

from contextlib import ExitStack
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from missions import MAP, METADATA, TRAINING, FixedSampler, sha
import json


def validate_map():
    from collections import Counter
    from pxr import Usd, UsdGeom, UsdPhysics
    metadata = json.loads(METADATA.read_text())
    if sha(MAP) != metadata['usd_sha256']:
        raise ValueError('Evaluation map hash differs')
    if sha(TRAINING / 'maps/real_terrains_v2.usd') != metadata['source_usd_sha256']:
        raise ValueError('Training map hash differs')
    grid = metadata['grid']
    if len(grid) != 7 or any(len(row) != 7 for row in grid):
        raise ValueError('Expected a 7 by 7 map')
    counts = Counter(name for row in grid for name in row)
    if len(counts) != 7 or set(counts.values()) != {7}:
        raise ValueError('Expected seven cells of each terrain')
    stage = Usd.Stage.Open(str(MAP))
    if not stage or len(list(stage.GetPrimAtPath('/World/Grid').GetChildren())) != 49:
        raise ValueError('Expected 49 cells')
    boxes = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
    for row, cells in enumerate(grid):
        for col, name in enumerate(cells):
            prim = stage.GetPrimAtPath(f'/World/Grid/Cell_{row}_{col}_{name}/terrain/mesh')
            if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError('Missing terrain collision cell')
            bounds = boxes.ComputeWorldBound(prim).ComputeAlignedBox()
            np.testing.assert_allclose(list(bounds.GetMin())[:2], [6*col-3, 3*row-1.5], atol=1e-5)
            np.testing.assert_allclose(list(bounds.GetMax())[:2], [6*col+3, 3*row+1.5], atol=1e-5)
    return metadata


def make_logging_env(base_class, torch):
    class LoggedEnv(base_class):
        def __init__(self, *args, suite, **kwargs):
            self.suite = suite
            self.active_case = None
            self.terminal_sample = None
            self.power_log = None
            self.last_power_tick = None
            super().__init__(*args, **kwargs)

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            self.step_was_alignment[env_ids] = True
            self.active_case = self.suite['cases'][min(int(self.eval_indices[0]) - 1, len(self.suite['cases']) - 1)]
            self.mission_start_tick = self._sim_step_counter
            self.last_power_tick = None

        def sample(self, terminal=False):
            d = self.robot.data
            position = (d.root_pos_w[0] - self.scene.env_origins[0]).tolist()
            velocity = d.root_lin_vel_w[0].tolist()
            angular = d.root_ang_vel_w[0].tolist()
            return dict(episode=self.active_case['episode'], step=int(self.episode_length_buf[0]),
                        time_s=float(self.episode_length_buf[0]) * self.step_dt,
                        phase='alignment' if bool(self.step_was_alignment[0]) else 'navigation',
                        x_m=position[0], y_m=position[1], z_m=position[2], yaw_rad=float(self.yaw()[0]),
                        vx_m_s=velocity[0], vy_m_s=velocity[1], vz_m_s=velocity[2],
                        speed_xy_m_s=math.hypot(*velocity[:2]),
                        wx_rad_s=angular[0], wy_rad_s=angular[1], wz_rad_s=angular[2],
                        body_vx_m_s=float(d.root_lin_vel_b[0, 0]), body_wz_rad_s=float(d.root_ang_vel_b[0, 2]),
                        command_vx_m_s=float(self.command[0, 0]), command_wz_rad_s=float(self.command[0, 2]),
                        action_forward=float(self.previous_action[0, 0]), action_yaw=float(self.previous_action[0, 1]),
                        energy_j=float(self.episode_energy[0]), path_m=float(self.path_length[0]),
                        goal_distance_m=float(self._distance()[0]), terminal=int(terminal))

        def _episode_records(self):
            records = super()._episode_records()
            if records:
                # Save the final state before the automatic reset.
                self.terminal_sample = self.sample(terminal=True)
                for row in records:
                    seconds = row['steps'] * self.step_dt
                    row.update(episode=self.active_case['episode'], category=self.active_case['category'],
                               time_s=seconds, mean_speed_m_s=row['path_m'] / seconds,
                               falls=int(row['reason'] == 'fallen'), timeouts=int(row['reason'] == 'timeout'),
                               distance_group_m=self.active_case['distance_group_m'])
            return records

        def _measured_power(self):
            power = super()._measured_power()
            tick = self._sim_step_counter
            if self.power_log is not None and tick != self.last_power_tick:
                self.power_log.write(dict(episode=self.active_case['episode'],
                                          physics_step=tick - self.mission_start_tick,
                                          time_s=(tick - self.mission_start_tick) * self.physics_dt,
                                          power_w=float(power[0])))
                self.last_power_tick = tick
            return power

    return LoggedEnv


def create_environment(cfg, document, pca, model_path, map_backend):
    import torch
    from train_general_big_map_test7_SAC import env as env_module, scene as scene_module
    from train_general_big_map_test7_SAC.layouts import Condition

    metadata = validate_map()
    layout = SimpleNamespace(name=MAP.stem, usd=MAP, grid=tuple(tuple(row) for row in metadata['grid']))
    env_class = make_logging_env(env_module.GeneralMapEnv, torch)
    # Change constructor inputs only inside this evaluation process.
    with ExitStack() as stack:
        replacements = (
            (env_module, 'map_metadata', lambda: metadata),
            (env_module, 'MissionSampler', lambda: FixedSampler(document)),
            (scene_module, 'validate_terrain', validate_map),
        )
        for module, name, value in replacements:
            stack.enter_context(patch.object(module, name, value))
        env = env_class(cfg, (Condition(0, layout),), suite=document, pca=pca,
                        model_path=model_path, dino_batch_size=1, map_backend=map_backend)
    env.evaluating = True
    env.eval_stage = 4
    env.capture_fall_diagnostics = True
    return env

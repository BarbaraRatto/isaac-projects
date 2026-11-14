"""Evaluation-only map adapter and logging; training files stay unchanged."""

from contextlib import ExitStack
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from missions import MAP, METADATA, DRAG, TRAINING, FixedSampler, sha
import json


def validate_map():
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    metadata = json.loads(METADATA.read_text())
    drag = json.loads(DRAG.read_text())
    materials = json.loads((TRAINING / 'materials.json').read_text())
    digest = sha(MAP)
    if digest != metadata['usd_sha256'] or drag['maps_sha256'] != {MAP.name: digest}:
        raise ValueError('Evaluation map hashes do not match')
    if materials != drag['materials'] or sha(TRAINING / 'materials.json') != metadata['materials_sha256']:
        raise ValueError('Evaluation materials differ from training materials')
    if sha(TRAINING / 'maps/lab_15x15.usda') != metadata['training_map_sha256']:
        raise ValueError('Original training map changed')
    grid = metadata['grid']
    if len(grid) != 5 or any(len(row) != 5 for row in grid):
        raise ValueError('Expected a 5 by 5 map')
    if sum(cell == 'lab_blackice' for row in grid for cell in row) != 8:
        raise ValueError('Expected eight black cells')
    stage = Usd.Stage.Open(str(MAP))
    if not stage or len(list(stage.GetPrimAtPath('/World/Grid').GetChildren())) != 25:
        raise ValueError('Expected 25 terrain collision cells')
    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    expected_regions = []
    for row, cells in enumerate(grid):
        for col, name in enumerate(cells):
            if name not in ('t1_asphalt', 'lab_blackice'):
                raise ValueError('Unknown terrain')
            key = 'black' if name == 'lab_blackice' else 'grey'
            bounds = [3 * col, 3 * row, 3 * (col + 1), 3 * (row + 1)]
            expected_regions.append(dict(material=key, bounds_xy_m=bounds))
            prim = stage.GetPrimAtPath(f'/World/Grid/Cell_{row}_{col}_{name}/terrain/mesh')
            if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError('Missing collision cell')
            box = bounds_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            np.testing.assert_allclose(box.GetMin(), [bounds[0], bounds[1], -.1], atol=1e-6)
            np.testing.assert_allclose(box.GetMax(), [bounds[2], bounds[3], 0], atol=1e-6)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            if material.GetPath().name != key:
                raise ValueError('Wrong bound terrain material')
            api = UsdPhysics.MaterialAPI(material.GetPrim())
            for value, expected in ((api.GetStaticFrictionAttr().Get(), materials[key]['static_friction']),
                                    (api.GetDynamicFrictionAttr().Get(), materials[key]['dynamic_friction']),
                                    (api.GetRestitutionAttr().Get(), materials[key]['restitution'])):
                if not math.isclose(value, expected, abs_tol=1e-6):
                    raise ValueError('Terrain friction differs from training')
    if drag['surface_regions'] != {MAP.name: expected_regions}:
        raise ValueError('Resistance regions do not follow the evaluation map')
    if any(cfg.get('horizontal_drag_ns_per_m', 0.) != 0 for cfg in materials.values()):
        raise ValueError('Test13 must not apply additional resistance')
    if drag['runtime_drag']['model'] != 'disabled':
        raise ValueError('The resistance model must be disabled for test13')
    if not math.isclose(materials['black']['static_friction'], .10) or not math.isclose(materials['black']['dynamic_friction'], .08):
        raise ValueError('Expected the trained black ice friction')
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
            self.active_case = self.suite['cases'][min(int(self.eval_indices[0]) - 1, 19)]
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
                               direct_black_length_m=self.active_case['direct_black_length_m'])
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


def create_environment(cfg, document, pca, model_path, map_backend, rgb_backend):
    import torch
    from train_general_big_map_test13_SAC_Re import env as env_module, scene as scene_module
    from train_general_big_map_test13_SAC_Re.fast_drag import BatchedSurfaceDrag
    from train_general_big_map_test13_SAC_Re.layouts import Condition

    metadata = validate_map()
    layout = SimpleNamespace(name=MAP.stem, usd=MAP, grid=tuple(tuple(row) for row in metadata['grid']))
    env_class = make_logging_env(env_module.GeneralMapEnv, torch)
    # Scope the constructor's map dependencies to this evaluation only.
    with ExitStack() as stack:
        replacements = (
            (env_module, 'map_metadata', lambda: metadata),
            (env_module, 'TRAIN_LAYOUTS', (layout,)),
            (env_module, 'MissionSampler', lambda: FixedSampler(document)),
            (env_module, 'BatchedSurfaceDrag', lambda *args: BatchedSurfaceDrag(DRAG, MAP)),
            (scene_module, 'validate_terrain', validate_map),
        )
        for module, name, value in replacements:
            stack.enter_context(patch.object(module, name, value))
        env = env_class(cfg, (Condition(0, layout),), suite=document, pca=pca,
                        model_path=model_path, dino_batch_size=1, map_backend=map_backend, rgb_backend=rgb_backend)
    env.evaluating = True
    env.eval_stage = 3
    env.capture_fall_diagnostics = True
    return env

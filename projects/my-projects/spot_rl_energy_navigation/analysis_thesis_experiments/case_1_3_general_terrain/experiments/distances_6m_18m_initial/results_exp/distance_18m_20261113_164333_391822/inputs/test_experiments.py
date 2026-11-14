"""Offline checks for map switching, fixed missions and terminal logging."""

import contextlib
import copy
import csv
import io
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from missions import METADATA, MAP, SUITE, FixedSampler, load_suite, segment_cells, SurfaceChecks
import runtime
from create_exp import CSVLog, validate_logs


class FakeBase:
    def __init__(self):
        self.eval_indices = np.zeros(1, dtype=int)
        self._sim_step_counter = 50
        self.step_dt, self.physics_dt = .2, .005
        self.step_was_alignment = torch.ones(1, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.scene = SimpleNamespace(env_origins=torch.tensor([[50., 80., 0.]]))
        self.robot = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor([[51., 81., .5]]),
            root_lin_vel_w=torch.tensor([[.3, .4, 0.]]), root_ang_vel_w=torch.tensor([[0., 0., .2]]),
            root_lin_vel_b=torch.tensor([[.1, .2, 0.]]), root_ang_vel_b=torch.tensor([[0., 0., .2]])))
        self.command = torch.tensor([[1., 0., .5]])
        self.previous_action = torch.zeros(1, 2)
        self.episode_energy, self.path_length = torch.tensor([2.]), torch.tensor([.2])

    def _reset_idx(self, env_ids):
        self.eval_indices[0] += 1
        self.episode_length_buf.zero_()
        self.robot.data.root_pos_w[0, :2] = torch.tensor([51., 81.])

    def yaw(self):
        return torch.tensor([0.])

    def _distance(self):
        return torch.tensor([.4])

    def _episode_records(self):
        return [dict(steps=1, path_m=.2, energy_j=2., reason='goal')]

    def _measured_power(self):
        return torch.tensor([10.])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.document = load_suite()
        self.metadata = json.loads(METADATA.read_text())

    def test_mission_coverage_and_surface_safety(self):
        from collections import Counter
        cases = self.document['cases']
        counts = Counter((c['distance_group_m'], c['category']) for c in cases)
        self.assertEqual(len(cases), 80)
        self.assertEqual(len(counts), 8)
        self.assertEqual(set(counts.values()), {10})
        checker = SurfaceChecks()
        for case in cases:
            m = case['mission']
            self.assertAlmostEqual(math.dist(m['start'][:2], m['goal'][:2]), case['distance_group_m'], places=8)
            for key in ('start', 'goal'):
                actual = checker.endpoint(m[key][:2])
                np.testing.assert_allclose(actual, m[key], atol=1e-8)

    def test_exact_segment_intersections(self):
        cells = segment_cells([0., 0.], [36., 0.], self.metadata['grid'])
        self.assertEqual([(c['row'], c['col']) for c in cells], [(0, c) for c in range(7)])
        self.assertAlmostEqual(sum(c['length_m'] for c in cells), 36.)
        self.assertAlmostEqual(cells[0]['length_m'], 3.)
        self.assertAlmostEqual(cells[-1]['length_m'], 3.)

    def test_long_subset_keeps_global_episode_ids(self):
        document = load_suite(distance='18')
        self.assertEqual([c['episode'] for c in document['cases']], list(range(41,81)))
        with contextlib.redirect_stdout(io.StringIO()):
            sampler = FixedSampler(document)
            self.assertEqual(sampler.evaluation_mission(0,0,0), document['cases'][0]['mission'])
            self.assertEqual(sampler.evaluation_mission(0,40,0), document['cases'][-1]['mission'])

    def test_fixed_sampler_and_unused_final_reset(self):
        sampler = FixedSampler(self.document)
        with contextlib.redirect_stdout(io.StringIO()):
            for index in range(80):
                self.assertEqual(sampler.evaluation_mission(0, index, 99), self.document['cases'][index]['mission'])
            last = sampler.evaluation_mission(0, 80, 99)
            last['start'][0] = -100
            self.assertNotEqual(sampler.evaluation_mission(0, 80, 99)['start'][0], -100)
        with self.assertRaises(RuntimeError):
            sampler.training(0)

    def test_stale_map_or_invalid_endpoint_is_rejected(self):
        for change in ('map', 'endpoint'):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                document = copy.deepcopy(self.document)
                if change == 'map':
                    document['map_sha256'] = 'invalid'
                else:
                    document['cases'][0]['mission']['start'][0] = -2.99
                path = Path(directory) / 'missions.json'
                path.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    load_suite(path)

    def test_terminal_snapshot_and_measured_velocity(self):
        env = runtime.make_logging_env(FakeBase, torch)(suite=self.document)
        env._reset_idx(torch.tensor([0]))
        self.assertTrue(bool(env.step_was_alignment[0]))
        env.episode_length_buf[0] = 1
        env.step_was_alignment[0] = False
        env.robot.data.root_pos_w[0, :2] = torch.tensor([55., 86.])
        record = env._episode_records()[0]
        env._reset_idx(torch.tensor([0]))
        self.assertEqual(record['episode'], 1)
        self.assertEqual(env.terminal_sample['episode'], 1)
        self.assertEqual(env.active_case['episode'], 2)
        self.assertEqual(env.terminal_sample['x_m'], 5.)
        self.assertEqual(env.terminal_sample['y_m'], 6.)
        self.assertEqual(env.terminal_sample['terminal'], 1)
        self.assertAlmostEqual(env.terminal_sample['speed_xy_m_s'], .5, places=6)
        self.assertEqual(env.terminal_sample['command_vx_m_s'], 1.)
        self.assertEqual(env.terminal_sample['phase'], 'navigation')

    def test_power_boundary_not_duplicated(self):
        env = runtime.make_logging_env(FakeBase, torch)(suite=self.document)
        env._reset_idx(torch.tensor([0]))
        samples = []
        env.power_log = SimpleNamespace(write=samples.append)
        env._measured_power()
        env._measured_power()
        env._sim_step_counter += 1
        env._measured_power()
        self.assertEqual([row['physics_step'] for row in samples], [0, 1])
        self.assertEqual(samples[-1]['episode'], 1)

    def test_constructor_uses_new_map_and_restores_training_globals(self):
        import train_general_big_map_test7_SAC as package
        env_module, scene_module = ModuleType('fake_env'), ModuleType('fake_scene')
        env_module.map_metadata = lambda: 'training metadata'
        env_module.MissionSampler = lambda: 'training sampler'
        scene_module.validate_terrain = lambda: 'training validation'
        original_metadata = env_module.map_metadata

        class CapturingBase:
            def __init__(self, cfg, conditions, **kwargs):
                self.metadata = env_module.map_metadata()
                self.sampler = env_module.MissionSampler()
                self.conditions = conditions
                self.validation = scene_module.validate_terrain()

        env_module.GeneralMapEnv = CapturingBase
        with patch.object(package, 'env', env_module, create=True), patch.object(package, 'scene', scene_module, create=True), \
                patch.object(runtime, 'validate_map', return_value=self.metadata):
            env = runtime.create_environment(None, self.document, None, None, 'cpu')
        self.assertEqual(env.conditions[0].layout.usd, MAP)
        self.assertEqual(env.conditions[0].layout.grid, tuple(tuple(row) for row in self.metadata['grid']))
        self.assertEqual(env.metadata, self.metadata)
        self.assertIsInstance(env.sampler, FixedSampler)
        self.assertIs(env_module.map_metadata, original_metadata)
        self.assertEqual(scene_module.validate_terrain(), 'training validation')

    def test_energy_integrity_and_missing_terminal_detection(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            episodes = CSVLog(output / 'episodes.csv', ['episode', 'steps', 'energy_j', 'path_m'])
            trajectory = CSVLog(output / 'trajectories.csv', ['episode', 'step', 'time_s', 'x_m', 'y_m', 'energy_j', 'terminal'])
            power = CSVLog(output / 'power_samples.csv', ['episode', 'physics_step', 'time_s', 'power_w'])
            for case in self.document['cases']:
                episode = case['episode']
                x, y = case['mission']['start'][:2]
                episodes.write(dict(episode=episode, steps=1, energy_j=2., path_m=.2))
                for step in (0, 1):
                    trajectory.write(dict(episode=episode, step=step, time_s=step * .2, x_m=x + step * .2,
                                          y_m=y, energy_j=step * 2, terminal=step))
                for tick in range(41):
                    power.write(dict(episode=episode, physics_step=tick, time_s=tick * .005, power_w=10.))
            for log in (episodes, trajectory, power):
                log.close()
            self.assertEqual(validate_logs(output, self.document, .005, 40)['logging_checks'], 'passed')
            with (output / 'episodes.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            rows[0]['energy_j'] = 20.
            with (output / 'episodes.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'Power integral'):
                validate_logs(output, self.document, .005, 40)
            with (output / 'trajectories.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            rows[1]['terminal'] = 0
            with (output / 'trajectories.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'terminal position'):
                validate_logs(output, self.document, .005, 40)


if __name__ == '__main__':
    unittest.main()

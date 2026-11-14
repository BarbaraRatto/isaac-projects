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

from missions import METADATA, MAP, DRAG, SUITE, FixedSampler, load_suite, segment_cells
import runtime
from run import CSVLog, validate_logs


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

    def test_mission_coverage(self):
        cases = self.document['cases']
        self.assertEqual(len(cases), 20)
        self.assertEqual(sum(not case['direct_black_cells'] for case in cases), 1)
        self.assertTrue(all(case['mission']['goal'][4] == 0 for case in cases))
        self.assertEqual({case['mission']['band'] for case in cases}, {0, 1, 2, 3})
        for case in cases:
            self.assertEqual(case['mission']['start'][4], 0)
            lo, hi = case['distance_range_m']
            self.assertTrue(lo <= case['mission']['distance_m'] <= hi)

    def test_exact_segment_intersections(self):
        cells = segment_cells([1.5, 1.5], [13.5, 1.5], self.metadata['grid'])
        self.assertEqual([(c['row'], c['col']) for c in cells], [(0, c) for c in range(5)])
        self.assertAlmostEqual(sum(c['length_m'] for c in cells), 12.)
        self.assertAlmostEqual(sum(c['length_m'] for c in cells if c['terrain'] == 'lab_blackice'), 1.5)

    def test_shared_missions_and_replaced_black_goals(self):
        reference_file = Path(__file__).resolve().parents[2] / 'train_general_big_map_test12_SAC_Re/new_map_eval/missions_20.json'
        reference = json.loads(reference_file.read_text())
        for index, (case, source) in enumerate(zip(self.document['cases'], reference['cases'])):
            self.assertEqual(case['mission']['start'], source['mission']['start'])
            self.assertEqual(case['mission']['yaw'], source['mission']['yaw'])
            if index < 16:
                self.assertEqual(case['mission'], source['mission'])
                self.assertFalse(case['reference_goal_replaced'])
            else:
                self.assertNotEqual(case['mission']['goal'], source['mission']['goal'])
                self.assertEqual(case['mission']['goal'][4], 0)
                self.assertTrue(case['reference_goal_replaced'])
                self.assertEqual(case['distance_range_m'], source['distance_range_m'])

    def test_fixed_sampler_and_unused_final_reset(self):
        sampler = FixedSampler(self.document)
        with contextlib.redirect_stdout(io.StringIO()):
            for index in range(20):
                self.assertEqual(sampler.evaluation_mission(0, index, 99), self.document['cases'][index]['mission'])
            last = sampler.evaluation_mission(0, 20, 99)
            last['start'][0] = -100
            self.assertNotEqual(sampler.evaluation_mission(0, 20, 99)['start'][0], -100)
        with self.assertRaises(RuntimeError):
            sampler.training(0)

    def test_stale_map_or_invalid_endpoint_is_rejected(self):
        for change in ('map', 'endpoint'):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                document = copy.deepcopy(self.document)
                if change == 'map':
                    document['map_sha256'] = 'invalid'
                else:
                    document['cases'][0]['mission']['start'][0] = 3.01
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

    def test_new_drag_regions_and_world_offsets(self):
        from train_general_big_map_test13_SAC_Re.fast_drag import BatchedSurfaceDrag
        drag = BatchedSurfaceDrag(DRAG, MAP)
        positions = torch.tensor([[3 * c + 1.5, 3 * r + 1.5, .5] for r in range(5) for c in range(5)])
        origins = torch.tensor([[50., 80., 0.]]).expand(25, -1)
        robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=positions + origins,
                                                     root_lin_vel_w=torch.tensor([[.5, -.2, 0.]]).expand(25, -1)),
                                set_external_force_and_torque=lambda **kwargs: None)
        forces = drag.apply(robot, origins)
        for index, cell in enumerate(cell for row in self.metadata['grid'] for cell in row):
            expected = [0., 0., 0.]
            torch.testing.assert_close(forces[index], torch.tensor(expected))

    def test_constructor_uses_new_map_and_restores_training_globals(self):
        import train_general_big_map_test13_SAC_Re as package
        env_module, scene_module = ModuleType('fake_env'), ModuleType('fake_scene')
        env_module.map_metadata = lambda: 'training metadata'
        env_module.TRAIN_LAYOUTS = ('training layout',)
        env_module.MissionSampler = lambda: 'training sampler'
        env_module.BatchedSurfaceDrag = lambda *args: 'training resistance'
        scene_module.validate_terrain = lambda: 'training validation'
        original_metadata = env_module.map_metadata

        class CapturingBase:
            def __init__(self, cfg, conditions, **kwargs):
                self.metadata = env_module.map_metadata()
                self.sampler = env_module.MissionSampler()
                self.drag = env_module.BatchedSurfaceDrag(None, None)
                self.conditions = conditions
                self.validation = scene_module.validate_terrain()

        env_module.GeneralMapEnv = CapturingBase
        with patch.object(package, 'env', env_module, create=True), patch.object(package, 'scene', scene_module, create=True), \
                patch.object(runtime, 'validate_map', return_value=self.metadata):
            env = runtime.create_environment(None, self.document, None, None, 'cpu', 'opencv')
        self.assertEqual(env.conditions[0].layout.usd, MAP)
        self.assertEqual(env.conditions[0].layout.grid, tuple(tuple(row) for row in self.metadata['grid']))
        self.assertEqual(env.metadata, self.metadata)
        self.assertIsInstance(env.sampler, FixedSampler)
        self.assertEqual(sum(coefficient > 0 for coefficient, _ in env.drag.regions), 0)
        self.assertIs(env_module.map_metadata, original_metadata)
        self.assertEqual(env_module.TRAIN_LAYOUTS, ('training layout',))
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

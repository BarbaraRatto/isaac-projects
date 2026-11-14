"""Offline checks; no simulator or experiment runs."""

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

import create_exp as runner


@dataclass(frozen=True)
class Geometry:
    start_jitter_m: float = 0.05


class FakeBase:
    def __init__(self):
        self.device = 'cpu'
        self.geometry = Geometry()
        self.scene = SimpleNamespace(env_origins=torch.tensor([[10., 20., 0.]]))
        self.settled_root = torch.zeros(1, 13)
        self.goal_w = torch.zeros(1, 3)
        self._sim_step_counter = 100
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.step_dt, self.physics_dt = 0.2, 0.005
        self.previous_action = torch.zeros(1, 2)
        self.command = torch.zeros(1, 3)
        self.episode_energy, self.path_length = torch.zeros(1), torch.zeros(1)
        self.fallen, self.timeout = torch.tensor([False]), torch.tensor([False])
        self.robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(1, 3),
                                                          joint_vel=torch.tensor([[2., -3.]])),
                                    root_physx_view=SimpleNamespace(
                                        get_dof_projected_joint_forces=lambda: torch.tensor([[4., 5.]])))

    def _reset_idx(self, env_ids):
        if self.geometry.start_jitter_m != 0:
            raise AssertionError('Unexpected reset jitter')
        self.robot.data.root_pos_w = self.settled_root[:, :3].clone()
        self.previous_distance = self._distance().clone()
        self.episode_length_buf.zero_()

    def _distance(self):
        return (self.goal_w[:, :2] - self.robot.data.root_pos_w[:, :2]).norm(dim=1)

    def yaw(self):
        return torch.tensor([math.pi / 4])

    def _episode_records(self):
        return [dict(steps=1, energy_j=2., path_m=0.1)]


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.missions = runner.read_missions(runner.HERE / 'exp_coordinates.csv')

    def test_fixed_missions(self):
        self.assertEqual([r['mission_id'] for r in self.missions],
                         [f'{prefix}{index}' for prefix in ('A', 'B') for index in range(1, 6)])
        self.assertEqual(self.missions[0]['start_x_m'], -0.0289)
        self.assertEqual(self.missions[-1]['goal_y_m'], 3.9914)
        self.assertEqual(sum(r['map'] == 'rocks_right' for r in self.missions), 5)

    def test_invalid_coordinates(self):
        for field, value in (('start_x_m', 0.2), ('goal_y_m', float('nan')),
                              ('mission_id', 'A2')):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                rows = [dict(row) for row in self.missions]
                rows[0][field] = value
                path = Path(directory) / 'coordinates.csv'
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=runner.COORD_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    runner.read_missions(path)

    def test_exact_reset_and_terminal_capture(self):
        env = runner.make_experiment_env(FakeBase, torch)(missions=self.missions[:5])
        ids = torch.tensor([0])
        for mission in self.missions[:5]:
            env._reset_idx(ids)
            self.assertEqual(env.active_mission['mission_id'], mission['mission_id'])
            torch.testing.assert_close(env.robot.data.root_pos_w[0, :2], torch.tensor(
                [10 + mission['start_x_m'], 20 + mission['start_y_m']]))
            torch.testing.assert_close(env.goal_w[0, :2], torch.tensor(
                [10 + mission['goal_x_m'], 20 + mission['goal_y_m']]))
            self.assertAlmostEqual(float(env.previous_distance[0]), math.hypot(
                mission['goal_x_m'] - mission['start_x_m'],
                mission['goal_y_m'] - mission['start_y_m']), places=5)
            self.assertAlmostEqual(float(env.settled_root[0, 3]), math.cos(math.pi / 8), places=6)
        env.robot.data.root_pos_w[0, :2] = torch.tensor([14., 24.])
        env.episode_length_buf[0] = 1
        records = env._episode_records()
        env._reset_idx(ids)
        self.assertEqual(records[0]['mission_id'], 'A5')
        self.assertEqual(records[0]['final_x_m'], 4.)
        self.assertEqual(env.terminal_sample['x_m'], 4.)
        self.assertEqual(env.terminal_sample['terminal'], 1)
        self.assertEqual(env.active_mission['mission_id'], 'A5')

    def test_joint_power_and_duplicate_boundary(self):
        env = runner.make_experiment_env(FakeBase, torch)(missions=self.missions[:5])
        env._reset_idx(torch.tensor([0]))
        rows = []
        env.joint_log = SimpleNamespace(write=rows.append)
        env.effort_fields, env.velocity_fields = ['a_nm', 'b_nm'], ['a_rad_s', 'b_rad_s']
        self.assertEqual(float(env._measured_power()[0]), 23.)
        env._measured_power()
        env._sim_step_counter += 1
        env._measured_power()
        self.assertEqual(len(rows), 2)
        self.assertEqual([row['physics_step'] for row in rows], [0, 1])
        self.assertEqual(rows[1]['time_s'], 0.005)

    def test_logging_integrity_rejects_missing_terminal_and_wrong_energy(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            tables = {
                'episodes.csv': [dict(mission_id='A1', steps=1, energy_j=2., path_m=0.1)],
                'trajectories.csv': [dict(mission_id='A1', step=0, terminal=0, x_m=0, y_m=0),
                                     dict(mission_id='A1', step=1, terminal=1, x_m=0.1, y_m=0)],
                'joint_samples.csv': [dict(mission_id='A1', physics_step=0, time_s=0, power_w=10),
                                      dict(mission_id='A1', physics_step=1, time_s=0.2, power_w=10)]}

            def save():
                for filename, rows in tables.items():
                    with (output / filename).open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)

            save()
            checks = runner.validate_results(output, self.missions[:1], 1, 0.2)
            self.assertEqual(checks['logging_checks'], 'passed')
            tables['episodes.csv'][0]['energy_j'] = 20
            save()
            with self.assertRaises(ValueError):
                runner.validate_results(output, self.missions[:1], 1, 0.2)
            tables['episodes.csv'][0]['energy_j'] = 2
            tables['trajectories.csv'][1]['terminal'] = 0
            save()
            with self.assertRaises(ValueError):
                runner.validate_results(output, self.missions[:1], 1, 0.2)


if __name__ == '__main__':
    unittest.main()

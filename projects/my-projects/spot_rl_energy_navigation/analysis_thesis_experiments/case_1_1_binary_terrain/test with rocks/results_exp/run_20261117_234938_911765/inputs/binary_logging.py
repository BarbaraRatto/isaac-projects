"""Binary-experiment logging and reset helpers copied from the reference runner."""
from dataclasses import replace
from pathlib import Path
import csv
import hashlib
import json
import math

def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class CSVWriter:
    def __init__(self, path, fields):
        self.stream = path.open('w', newline='', buffering=65536)
        self.writer = csv.DictWriter(self.stream, fieldnames=fields)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.close()


def make_experiment_env(base_class, torch):
    class ExperimentEnv(base_class):
        def __init__(self, *args, missions, **kwargs):
            self.missions = missions
            self.next_mission = 0
            self.active_mission = None
            self.terminal_sample = None
            self.joint_log = None
            self.last_joint_tick = None
            super().__init__(*args, **kwargs)
            self.geometry = replace(self.geometry, start_jitter_m=0.0)

        def _reset_idx(self, env_ids):
            if len(env_ids) != 1 or int(env_ids[0]) != 0:
                raise ValueError('This experiment requires one environment')
            # The last auto-reset is unused.
            self.active_mission = self.missions[min(self.next_mission, len(self.missions) - 1)]
            self.next_mission += 1
            mission = self.active_mission
            origin = self.scene.env_origins[0, :2]
            self.settled_root[0, :2] = origin + torch.tensor(
                [mission['start_x_m'], mission['start_y_m']], device=self.device)
            half_yaw = math.radians(mission['start_yaw_deg']) / 2
            self.settled_root[0, 3:7] = torch.tensor(
                [math.cos(half_yaw), 0, 0, math.sin(half_yaw)], device=self.device)
            self.goal_w[0, :2] = origin + torch.tensor(
                [mission['goal_x_m'], mission['goal_y_m']], device=self.device)
            super()._reset_idx(env_ids)
            self.mission_start_tick = self._sim_step_counter
            self.last_joint_tick = None

        def sample(self, terminal=False):
            position = (self.robot.data.root_pos_w[0] - self.scene.env_origins[0]).tolist()
            return dict(mission_id=self.active_mission['mission_id'],
                        step=int(self.episode_length_buf[0]),
                        time_s=float(self.episode_length_buf[0]) * self.step_dt,
                        x_m=position[0], y_m=position[1], z_m=position[2],
                        yaw_rad=float(self.yaw()[0]), energy_j=float(self.episode_energy[0]),
                        path_m=float(self.path_length[0]), goal_distance_m=float(self._distance()[0]),
                        action_forward=float(self.previous_action[0, 0]),
                        action_yaw=float(self.previous_action[0, 1]),
                        command_forward_m_s=float(self.command[0, 0]),
                        command_yaw_rad_s=float(self.command[0, 2]), terminal=int(terminal))

        def _episode_records(self):
            records = super()._episode_records()
            if records:
                # Capture the final pose before auto-reset.
                self.terminal_sample = self.sample(terminal=True)
                for record in records:
                    mission = self.active_mission
                    record.update(mission)
                    record.update(time_s=record['steps'] * self.step_dt,
                                  straight_distance_m=math.hypot(
                                      mission['goal_x_m'] - mission['start_x_m'],
                                      mission['goal_y_m'] - mission['start_y_m']),
                                  falls=int(bool(self.fallen[0])),
                                  timeouts=int(bool(self.timeout[0])),
                                  final_x_m=self.terminal_sample['x_m'],
                                  final_y_m=self.terminal_sample['y_m'])
            return records

        def _measured_power(self):
            effort = self.robot.root_physx_view.get_dof_projected_joint_forces()
            speed = self.robot.data.joint_vel
            if effort.shape != speed.shape:
                raise RuntimeError('Joint effort/velocity mismatch')
            power = (effort * speed).abs().sum(1)
            if not bool(torch.isfinite(power).all()):
                raise RuntimeError('Non-finite mechanical power')
            tick = self._sim_step_counter
            if self.joint_log is not None and tick != self.last_joint_tick:
                self.last_joint_tick = tick
                row = dict(mission_id=self.active_mission['mission_id'],
                           physics_step=tick - self.mission_start_tick,
                           time_s=(tick - self.mission_start_tick) * self.physics_dt,
                           power_w=float(power[0]))
                row.update(zip(self.effort_fields, effort[0].tolist()))
                row.update(zip(self.velocity_fields, speed[0].tolist()))
                self.joint_log.write(row)
            return power

    return ExperimentEnv


def validate_results(output, missions, decimation, physics_dt):
    with (output / 'episodes.csv').open(newline='') as stream:
        episodes = list(csv.DictReader(stream))
    expected = {row['mission_id'] for row in missions}
    if len(episodes) != len(missions) or {r['mission_id'] for r in episodes} != expected:
        raise ValueError('Missing or duplicate completed missions')
    with (output / 'trajectories.csv').open(newline='') as stream:
        trajectories = list(csv.DictReader(stream))
    integrals = {name: 0.0 for name in expected}
    counts = {name: 0 for name in expected}
    previous = {}
    with (output / 'joint_samples.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            name = row['mission_id']
            step, power = int(row['physics_step']), float(row['power_w'])
            if step != counts[name] or not math.isfinite(power) or power < 0:
                raise ValueError(f'Invalid joint sample: {name}, {step}')
            if not math.isclose(float(row['time_s']), step * physics_dt, abs_tol=1e-8):
                raise ValueError('Invalid joint timestamp')
            if name in previous:
                integrals[name] += 0.5 * (previous[name] + power) * physics_dt
            previous[name] = power
            counts[name] += 1
    for episode in episodes:
        name, steps = episode['mission_id'], int(episode['steps'])
        data = [row for row in trajectories if row['mission_id'] == name]
        if [int(row['step']) for row in data] != list(range(steps + 1)):
            raise ValueError(f'Missing trajectory samples: {name}')
        if [int(row['terminal']) for row in data] != [0] * steps + [1]:
            raise ValueError(f'Missing terminal pose: {name}')
        if counts[name] != steps * decimation + 1:
            raise ValueError(f'Missing joint samples: {name}')
        if not math.isclose(integrals[name], float(episode['energy_j']), rel_tol=1e-4, abs_tol=0.01):
            raise ValueError(f'Joint work does not match recorded energy: {name}')
        path = sum(math.hypot(float(b['x_m']) - float(a['x_m']),
                              float(b['y_m']) - float(a['y_m']))
                   for a, b in zip(data, data[1:]))
        if not math.isclose(path, float(episode['path_m']), rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f'Trajectory does not match recorded distance: {name}')
    return dict(completed_missions=len(episodes), logging_checks='passed')

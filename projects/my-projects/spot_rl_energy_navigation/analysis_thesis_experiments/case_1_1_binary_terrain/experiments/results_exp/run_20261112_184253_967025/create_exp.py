"""Run the ten binary-terrain missions with the multi_terrain_test13 PPO policy."""

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
TRAINING = PROJECT / 'multi_terrain_test13'
CHECKPOINT = TRAINING / 'runs/test13_ppo_20261023_122929_726370/policy_final.pt'
DINO = PROJECT / 'extracted_features_comparison_v2/models/dinov2'
MAPS = ('rocks_right', 'rocks_left')
COORD_FIELDS = ('mission_id', 'map', 'start_x_m', 'start_y_m',
                'goal_x_m', 'goal_y_m', 'start_yaw_deg')
TRAJECTORY_FIELDS = ('mission_id', 'step', 'time_s', 'x_m', 'y_m', 'z_m',
                     'yaw_rad', 'energy_j', 'path_m', 'goal_distance_m',
                     'action_forward', 'action_yaw', 'command_forward_m_s',
                     'command_yaw_rad_s', 'terminal')


def read_missions(path):
    # Fixed samples: uniform +/-5 cm around start (0, 0) and goal (4, 4).
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(COORD_FIELDS):
            raise ValueError(f'Expected CSV columns: {COORD_FIELDS}')
        rows = list(reader)
    seen = set()
    for row in rows:
        if row['mission_id'] in seen or not row['mission_id']:
            raise ValueError('Mission IDs must be unique and nonempty')
        seen.add(row['mission_id'])
        if row['map'] not in MAPS:
            raise ValueError(f"Unknown map: {row['map']}")
        for field in COORD_FIELDS[2:]:
            row[field] = float(row[field])
            if not math.isfinite(row[field]):
                raise ValueError(f'Non-finite coordinate: {field}')
        for field, center in (('start_x_m', 0), ('start_y_m', 0),
                              ('goal_x_m', 4), ('goal_y_m', 4)):
            if abs(row[field] - center) > 0.050000001:
                raise ValueError(f'{field} must be within 5 cm of {center}')
        if row['start_yaw_deg'] != 45:
            raise ValueError('Use the training heading: 45 degrees')
    if len(rows) != 10 or any(sum(r['map'] == name for r in rows) != 5 for name in MAPS):
        raise ValueError('Expected five missions per map, ten in total')
    return rows


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


def plot_results(output, layout, missions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    with (output / 'trajectories.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'t1_asphalt': '#b9b9b9', 't2_slippery': '#b3def0', 't5_large_rocks': '#bb9977'}
    for row_index, row in enumerate(layout.grid):
        for col_index, terrain in enumerate(row):
            axes[0].add_patch(Rectangle((2 * col_index - 1, 2 * row_index - 1), 2, 2,
                                       facecolor=colors[terrain], edgecolor='white'))
    for mission in missions:
        data = [row for row in rows if row['mission_id'] == mission['mission_id']]
        if not data:
            continue
        line, = axes[0].plot([float(r['x_m']) for r in data], [float(r['y_m']) for r in data],
                             label=mission['mission_id'])
        color = line.get_color()
        axes[0].scatter(mission['start_x_m'], mission['start_y_m'], color=color, marker='o')
        axes[0].scatter(mission['goal_x_m'], mission['goal_y_m'], color=color, marker='*')
        for axis, field in ((axes[1], 'energy_j'), (axes[2], 'goal_distance_m')):
            axis.plot([float(r['time_s']) for r in data], [float(r[field]) for r in data],
                      color=color, label=mission['mission_id'])
    axes[0].set(xlim=(-1, 5), ylim=(-1, 5), xlabel='x [m]', ylabel='y [m]', title=layout.name)
    axes[0].set_aspect('equal')
    axes[1].set(xlabel='Time [s]', ylabel='Absolute joint work [J]')
    axes[2].set(xlabel='Time [s]', ylabel='Distance to goal [m]')
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / 'mission_plots.png', dpi=200)
    figure.savefig(output / 'mission_plots.pdf')
    plt.close(figure)


def run_worker(args, missions):
    from isaaclab.app import AppLauncher
    args.enable_cameras = True
    app = AppLauncher(args).app
    env = None
    logs = []
    output = args.output / args.map
    output.mkdir(exist_ok=False)
    selected = [row for row in missions if row['map'] == args.map]
    try:
        import torch
        sys.path.insert(0, str(PROJECT))
        from multi_terrain_test13.artifacts import source_identity
        from multi_terrain_test13.assets import validate_assets
        from multi_terrain_test13.config import experiment_signature
        from multi_terrain_test13.env import Test13Env
        from multi_terrain_test13.layouts import ROCKS, MIRRORED, Condition
        from multi_terrain_test13.networks import ActorCritic
        from multi_terrain_test13.perception import FrozenPCA
        from multi_terrain_test13.scene import Test13Cfg

        validate_assets()
        torch.set_num_threads(args.torch_threads)
        state = torch.load(args.checkpoint, map_location=args.agent_device, weights_only=False)
        identity = source_identity(args.dino_model)
        if state['signature'] != experiment_signature() or state['identity'] != identity:
            raise ValueError('Checkpoint/configuration/asset mismatch')
        policy = ActorCritic().to(args.agent_device).eval().requires_grad_(False)
        policy.load_state_dict(state['learner']['policy'])
        layout = ROCKS if args.map == 'rocks_right' else MIRRORED
        cfg = Test13Cfg()
        cfg.scene.num_envs = 1
        cfg.seed, cfg.sim.device = 0, args.device
        experiment_class = make_experiment_env(Test13Env, torch)
        env = experiment_class(cfg, (Condition(0, layout),), missions=selected,
                               pca=FrozenPCA(**state['pca']), model_path=args.dino_model,
                               dino_batch_size=1)
        metadata = dict(map=args.map, layout=layout.name, checkpoint=str(args.checkpoint),
                        checkpoint_sha256=file_hash(args.checkpoint),
                        training_transitions=state['learner']['transitions'],
                        source_identity=identity, signature=state['signature'],
                        physics_dt_s=env.physics_dt, action_dt_s=env.step_dt,
                        goal_radius_m=env.nav.goal_radius_m,
                        timeout_s=env.nav.max_episode_seconds,
                        energy='Absolute joint mechanical work; trapezoidal physics-step integration',
                        policy='Deterministic PPO actor; frozen normalizer, DINOv2 and PCA',
                        device=args.device, agent_device=args.agent_device,
                        simulation_seed=0, missions=selected)
        write_json(output / 'run_config.json', metadata)
        coordinates = CSVWriter(output / 'missions.csv', COORD_FIELDS)
        for mission in selected:
            coordinates.write(mission)
        coordinates.close()
        trajectory = CSVWriter(output / 'trajectories.csv', TRAJECTORY_FIELDS)
        logs.append(trajectory)
        env.effort_fields = [f'{name}_effort_nm' for name in env.robot.joint_names]
        env.velocity_fields = [f'{name}_velocity_rad_s' for name in env.robot.joint_names]
        env.joint_log = CSVWriter(output / 'joint_samples.csv',
                                 ['mission_id', 'physics_step', 'time_s', 'power_w',
                                  *env.effort_fields, *env.velocity_fields])
        logs.append(env.joint_log)
        observations, _ = env.reset(seed=0)
        trajectory.write(env.sample())
        episode_log = None
        completed = 0
        while app.is_running() and completed < len(selected):
            env.terminal_sample = None
            with torch.inference_mode():
                observation = observations['policy'].to(args.agent_device)
                actions = policy.actor(policy.normalizer.actor(observation)).tanh()
                observations, _, _, _, extras = env.step(actions)
            trajectory.write(env.terminal_sample if env.terminal_sample is not None else env.sample())
            for record in extras['episodes']:
                if episode_log is None:
                    episode_log = CSVWriter(output / 'episodes.csv', list(record))
                    logs.append(episode_log)
                episode_log.write(record)
                completed += 1
                for log in logs:
                    log.flush()
                print(f"[EXPERIMENT] {record['mission_id']}: {record['reason']}, "
                      f"{record['energy_j']:.2f} J, {record['time_s']:.1f} s", flush=True)
                if completed < len(selected):
                    trajectory.write(env.sample())
        if completed != len(selected):
            raise RuntimeError(f'Interrupted: completed {completed}/{len(selected)} missions')
        for log in logs:
            log.flush()
        checks = validate_results(output, selected, cfg.decimation, env.physics_dt)
        if not args.no_plots:
            plot_results(output, layout, selected)
        write_json(output / 'complete.json', checks)
    except BaseException as error:
        write_json(output / 'interrupted.json', dict(error=repr(error)))
        raise
    finally:
        for log in logs:
            log.close()
        if env is not None:
            env.close()
        app.close(skip_cleanup=args.headless)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', choices=('both', *MAPS), default='both')
    parser.add_argument('--coordinates', type=Path, default=HERE / 'exp_coordinates.csv')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--dino-model', type=Path, default=DINO)
    parser.add_argument('--agent-device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--check', action='store_true', help='Check inputs without starting the simulator')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return parser


def main():
    parser = build_parser()
    if '--worker' in sys.argv[1:]:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
    else:
        # Simulator options are forwarded to each map session.
        parser.add_argument('--device', default='cuda:0')
        parser.add_argument('--headless', action='store_true')
        args = parser.parse_args()
    missions = read_missions(args.coordinates)
    for path in (args.checkpoint, args.dino_model / 'model.safetensors',
                 args.dino_model / 'config.json', args.dino_model / 'preprocessor_config.json',
                 TRAINING / 'maps/rocks_right.usd', TRAINING / 'maps/rocks_left.usd'):
        if not path.is_file():
            parser.error(f'Missing input: {path}')
    if args.torch_threads < 1:
        parser.error('--torch-threads must be positive')
    if args.check:
        print('Inputs verified: ten fixed missions, two maps, checkpoint and DINOv2 files.')
        return
    if args.worker:
        if args.map == 'both' or args.output is None:
            parser.error('A worker requires one map and an output directory')
        run_worker(args, missions)
        return
    output = (args.output or HERE / 'results_exp' /
              datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.coordinates, output / 'exp_coordinates.csv')
    shutil.copy2(__file__, output / 'create_exp.py')
    write_json(output / 'run_config.json', dict(
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_hash(args.checkpoint),
        coordinates_sha256=file_hash(args.coordinates), runner_sha256=file_hash(__file__),
        map_selection=args.map, command=sys.argv, started_at=datetime.now().astimezone().isoformat()))
    print(f'[EXPERIMENT] Results: {output}', flush=True)
    try:
        for name in MAPS if args.map == 'both' else (args.map,):
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                       '--map', name, '--output', str(output), '--worker']
            subprocess.run(command, check=True)
        write_json(output / 'complete.json', dict(maps=list(MAPS) if args.map == 'both' else [args.map]))
    except BaseException as error:
        write_json(output / 'interrupted.json', dict(error=repr(error)))
        raise


if __name__ == '__main__':
    main()

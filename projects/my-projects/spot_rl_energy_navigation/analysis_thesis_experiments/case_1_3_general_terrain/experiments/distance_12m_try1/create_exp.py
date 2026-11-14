"""Run 800 twelve-metre missions with test7 policy_best_stage_3."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from missions import HERE, TRAINING, PROJECT, MAP, METADATA, SUITE, CHECKPOINT, load_suite, sha, write_json


class CSVLog:
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


def validate_logs(output, document, dt, decimation):
    with (output / 'episodes.csv').open(newline='') as stream:
        episodes = list(csv.DictReader(stream))
    if [int(row['episode']) for row in episodes] != [c['episode'] for c in document['cases']]:
        raise ValueError('Missing or unordered episode results')
    with (output / 'trajectories.csv').open(newline='') as stream:
        trajectories = list(csv.DictReader(stream))
    counts, integrals, previous = {}, {}, {}
    with (output / 'power_samples.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            episode, tick, power = int(row['episode']), int(row['physics_step']), float(row['power_w'])
            if tick != counts.get(episode, 0) or not math.isfinite(power) or power < 0:
                raise ValueError('Invalid physics power log')
            if not math.isclose(float(row['time_s']), tick * dt, abs_tol=1e-8):
                raise ValueError('Invalid physics timestamp')
            if episode in previous:
                integrals[episode] += .5 * (previous[episode] + power) * dt
            else:
                integrals[episode] = 0.
            previous[episode] = power
            counts[episode] = tick + 1
    for result, case in zip(episodes, document['cases']):
        episode, steps = int(result['episode']), int(result['steps'])
        data = [row for row in trajectories if int(row['episode']) == episode]
        if [int(row['step']) for row in data] != list(range(steps + 1)):
            raise ValueError(f'Missing trajectory samples: Ep {episode}')
        if [int(row['terminal']) for row in data] != [0] * steps + [1]:
            raise ValueError('Missing terminal position')
        for row in data:
            if not math.isclose(float(row['time_s']), int(row['step']) * dt * decimation, abs_tol=1e-8):
                raise ValueError('Invalid trajectory timestamp')
        start = case['mission']['start']
        if math.dist([float(data[0]['x_m']), float(data[0]['y_m'])], start[:2]) > 1e-4:
            raise ValueError('Reset does not match the fixed start')
        path = sum(math.hypot(float(b['x_m']) - float(a['x_m']), float(b['y_m']) - float(a['y_m']))
                   for a, b in zip(data, data[1:]))
        if not math.isclose(path, float(result['path_m']), rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError('Trajectory and traveled distance differ')
        if counts[episode] != steps * decimation + 1:
            raise ValueError('Missing power samples')
        if not math.isclose(integrals[episode], float(result['energy_j']), rel_tol=1e-4, abs_tol=.02):
            raise ValueError('Power integral and recorded energy differ')
        if not math.isclose(float(data[-1]['energy_j']), float(result['energy_j']), rel_tol=1e-6, abs_tol=.001):
            raise ValueError('Terminal energy differs')
    return dict(logging_checks='passed', completed_missions=len(document['cases']))


def load_checkpoint(args):
    import torch
    from train_general_big_map_test7_SAC.artifacts import source_identity
    from train_general_big_map_test7_SAC.config import SAC_CFG
    from train_general_big_map_test7_SAC.networks import SACPolicy
    from train_general_big_map_test7_SAC.perception import FrozenPCA
    from train_general_big_map_test7_SAC.runtime_config import playback_signature

    torch.set_num_threads(args.torch_threads)
    state = torch.load(args.checkpoint, map_location=args.agent_device, weights_only=False)
    # Validate training identity separately from the new evaluation map.
    if state['identity'] != source_identity(args.dino_model):
        raise ValueError('Checkpoint does not match the trained DINO/material/map identity')
    if state['signature'] != playback_signature(SAC_CFG, state['signature']):
        raise ValueError('Checkpoint and policy configuration differ')
    policy = SACPolicy().to(args.agent_device).eval().requires_grad_(False)
    policy.load_state_dict(state['learner']['policy'])
    return state, policy, FrozenPCA(**state['pca'])


def main():
    from train_general_big_map_test7_SAC.config import LOCAL_DINO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--missions', type=Path, default=SUITE)
    parser.add_argument('--distance', choices=('12', 'all'), default='12')
    parser.add_argument('--category', choices=('all','regular','ice','stairs','surface_choice'), default='all')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--agent-device', default='cpu')
    parser.add_argument('--dino-model', type=Path, default=LOCAL_DINO)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--map-backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--check', action='store_true', help='Check inputs without starting the simulator')
    offline = '--check' in sys.argv or '--help' in sys.argv or '-h' in sys.argv
    if offline:
        parser.add_argument('--device', default='cuda:0')
        parser.add_argument('--headless', action='store_true')
    else:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.torch_threads < 1 or not args.checkpoint.is_file():
        parser.error('A valid checkpoint and positive thread count are required')
    document = load_suite(args.missions, args.distance)
    if args.category != 'all':
        document['cases'] = [case for case in document['cases'] if case['category'] == args.category]
    total = len(document['cases'])
    if args.output is not None and args.output.exists():
        parser.error('--output must be a new directory')
    if args.check:
        from runtime import validate_map
        from train_general_big_map_test7_SAC.assets import validate_assets
        validate_assets()
        validate_map()
        state, _, _ = load_checkpoint(args)
        print(f"Inputs verified: {total} missions, new map, frozen policy at {state['learner']['transitions']} transitions.")
        return

    args.enable_cameras = True
    app = AppLauncher(args).app
    env, logs = None, []
    output_created = False
    output = args.output or HERE / 'results_exp' / datetime.now().strftime(f'distance_{args.distance}m_{args.category}_%Y%m%d_%H%M%S_%f')
    try:
        import torch
        from runtime import create_environment
        from train_general_big_map_test7_SAC.assets import validate_assets
        from train_general_big_map_test7_SAC.scene import GeneralMapCfg

        validate_assets()
        state, policy, pca = load_checkpoint(args)
        output.mkdir(parents=True, exist_ok=False)
        output_created = True
        frozen = output / 'inputs'
        frozen.mkdir()
        for path in (MAP, METADATA, args.missions, HERE / 'selection_provenance.json', HERE / 'retained_reference.json', HERE / 'coverage_summary.json', HERE / 'requested_cell_pairs.json',
                     HERE / 'README.md', *sorted(HERE.glob('exp_coordinates*.csv')), *sorted(HERE.glob('*.py'))):
            shutil.copy2(path, frozen / path.name)
        training_sources = frozen / 'training_sources'
        training_sources.mkdir()
        for path in sorted(TRAINING.glob('*.py')):
            shutil.copy2(path, training_sources / path.name)
        write_json(frozen / 'selected_missions.json', document)
        config = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                      checkpoint_transitions=state['learner']['transitions'], requested_distance=args.distance, requested_category=args.category, selected_episodes=[c['episode'] for c in document['cases']],
                      checkpoint_best_score=state['best_score'], checkpoint_training_state=state['training_state'], map=str(MAP), map_sha256=sha(MAP),
                      map_metadata_sha256=sha(METADATA),
                      mission_file='selected_missions.json', coordinate_source=args.missions.name, missions_sha256=sha(args.missions),
                      signature=state['signature'], training_identity=state['identity'],
                      command=sys.argv, device=args.device, agent_device=args.agent_device,
                      map_backend=args.map_backend, simulation_seed=20000,
                      energy='Absolute joint mechanical work; physics-step trapezoidal integration',
                      time='Simulated time; includes automatic initial alignment',
                      policy='Deterministic; no learning or normalizer/PCA updates')
        write_json(output / 'evaluation_config.json', config)
        cfg = GeneralMapCfg()
        cfg.scene.num_envs = 1
        cfg.seed, cfg.sim.device = 20000, args.device
        env = create_environment(cfg, document, pca, args.dino_model, args.map_backend)
        config.update(goal_radius_m=env.nav.goal_radius_m, physics_dt_s=env.physics_dt,
                      navigation_dt_s=env.step_dt)
        write_json(output / 'evaluation_config.json', config)
        env.power_log = CSVLog(output / 'power_samples.csv', ['episode', 'physics_step', 'time_s', 'power_w'])
        logs.append(env.power_log)
        observations, _ = env.reset(seed=20000)
        initial = env.sample()
        trajectory = CSVLog(output / 'trajectories.csv', list(initial))
        trajectory.write(initial)
        logs.append(trajectory)
        episode_log, diagnostic_log = None, None
        records = []
        while app.is_running() and len(records) < total:
            env.terminal_sample = None
            with torch.no_grad():
                action, _, _, _ = policy.act(observations['policy'].to(args.agent_device),
                                             observations['critic'].to(args.agent_device), deterministic=True)
                observations, _, _, _, extras = env.step(action)
            trajectory.write(env.terminal_sample if env.terminal_sample is not None else env.sample())
            for sample in extras['fall_diagnostics']:
                episode = document['cases'][len(records)]['episode']
                if diagnostic_log is None:
                    diagnostic_log = CSVLog(output / 'fall_diagnostics.csv', ['episode', *sample])
                    logs.append(diagnostic_log)
                diagnostic_log.write(dict(episode=episode, **sample))
            for record in extras['episodes']:
                if episode_log is None:
                    episode_log = CSVLog(output / 'episodes.csv', list(record))
                    logs.append(episode_log)
                episode_log.write(record)
                records.append(record)
                for log in logs:
                    log.flush()
                write_json(output / 'summary.json', dict(completed_missions=len(records), episodes=records))
                print(f"[RESULT] Ep {record['episode']} ({len(records)}/{total}): {record['reason']}; "
                      f"{record['energy_j']:.1f} J; {record['path_m']:.2f} m; {record['time_s']:.1f} s", flush=True)
                if len(records) < total:
                    trajectory.write(env.sample())
        if len(records) != total:
            raise RuntimeError(f'Interrupted after {len(records)}/{total} completed missions')
        for log in logs:
            log.flush()
        checks = validate_logs(output, document, env.physics_dt, cfg.decimation)
        if not args.no_plots:
            from plots import plot_results
            plot_results(output, document)
        write_json(output / 'complete.json', checks)
        print(f'[RESULTS] {output.resolve()}', flush=True)
    except BaseException as error:
        if output_created:
            write_json(output / 'interrupted.json', dict(error=repr(error)))
        raise
    finally:
        for log in logs:
            log.close()
        if env is not None:
            env.close()
        app.close(skip_cleanup=args.headless)


if __name__ == '__main__':
    main()

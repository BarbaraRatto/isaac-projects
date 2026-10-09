"""Evaluate a medium-distance checkpoint with live DINO; no training."""

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--atlas', type=Path, default=HERE.parent / 'runs' / 'ice_choice_atlas.npz')
parser.add_argument('--episodes', type=int, default=20)
parser.add_argument('--seed', type=int, default=0, help='fixed seed for reproducible new evaluations')
parser.add_argument('--output', type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint.is_file():
    parser.error(f'Missing checkpoint: {args.checkpoint}')
if not args.atlas.is_file():
    parser.error(f'Missing PCA atlas: {args.atlas}')
if args.episodes < 1:
    parser.error('--episodes must be positive')
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

from stable_baselines3 import PPO  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from medium_distance_choice.task_config import (  # noqa: E402
    GEOMETRY_MEDIUM, NAV_CFG_MEDIUM, START_XY, START_YAW_RAD, TERRAIN_LABELS, TERRAIN_NAMES,
    read_terrain_codes,
)
from medium_distance_choice.env import MediumDistanceEnv  # noqa: E402
from medium_distance_choice.plot_trajectories import render_trajectory_plots  # noqa: E402
from full_grid_energy_choice.scene import FullGridChoiceCfg  # noqa: E402
from rl_config import TERRAIN_USD, validate_assets  # noqa: E402


def number(info, key):
    value = info[key]
    return float(value.item()) if hasattr(value, 'item') else float(value)


def main():
    validate_assets()
    terrain_codes = read_terrain_codes()
    saved_config = args.checkpoint.parent / 'config.json'
    if not saved_config.is_file():
        raise RuntimeError(f'Missing checkpoint configuration: {saved_config}')
    saved = json.loads(saved_config.read_text(encoding='utf-8'))
    if (saved.get('environment') != asdict(NAV_CFG_MEDIUM)
            or saved.get('scene') != json.loads(json.dumps(asdict(GEOMETRY_MEDIUM)))
            or saved.get('terrain_usd') != str(TERRAIN_USD)
            or saved.get('atlas_sha256') != hashlib.sha256(args.atlas.read_bytes()).hexdigest()):
        raise RuntimeError('Checkpoint was trained with another scene or velocity limits')
    cfg = FullGridChoiceCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.episode_length_s = NAV_CFG_MEDIUM.max_episode_seconds
    cfg.robot_cfg.init_state.pos = (*START_XY, NAV_CFG_MEDIUM.spawn_height_m)
    half = START_YAW_RAD / 2.0
    cfg.robot_cfg.init_state.rot = (math.cos(half), 0.0, 0.0, math.sin(half))
    cfg.viewer.eye = (31.0, 24.0, 23.0)
    cfg.viewer.lookat = (12.0, 9.0, 0.0)
    env = MediumDistanceEnv(cfg, args.atlas, dino_batch_size=1)
    try:
        wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
        model = PPO.load(str(args.checkpoint), device='cpu')
        if model.observation_space != wrapped.observation_space or model.action_space != wrapped.action_space:
            raise RuntimeError('Checkpoint and full-grid environment have incompatible spaces')
        obs = wrapped.reset()
        rows = []
        trajectory_rows = []
        for episode in range(1, args.episodes + 1):
            steps = 0
            # The previous terminal step has already caused Isaac Lab to reset
            # Spot. Read its current local position as the next route's start.
            initial_xy = (env.robot.data.root_pos_w[0, :2]
                          - env.scene.env_origins[0, :2]).detach().cpu().tolist()
            trajectory_rows.append({
                'episode': episode, 'decision': 0, 'time_s': 0.0,
                'x_m': round(initial_xy[0], 5), 'y_m': round(initial_xy[1], 5),
                'energy_j': 0.0,
            })
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = wrapped.step(action)
                steps += 1
                # These info values were captured before Isaac Lab reset Spot,
                # including the final position of a completed episode.
                step_info = infos[0]
                trajectory_rows.append({
                    'episode': episode, 'decision': steps,
                    'time_s': round(steps * env.step_dt, 3),
                    'x_m': round(number(step_info, 'body_x_m'), 5),
                    'y_m': round(number(step_info, 'body_y_m'), 5),
                    'energy_j': round(number(step_info, 'episode_energy_j'), 3),
                })
                if bool(dones[0]):
                    break
            info = infos[0]
            success = bool(number(info, 'is_success'))
            row = {
                'episode': episode,
                'success': int(success),
                'reason': 'goal' if success else ('timeout' if info.get('TimeLimit.truncated') else 'unsafe'),
                'energy_j': round(number(info, 'episode_energy_j'), 2),
                'path_length_m': round(number(info, 'path_length_m'), 3),
                'final_distance_m': round(number(info, 'final_distance_m'), 3),
                'decisions': steps,
            }
            row.update({f'{label}_entry': int(number(info, f'{label}_entry'))
                        for label in TERRAIN_LABELS})
            row['energy_j_per_m'] = round(row['energy_j'] / row['path_length_m'], 2) if row['path_length_m'] > 0 else float('nan')
            rows.append(row)
            print(f'[PLAY-MEDIUM] {row}', flush=True)
        output = args.output or HERE / 'results' / f'medium_eval_{datetime.now():%Y%m%d_%H%M%S}.csv'
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        trajectory_output = output.with_name(output.stem + '_trajectory.csv')
        with trajectory_output.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(trajectory_rows[0]))
            writer.writeheader()
            writer.writerows(trajectory_rows)
        grid_output = output.with_name(output.stem + '_grid.csv')
        with grid_output.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=('row', 'col', 'terrain'))
            writer.writeheader()
            for row_index, terrain_row in enumerate(terrain_codes):
                for col_index, code in enumerate(terrain_row):
                    writer.writerow({'row': row_index, 'col': col_index,
                                     'terrain': TERRAIN_NAMES[code]})
        print(f'[PLAY-MEDIUM] summary={output.resolve()}', flush=True)
        print(f'[PLAY-MEDIUM] trajectory={trajectory_output.resolve()}', flush=True)
        print(f'[PLAY-MEDIUM] grid={grid_output.resolve()}', flush=True)
        for plot_path in render_trajectory_plots(output):
            print(f'[PLAY-MEDIUM] plot={plot_path.resolve()}', flush=True)
        arrivals = [row for row in rows if row['success']]
        if arrivals:
            mean_energy = sum(row['energy_j'] for row in arrivals) / len(arrivals)
            print(f'[PLAY-MEDIUM] arrivals={len(arrivals)}/{len(rows)} '
                  f'mean_energy_of_arrivals={mean_energy:.2f}J', flush=True)
        else:
            print(f'[PLAY-MEDIUM] arrivals=0/{len(rows)}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(skip_cleanup=args.headless)

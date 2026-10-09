"""Evaluate a ramp-choice checkpoint with live DINO; no weights are updated."""

import argparse
import csv
import sys
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--atlas', type=Path, default=HERE.parent / 'runs' / 'ice_choice_atlas.npz')
parser.add_argument('--episodes', type=int, default=20)
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

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from ramp_energy_choice.build_scene import build as build_scene  # noqa: E402
from ramp_energy_choice.env import RampChoiceEnv  # noqa: E402
from ramp_energy_choice.scene import RampChoiceCfg  # noqa: E402
from rl_config import validate_assets  # noqa: E402


def number(info, key):
    value = info[key]
    return float(value.item()) if hasattr(value, 'item') else float(value)


def main():
    validate_assets()
    build_scene()
    cfg = RampChoiceCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = RampChoiceEnv(cfg, args.atlas, dino_batch_size=1)
    try:
        wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
        model = PPO.load(str(args.checkpoint), device='cpu')
        if model.observation_space != wrapped.observation_space or model.action_space != wrapped.action_space:
            raise RuntimeError('Checkpoint and ramp-choice environment have incompatible spaces')
        obs = wrapped.reset()
        rows = []
        for episode in range(1, args.episodes + 1):
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = wrapped.step(action)
                steps += 1
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
                'ramp_entry': int(number(info, 'ramp_entry')),
                'asphalt_entry': int(number(info, 'asphalt_entry')),
                'final_distance_m': round(number(info, 'final_distance_m'), 3),
                'decisions': steps,
            }
            row['energy_j_per_m'] = round(row['energy_j'] / row['path_length_m'], 2) if row['path_length_m'] > 0 else float('nan')
            rows.append(row)
            print(f'[PLAY-RAMP] {row}', flush=True)
        output = args.output or HERE / 'results' / f'ramp_energy_eval_{datetime.now():%Y%m%d_%H%M%S}.csv'
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'[PLAY-RAMP] saved={output.resolve()}', flush=True)
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

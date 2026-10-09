"""Optional measured-energy sanity check for two hand-guided routes.

No policy is trained. Both routes use the same Spot controller and speed cap.
The result compares these specified routes, not their global optima.
"""

import argparse
import csv
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
parser = argparse.ArgumentParser()
parser.add_argument('--atlas', type=Path, default=HERE.parent / 'runs' / 'ice_choice_atlas.npz')
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--speed', type=float, default=0.8)
parser.add_argument('--output', type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.atlas.is_file():
    parser.error(f'Missing PCA atlas: {args.atlas}')
if args.repeats < 1 or not 0 < args.speed <= 0.9:
    parser.error('--repeats must be positive and --speed must be in (0, 0.9]')
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from ramp_energy_choice.build_scene import build as build_scene  # noqa: E402
from ramp_energy_choice.env import RampChoiceEnv  # noqa: E402
from ramp_energy_choice.scene import RampChoiceCfg  # noqa: E402
from rl_config import validate_assets  # noqa: E402

ROUTES = {
    'direct_ramp': ((28.5, 3.0),),
    'asphalt_detour': ((21.0, 0.8), (27.0, 0.8), (28.5, 3.0)),
}


def number(info, key):
    value = info[key]
    return float(value.item()) if hasattr(value, 'item') else float(value)


def velocity_action(env, target):
    pose = env.robot.data.root_pos_w[0] - env.scene.env_origins[0]
    q = env.robot.data.root_quat_w[0]
    x, y = float(pose[0]), float(pose[1])
    yaw = math.atan2(2 * float(q[0] * q[3] + q[1] * q[2]),
                     1 - 2 * float(q[2] ** 2 + q[3] ** 2))
    dx, dy = target[0] - x, target[1] - y
    distance = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)
    error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
    forward = args.speed * max(0.0, math.cos(error)) if abs(error) < 0.65 else 0.0
    turn = float(np.clip(2.0 * error, -env.nav.max_yaw_rad_s, env.nav.max_yaw_rad_s))
    return np.array([[forward / env.nav.max_forward_m_s, 0.0,
                      turn / env.nav.max_yaw_rad_s]], dtype=np.float32), distance


def main():
    validate_assets()
    build_scene()
    cfg = RampChoiceCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = RampChoiceEnv(cfg, args.atlas, dino_batch_size=1)
    try:
        wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
        rows = []
        for route_name, waypoints in ROUTES.items():
            for trial in range(1, args.repeats + 1):
                wrapped.reset()
                waypoint = 0
                for step in range(1, env.max_episode_length + 2):
                    action, distance = velocity_action(env, waypoints[waypoint])
                    if distance < 0.5 and waypoint < len(waypoints) - 1:
                        waypoint += 1
                        action, _ = velocity_action(env, waypoints[waypoint])
                    _, _, dones, infos = wrapped.step(action)
                    if bool(dones[0]):
                        break
                info = infos[0]
                row = {
                    'route': route_name, 'trial': trial,
                    'success': int(number(info, 'is_success')),
                    'energy_j': round(number(info, 'episode_energy_j'), 2),
                    'path_length_m': round(number(info, 'path_length_m'), 3),
                    'ramp_entry': int(number(info, 'ramp_entry')),
                    'asphalt_entry': int(number(info, 'asphalt_entry')),
                    'final_distance_m': round(number(info, 'final_distance_m'), 3),
                    'decisions': step, 'speed_cap_m_s': args.speed,
                }
                rows.append(row)
                print(f'[ROUTE-CHECK] {row}', flush=True)
        output = args.output or HERE / 'results' / f'fixed_routes_{datetime.now():%Y%m%d_%H%M%S}.csv'
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'[ROUTE-CHECK] saved={output.resolve()}', flush=True)
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

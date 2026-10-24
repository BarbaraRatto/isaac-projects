"""Evaluate the test8 policy on one centered rocks/gravel layout."""

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--layout", required=True, help="layout name from layouts.py")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--agent-device", default="cpu")
parser.add_argument("--torch-threads", type=int, default=4)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint.is_file():
    parser.error("--checkpoint must point to an existing file")
if args.torch_threads < 1:
    parser.error("--torch-threads must be positive")
if args.episodes < 1:
    parser.error("--episodes must be positive")
if args.output is not None and not args.output.resolve().is_relative_to(HERE.resolve()):
    parser.error("--output must be inside multi_terrain_test8")
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
import torch  # noqa: E402
from multi_terrain_test8.networks import TwoBranchExtractor  # noqa: E402,F401
from multi_terrain_test8.utils import validate_checkpoint_config, reason  # noqa: E402

from multi_terrain_test8.config import TERRAIN_LABELS  # noqa: E402
from multi_terrain_test8.terrain import read_terrain_codes  # noqa: E402
from multi_terrain_test8.assets import validate_assets  # noqa: E402
from multi_terrain_test8.config import NAV_CFG  # noqa: E402
from multi_terrain_test8.env import Test8Env  # noqa: E402
from multi_terrain_test8.layouts import (  # noqa: E402
    BY_NAME, evaluation_condition,
)
from multi_terrain_test8.scene import Test8Cfg  # noqa: E402


def number(info, key):
    value = info[key]
    return float(value.item()) if hasattr(value, "item") else float(value)


def main() -> None:
    validate_assets()
    validate_checkpoint_config(args.checkpoint)
    torch.set_num_threads(args.torch_threads)
    if args.layout in BY_NAME:
        condition = evaluation_condition(args.layout)
    else:
        raise ValueError(f"Unknown layout: {args.layout}")
    case = condition.case
    terrain_codes = read_terrain_codes(case)
    cfg = Test8Cfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.episode_length_s = NAV_CFG.max_episode_seconds
    cfg.robot_cfg.init_state.pos = (0.0, 0.0, NAV_CFG.spawn_height_m)
    half = math.pi / 8.0
    cfg.robot_cfg.init_state.rot = (math.cos(half), 0.0, 0.0, math.sin(half))
    cfg.viewer.eye = (8.0, 9.0, 11.0)
    cfg.viewer.lookat = (2.0, 2.0, 0.0)
    env = Test8Env(cfg, (condition,), dino_batch_size=1)
    try:
        wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
        model = SAC.load(str(args.checkpoint), device=args.agent_device)
        if model.observation_space != wrapped.observation_space or model.action_space != wrapped.action_space:
            raise RuntimeError("Checkpoint has incompatible action or observation space")
        obs = wrapped.reset()
        rows = []
        trajectories = []
        for episode in range(1, args.episodes + 1):
            start_xy = (env.robot.data.root_pos_w[0, :2]
                        - env.scene.env_origins[0, :2]).detach().cpu().tolist()
            trajectories.append({"episode": episode, "decision": 0, "time_s": 0.0,
                                 "x_m": round(start_xy[0], 5), "y_m": round(start_xy[1], 5),
                                 "energy_j": 0.0, "command_vx_m_s": 0.0,
                                 "command_vy_m_s": 0.0, "command_wz_rad_s": 0.0})
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = wrapped.step(action)
                steps += 1
                info = infos[0]
                trajectories.append({
                    "episode": episode, "decision": steps,
                    "time_s": round(steps * env.step_dt, 3),
                    "x_m": round(number(info, "body_x_m"), 5),
                    "y_m": round(number(info, "body_y_m"), 5),
                    "energy_j": round(number(info, "episode_energy_j"), 3),
                    "command_vx_m_s": round(number(info, "command_vx_m_s"), 4),
                    "command_vy_m_s": round(number(info, "command_vy_m_s"), 4),
                    "command_wz_rad_s": round(number(info, "command_wz_rad_s"), 4),
                })
                if bool(dones[0]):
                    break
            success = bool(number(info, "is_success"))
            row = {
                "episode": episode, "layout": args.layout,
                "split": condition.layout.split, "geometry": condition.geometry,
                "success": int(success),
                "reason": reason(info),
                "energy_j": round(number(info, "episode_energy_j"), 2),
                "path_length_m": round(number(info, "path_length_m"), 3),
                "final_distance_m": round(number(info, "final_distance_m"), 3),
                "decisions": steps,
            }
            row.update({f"{label}_entry": int(number(info, f"{label}_entry"))
                        for label in TERRAIN_LABELS})
            row["energy_j_per_m"] = (round(row["energy_j"] / row["path_length_m"], 2)
                                     if row["path_length_m"] > 0 else float("nan"))
            rows.append(row)
            print(f"[PLAY-TEST8] {row}", flush=True)
        output = (args.output or HERE / "results" /
                  f"{args.layout}_{condition.geometry}_eval_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        track_output = output.with_name(output.stem + "_trajectory.csv")
        with track_output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(trajectories[0]))
            writer.writeheader()
            writer.writerows(trajectories)
        grid_output = output.with_name(output.stem + "_grid.csv")
        with grid_output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("row", "col", "terrain"))
            writer.writeheader()
            for row, terrain_row in enumerate(terrain_codes):
                for col, code in enumerate(terrain_row):
                    writer.writerow({"row": row, "col": col,
                                     "terrain": case.grid[row][col]})
        arrivals = [row for row in rows if row["success"]]
        print(f"[PLAY-TEST8] arrivals={len(arrivals)}/{len(rows)} "
              f"mean_energy_of_arrivals={sum(r['energy_j'] for r in arrivals) / len(arrivals):.2f}J"
              if arrivals else f"[PLAY-TEST8] arrivals=0/{len(rows)}", flush=True)
        print(f"[PLAY-TEST8] summary={output.resolve()}", flush=True)
        print(f"[PLAY-TEST8] trajectory={track_output.resolve()}", flush=True)
        print(f"[PLAY-TEST8] grid={grid_output.resolve()}", flush=True)
        metadata_output = output.with_name(output.stem + "_metadata.json")
        with metadata_output.open("x", encoding="utf-8") as stream:
            json.dump({
                "checkpoint": str(args.checkpoint.resolve()), "layout": args.layout,
                "split": condition.layout.split, "episodes": args.episodes,
                "seed": args.seed, "deterministic": True,
                "terrain_usd": str(condition.layout.usd),
                "terrain_sha256": hashlib.sha256(condition.layout.usd.read_bytes()).hexdigest(),
            }, stream, indent=2)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(skip_cleanup=args.headless)

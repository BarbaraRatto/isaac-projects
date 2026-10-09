"""Evaluate a binary-choice PPO checkpoint with live ZED X and DINO; no training."""

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--case", default=None, help="defaults to the case stored with the checkpoint")
parser.add_argument("--atlas", type=Path, default=HERE.parent / "runs" / "ice_choice_atlas.npz")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint.is_file() or not args.atlas.is_file():
    parser.error("--checkpoint and --atlas must point to existing files")
if args.episodes < 1:
    parser.error("--episodes must be positive")
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from binary_choice.config import NAV_CFG_BINARY, TERRAIN_LABELS, load_case  # noqa: E402
from binary_choice.env import BinaryChoiceEnv  # noqa: E402
from binary_choice.scene import BinaryChoiceCfg  # noqa: E402
from binary_choice.terrain import read_terrain_codes  # noqa: E402
from binary_choice.visual_features import (  # noqa: E402
    LOOK_X_M, LOOK_Y_M, RESOLUTION_M, X0_M, Y0_M,
)
from rl_config import validate_assets  # noqa: E402


def number(info, key):
    value = info[key]
    return float(value.item()) if hasattr(value, "item") else float(value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    validate_assets()
    saved_config = args.checkpoint.parent / "config.json"
    if not saved_config.is_file():
        raise RuntimeError(f"Missing training configuration: {saved_config}")
    saved = json.loads(saved_config.read_text(encoding="utf-8"))
    case_name = args.case or saved.get("case")
    case = load_case(case_name)
    visual_sampling = {"look_x_m": LOOK_X_M, "look_y_m": LOOK_Y_M,
                       "resolution_m": RESOLUTION_M, "x0_m": X0_M, "y0_m": Y0_M}
    if (saved.get("case") != case.name
            or saved.get("terrain_sha256") != sha256(case.terrain_usd)
            or saved.get("atlas_sha256") != sha256(args.atlas)
            or saved.get("scene") != json.loads(json.dumps(asdict(case.geometry)))
            or saved.get("environment") != json.loads(json.dumps(asdict(NAV_CFG_BINARY)))
            or saved.get("visual_sampling") != json.loads(json.dumps(visual_sampling))):
        raise RuntimeError("Checkpoint was trained with different case, terrain, PCA or action limits")
    terrain_codes = read_terrain_codes(case)
    geometry = case.geometry
    cfg = BinaryChoiceCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.episode_length_s = NAV_CFG_BINARY.max_episode_seconds
    cfg.robot_cfg.init_state.pos = (*case.start_xy, NAV_CFG_BINARY.spawn_height_m)
    half = geometry.start_yaw_rad / 2.0
    cfg.robot_cfg.init_state.rot = (math.cos(half), 0.0, 0.0, math.sin(half))
    cfg.viewer.eye = (8.0, 9.0, 11.0)
    cfg.viewer.lookat = (2.0, 2.0, 0.0)
    env = BinaryChoiceEnv(cfg, case, args.atlas, dino_batch_size=1)
    try:
        wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
        model = PPO.load(str(args.checkpoint), device="cpu")
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
                                 "energy_j": 0.0})
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
                })
                if bool(dones[0]):
                    break
            success = bool(number(info, "is_success"))
            row = {
                "episode": episode, "success": int(success),
                "reason": "goal" if success else
                          ("timeout" if info.get("TimeLimit.truncated") else "unsafe"),
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
            print(f"[PLAY-BINARY] {row}", flush=True)
        output = args.output or HERE / "results" / f"{case.name}_eval_{datetime.now():%Y%m%d_%H%M%S}.csv"
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
        print(f"[PLAY-BINARY] arrivals={len(arrivals)}/{len(rows)} "
              f"mean_energy_of_arrivals={sum(r['energy_j'] for r in arrivals) / len(arrivals):.2f}J"
              if arrivals else f"[PLAY-BINARY] arrivals=0/{len(rows)}", flush=True)
        print(f"[PLAY-BINARY] summary={output.resolve()}", flush=True)
        print(f"[PLAY-BINARY] trajectory={track_output.resolve()}", flush=True)
        print(f"[PLAY-BINARY] grid={grid_output.resolve()}", flush=True)
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

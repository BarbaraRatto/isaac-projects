"""Watch and measure a trained ice-choice policy, without updating weights."""

import argparse
import csv
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
sys.path.insert(0, str(PACKAGE_ROOT))
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("cached", "online"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--atlas", type=Path, default=PACKAGE_ROOT / "runs" / "ice_choice_atlas.npz")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint.is_file():
    parser.error(f"Missing checkpoint: {args.checkpoint}")
if not args.atlas.is_file():
    parser.error(f"Missing atlas: {args.atlas}")
if args.episodes < 1:
    parser.error("--episodes must be positive")
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from stable_baselines3 import PPO  # noqa: E402
from ice_choice.ice_choice_env import IceChoiceEnv  # noqa: E402
from rl_config import validate_assets  # noqa: E402


def main():
    validate_assets()
    env = IceChoiceEnv(simulation_app, args.atlas, args.mode,
                       device=args.device, render=not args.headless)
    try:
        model = PPO.load(str(args.checkpoint), device="cpu")
        if model.observation_space.shape != env.observation_space.shape:
            raise RuntimeError("Checkpoint observation shape differs from this environment")
        rows = []
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=episode)
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated or truncated:
                    break
            row = {
                "episode": episode + 1, "mode": args.mode,
                "success": int(info["is_success"]),
                "reason": info["termination_reason"],
                "route": info["route"],
                "energy_j": round(info["episode_energy_j"], 2),
                "distance_to_goal_m": round(info["final_distance_m"], 2),
                "path_length_m": round(info["path_length_m"], 2),
                "steps": steps,
                "mean_visual_valid": round(info["mean_visual_valid"], 2),
            }
            rows.append(row)
            print(f"[PLAY-ICE] {row}", flush=True)
        output = args.output or PACKAGE_ROOT / "results" / f"ice_choice_{args.mode}_eval.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[PLAY-ICE] saved={output.resolve()}", flush=True)
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

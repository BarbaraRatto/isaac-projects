"""Evaluate the cached-feature PPO checkpoint using live ZED X and DINOv2."""

import argparse
import csv
import math
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--mode", choices=("cached", "cached_visible", "online"), default="online")
parser.add_argument("--atlas", type=Path,
                    default=HERE / "runs" / "first_choice_feature_atlas.npz")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--output", type=Path, default=HERE / "results" / "first_choice_online_eval.csv")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1:
    parser.error("--episodes must be positive")
if args.checkpoint is not None and not args.checkpoint.is_file():
    parser.error(f"Checkpoint missing: {args.checkpoint}")
args.enable_cameras = args.mode != "cached"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from first_choice_env import FirstChoiceEnv  # noqa: E402
from rl_config import validate_assets  # noqa: E402

FIELDS = ("episode", "success", "reason", "route", "energy_j", "path_length_m",
          "goal_distance_m", "steps", "mean_valid_visual")


def main():
    validate_assets()
    env = FirstChoiceEnv(simulation_app, args.atlas, online=args.mode == "online",
                         cached_visible=args.mode == "cached_visible",
                         device=args.device, render=not args.headless)
    try:
        model = PPO.load(str(args.checkpoint), device="cpu") if args.checkpoint else None
        rows = []
        for episode in range(1, args.episodes + 1):
            obs, info = env.reset(seed=episode - 1)
            print(f"[PLAY-CHOICE] episode={episode} initial_valid_visual={info['visual_valid']} "
                  f"obs_shape={obs.shape}", flush=True)
            if model is None:
                quat = env.robot.data.root_quat_w[0]
                yaw = math.atan2(
                    2.0 * float(quat[0] * quat[3] + quat[1] * quat[2]),
                    1.0 - 2.0 * float(quat[2] ** 2 + quat[3] ** 2),
                )
                cached = env.visual.local(
                    env.robot.data.root_pos_w[0, :2].cpu().numpy(), yaw
                ).reshape(-1, 13)
                live = obs[11:].reshape(-1, 13)
                overlap = (cached[:, -1] > 0) & (live[:, -1] > 0)
                if overlap.any():
                    distances = np.linalg.norm(cached[overlap, :12] - live[overlap, :12], axis=1)
                    print(f"[PLAY-CHOICE] overlap={int(overlap.sum())}/12 "
                          f"feature_distance_mean={distances.mean():.3f}", flush=True)
                obs, reward, terminated, truncated, info = env.step(
                    np.array([0.4, 0.0, 0.0], dtype=np.float32)
                )
                print(f"[PLAY-CHOICE] check step_valid_visual={info['visual_valid']} "
                      f"reason={info['termination_reason']}", flush=True)
                break
            valid = [info["visual_valid"]]
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                valid.append(info["visual_valid"])
                steps += 1
                if terminated or truncated:
                    break
            row = {
                "episode": episode, "success": int(info["is_success"]),
                "reason": info["termination_reason"], "route": info["route"],
                "energy_j": round(info["episode_energy_j"], 2),
                "path_length_m": round(info["path_length_m"], 2),
                "goal_distance_m": round(info["final_distance_m"], 2),
                "steps": steps, "mean_valid_visual": round(float(np.mean(valid)), 2),
            }
            rows.append(row)
            print(f"[PLAY-CHOICE] {row}", flush=True)
        if rows:
            if args.mode != "online" and args.output == HERE / "results" / "first_choice_online_eval.csv":
                args.output = HERE / "results" / f"first_choice_{args.mode}_eval.csv"
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"[PLAY-CHOICE] CSV={args.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        if args.headless:
            os._exit(1)
        simulation_app.close()
        raise
    else:
        simulation_app.close(skip_cleanup=args.headless)

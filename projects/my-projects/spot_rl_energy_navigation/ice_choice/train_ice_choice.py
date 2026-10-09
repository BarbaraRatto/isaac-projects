"""Train one independent PPO policy: --mode cached or --mode online.

The two modes share every task, camera and PPO parameter. Only the DINO
feature source changes. This script never resumes another mode's checkpoint.
"""

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
sys.path.insert(0, str(PACKAGE_ROOT))
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("cached", "online"), required=True)
parser.add_argument("--atlas", type=Path, default=PACKAGE_ROOT / "runs" / "ice_choice_atlas.npz")
parser.add_argument("--timesteps", type=int, default=102_400)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.timesteps < 256 or args.timesteps % 256:
    parser.error("--timesteps must be a positive multiple of 256")
if not args.atlas.is_file():
    parser.error(f"Missing feature atlas: {args.atlas}; run build_ice_choice_atlas.py first")
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from ice_choice.ice_choice_config import ICE_GEOMETRY, ICE_NAV_CFG  # noqa: E402
from ice_choice.ice_choice_env import IceChoiceEnv  # noqa: E402
from ice_visual_features import LOCAL_CELLS, PCA_DIM  # noqa: E402
from rl_config import TERRAIN_USD, validate_assets  # noqa: E402


class TimingCallback(BaseCallback):
    """Record actual training throughput and visual-processing time."""

    def __init__(self, output: Path, interval: int = 2048):
        super().__init__()
        self.output = output
        self.interval = interval
        self.started = 0.0
        self.previous = 0.0
        self.visual_sum_ms = 0.0
        self.dino_sum_ms = 0.0
        self.count = 0
        self.handle = None
        self.writer = None

    def _on_training_start(self) -> None:
        self.started = self.previous = time.perf_counter()
        self.handle = self.output.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=(
            "timesteps", "wall_seconds", "interval_wall_seconds", "actions_per_second",
            "mean_visual_ms", "mean_dino_ms",
        ))
        self.writer.writeheader()

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self.visual_sum_ms += float(info.get("visual_ms", 0.0))
        self.dino_sum_ms += float(info.get("dino_ms", 0.0))
        self.count += 1
        if self.num_timesteps % self.interval == 0:
            now = time.perf_counter()
            elapsed = now - self.previous
            self.writer.writerow({
                "timesteps": self.num_timesteps,
                "wall_seconds": round(now - self.started, 3),
                "interval_wall_seconds": round(elapsed, 3),
                "actions_per_second": round(self.count / elapsed, 3),
                "mean_visual_ms": round(self.visual_sum_ms / self.count, 3),
                "mean_dino_ms": round(self.dino_sum_ms / self.count, 3),
            })
            self.handle.flush()
            self.previous = now
            self.visual_sum_ms = self.dino_sum_ms = 0.0
            self.count = 0
        return True

    def _on_training_end(self) -> None:
        if self.handle is not None:
            self.handle.close()


def main():
    validate_assets()
    run_dir = PACKAGE_ROOT / "runs" / datetime.now().strftime(f"ice_choice_{args.mode}_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    settings = {
        "mode": args.mode, "seed": args.seed, "timesteps": args.timesteps,
        "terrain_usd": str(TERRAIN_USD), "feature_atlas": str(args.atlas.resolve()),
        "scene": asdict(ICE_GEOMETRY), "environment": asdict(ICE_NAV_CFG),
        "headless": args.headless, "camera": "ZED X CameraRight optics, RGB-D 640x360",
        "dino": "facebook/dinov2-small, 518x518, frozen",
        "visual_components": PCA_DIM, "local_visual_cells": LOCAL_CELLS,
        "ppo": {"n_steps": 256, "batch_size": 128, "n_epochs": 5,
                "learning_rate": 3e-4, "gamma": 0.995, "net_arch": [128, 128]},
    }
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    env = IceChoiceEnv(simulation_app, args.atlas, args.mode,
                       device=args.device, render=not args.headless)
    try:
        monitored = Monitor(
            env, filename=str(run_dir / "episodes.monitor.csv"),
            info_keywords=("is_success", "episode_energy_j", "final_distance_m",
                           "route", "ice_contact", "path_length_m", "mean_visual_valid"),
        )
        model = PPO(
            "MlpPolicy", monitored, seed=args.seed, device="cpu",
            n_steps=256, batch_size=128, n_epochs=5, learning_rate=3e-4,
            gamma=0.995, policy_kwargs={"net_arch": [128, 128]}, verbose=1,
        )
        callback = CallbackList([
            TimingCallback(run_dir / "timing.csv"),
            CheckpointCallback(save_freq=5120, save_path=str(run_dir),
                               name_prefix="policy_step"),
        ])
        started = time.perf_counter()
        model.learn(total_timesteps=args.timesteps, callback=callback)
        training_seconds = time.perf_counter() - started
        model.save(str(run_dir / "policy_final"))
        (run_dir / "summary.json").write_text(json.dumps({
            "training_wall_seconds": training_seconds,
            "dino_model_load_seconds": env.dino_load_seconds,
            "checkpoint": str(run_dir / "policy_final.zip"),
        }, indent=2), encoding="utf-8")
        print(f"[TRAIN-ICE] mode={args.mode} checkpoint={run_dir / 'policy_final.zip'} "
              f"wall_seconds={training_seconds:.1f} ", flush=True)
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

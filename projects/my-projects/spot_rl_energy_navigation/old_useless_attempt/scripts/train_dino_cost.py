"""Compare DINO online inference and cached feature reads inside real PPO training.

Both modes mount/render the same ZED X camera and train the same 11-input
navigation policy. The extracted/read feature is deliberately not given to PPO:
this experiment measures training cost only, not perception-driven navigation.
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ZED_USD = Path("/home/isaac/isaac-sim/assets/Assets/Isaac/5.1/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc")
DEFAULT_CACHE = HERE / "runs" / "zed_feature_samples.npy"

parser = argparse.ArgumentParser(description="Train PPO while processing ZED X features.")
parser.add_argument("--mode", choices=("online", "cached"), required=True)
parser.add_argument("--timesteps", type=int, default=8192)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.timesteps < 128 or args.timesteps % 128:
    parser.error("--timesteps must be a positive multiple of 128")
if args.mode == "cached" and not args.cache.is_file():
    parser.error(f"Feature cache missing: {args.cache}")
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from navigation_config import NAV_CFG  # noqa: E402
from navigation_env import SpotNavigationEnv  # noqa: E402
from rl_config import validate_assets  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "IsaacRobotics" / "applications"))


def create_zed() -> Camera:
    if not ZED_USD.is_file():
        raise FileNotFoundError(ZED_USD)
    from spot_camera import attach_zed_camera

    stage = omni.usd.get_context().get_stage()
    path = attach_zed_camera(stage, str(ZED_USD.parents[4]))
    return Camera(cfg=CameraCfg(
        prim_path=path, update_period=0.0, height=360, width=640,
        data_types=["rgb"], spawn=None,
    ))


class FeatureTimingWrapper(gym.Wrapper):
    """Process one ZED image per RL action while preserving PPO observations."""

    def __init__(self, env: SpotNavigationEnv, mode: str, cache_path: Path):
        super().__init__(env)
        self.mode = mode
        self.timings = []
        self.camera_timings = []
        self.feature_checksum = 0.0
        self.count = 0
        self.load_seconds = 0.0
        self.cache = None
        self.dino = None
        if mode == "cached":
            self.cache = np.load(cache_path, mmap_mode="r")
            if self.cache.ndim != 4 or self.cache.shape[1:] != (37, 37, 384):
                raise RuntimeError(f"Unexpected cached feature shape: {self.cache.shape}")
        else:
            package_root = PROJECT_ROOT / "ros2_ws" / "src" / "spot_terrain_gridmap"
            sys.path.insert(0, str(package_root))
            from spot_terrain_gridmap.dino_feature_extractor import DinoFeatureExtractor
            start = time.perf_counter()
            self.dino = DinoFeatureExtractor(
                model_name="facebook/dinov2-small", device="cuda", input_size=518
            )
            self.load_seconds = time.perf_counter() - start

    def process(self, *, record: bool):
        start = time.perf_counter()
        rgb = self.env.sensor_camera.data.output["rgb"]
        if rgb.shape != (1, 360, 640, 3):
            raise RuntimeError(f"Unexpected ZED X RGB shape: {rgb.shape}")
        frame = rgb[0].cpu().numpy().copy()
        camera_seconds = time.perf_counter() - start
        start = time.perf_counter()
        if self.mode == "online":
            features = self.dino.extract(frame)
        else:
            # Feature contents are not used by PPO in this cost experiment.
            features = np.array(self.cache[self.count % len(self.cache)], copy=True)
        feature_seconds = time.perf_counter() - start
        if features.shape != (37, 37, 384):
            raise RuntimeError(f"Unexpected DINO feature shape: {features.shape}")
        if record:
            self.camera_timings.append(camera_seconds)
            self.timings.append(feature_seconds)
            self.feature_checksum += float(features[0, 0, 0])
            self.count += 1

    def step(self, action):
        result = self.env.step(action)
        self.process(record=True)
        return result


def main():
    validate_assets()
    run_dir = HERE / "runs" / datetime.now().strftime(f"dino_cost_{args.mode}_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "mode": args.mode, "timesteps": args.timesteps, "seed": args.seed,
        "cache": str(args.cache) if args.mode == "cached" else None,
        "environment": asdict(NAV_CFG), "camera": "ZED_X/CameraRight 640x360",
        "ppo_observation": "11 robot/goal values; DINO features not used by policy",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[TRAIN-DINO] mode={args.mode} run={run_dir}", flush=True)
    env = SpotNavigationEnv(
        simulation_app, device=args.device, render=not args.headless,
        camera_factory=create_zed,
    )
    feature_env = FeatureTimingWrapper(env, args.mode, args.cache)
    try:
        # Warm up camera and CUDA inference outside the measured PPO run.
        feature_env.reset(seed=args.seed)
        for _ in range(2):
            feature_env.env.step(np.zeros(3, dtype=np.float32))
            feature_env.process(record=False)
        monitored = Monitor(
            feature_env, filename=str(run_dir / "episodes.monitor.csv"),
            info_keywords=("is_success", "episode_energy_j", "final_distance_m"),
        )
        model = PPO(
            "MlpPolicy", monitored, seed=args.seed, device="cpu",
            n_steps=128, batch_size=64, n_epochs=5,
            learning_rate=3e-4, gamma=0.99, verbose=1,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model.learn(total_timesteps=args.timesteps)
        training_seconds = time.perf_counter() - started
        model.save(str(run_dir / "policy_final"))
        result = {
            **config,
            "actual_timesteps": model.num_timesteps,
            "training_wall_seconds": training_seconds,
            "transitions_per_wall_second": model.num_timesteps / training_seconds,
            "model_load_seconds": feature_env.load_seconds,
            "camera_mean_ms": 1000 * statistics.mean(feature_env.camera_timings),
            "camera_p95_ms": 1000 * float(np.percentile(feature_env.camera_timings, 95)),
            "feature_mean_ms": 1000 * statistics.mean(feature_env.timings),
            "feature_p95_ms": 1000 * float(np.percentile(feature_env.timings, 95)),
            "feature_calls": len(feature_env.timings),
            "feature_checksum": feature_env.feature_checksum,
            "gpu_peak_allocated_mb": (torch.cuda.max_memory_allocated() / 2**20
                                      if torch.cuda.is_available() else None),
        }
        (run_dir / "timing.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[TRAIN-DINO] results={json.dumps(result)}", flush=True)
    finally:
        feature_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=args.headless)

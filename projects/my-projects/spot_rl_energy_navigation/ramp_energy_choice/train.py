"""Train 32 Spot robots to choose between a ramp and an asphalt detour.

The --probe-steps mode runs no training and saves no policy. The main run can
be interrupted with Ctrl+C: the current policy and timing data are saved.
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
sys.path.append(str(HERE.parent))
parser = argparse.ArgumentParser()
parser.add_argument("--atlas", type=Path, default=HERE.parent / "runs" / "ice_choice_atlas.npz")
parser.add_argument("--timesteps", type=int, default=102_400,
                    help="total policy transitions across all robots")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num-envs", type=int, choices=(4, 8, 16, 32), default=32,
                    help="number of isolated Spot scenes; default is 32")
parser.add_argument("--dino-batch-size", type=int, default=4,
                    help="maximum images per DINO forward pass; default 4 limits GPU memory")
parser.add_argument("--probe-steps", type=int, default=0,
                    help="no training; check this many joint simulator actions")
parser.add_argument("--resume", type=Path, default=None,
                    help="resume a checkpoint with the same --num-envs; --timesteps is total target")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.atlas.is_file():
    parser.error(f"Missing atlas {args.atlas}; build_ice_choice_atlas.py first")
if args.timesteps < 256 or args.timesteps % 256:
    parser.error("--timesteps must be a positive multiple of 256")
if args.probe_steps < 0:
    parser.error("--probe-steps cannot be negative")
if not 1 <= args.dino_batch_size <= args.num_envs:
    parser.error("--dino-batch-size must be between 1 and --num-envs")
if args.resume and not args.resume.is_file():
    parser.error(f"Missing checkpoint: {args.resume}")
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402

from ramp_energy_choice.config import GEOMETRY, NAV_CFG_RAMP  # noqa: E402
from ice_visual_features import LOCAL_CELLS, PCA_DIM, VISUAL_SIZE  # noqa: E402
from ramp_energy_choice.env import RampChoiceEnv  # noqa: E402
from ramp_energy_choice.scene import RampChoiceCfg  # noqa: E402
from rl_config import validate_assets  # noqa: E402
from zed_sensor import ZED_USD
from ramp_energy_choice.build_scene import build as build_scene, OUTPUT as SCENE_USD  # noqa: E402


class TrainingLog(BaseCallback):
    def __init__(self, run_dir: Path, target: int, initial_steps: int = 0):
        super().__init__()
        self.run_dir = run_dir
        self.target = target
        self.initial_steps = initial_steps
        self.previous_steps = initial_steps
        self.previous_time = 0.0
        self.start_time = 0.0
        self.visual_ms = 0.0
        self.dino_ms = 0.0
        self.count = 0
        self.episodes = 0
        self.successes = 0

    def _on_training_start(self) -> None:
        self.start_time = self.previous_time = time.perf_counter()
        self.timing_handle = (self.run_dir / "timing.csv").open("w", newline="", encoding="utf-8")
        self.timing = csv.writer(self.timing_handle)
        self.timing.writerow(("timesteps", "wall_seconds", "interval_seconds",
                              "transitions_per_second", "mean_dino_ms_per_robot",
                              "mean_visual_ms_per_robot", "eta_seconds"))
        self.timing_handle.flush()
        self.episode_handle = (self.run_dir / "episodes.csv").open("w", newline="", encoding="utf-8")
        self.episode_writer = csv.writer(self.episode_handle)
        self.episode_writer.writerow(("timesteps", "env_id", "reward", "actions", "success",
                                      "energy_j", "path_length_m", "ramp_entry", "asphalt_entry", "final_distance_m"))

    @staticmethod
    def value(info, key):
        value = info[key]
        return float(value.item()) if hasattr(value, "item") else float(value)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for env_id, info in enumerate(infos):
            self.visual_ms += self.value(info, "visual_ms")
            self.dino_ms += self.value(info, "dino_ms")
            self.count += 1
            episode = info.get("episode")
            if episode:
                success = self.value(info, "is_success")
                self.episodes += 1
                self.successes += int(success)
                self.episode_writer.writerow((
                    self.num_timesteps, env_id, episode["r"], episode["l"], int(success),
                    self.value(info, "episode_energy_j"), self.value(info, "path_length_m"),
                    int(self.value(info, "ramp_entry")),
                    int(self.value(info, "asphalt_entry")),
                    self.value(info, "final_distance_m"),
                ))
                self.episode_handle.flush()
        if self.num_timesteps - self.previous_steps >= 1024:
            now = time.perf_counter()
            elapsed = now - self.previous_time
            transitions = self.num_timesteps - self.previous_steps
            rate = transitions / elapsed
            remaining = max(0, self.target - self.num_timesteps) / rate
            row = (self.num_timesteps, round(now - self.start_time, 2), round(elapsed, 2),
                   round(rate, 3), round(self.dino_ms / self.count, 2),
                   round(self.visual_ms / self.count, 2), round(remaining))
            self.timing.writerow(row)
            self.timing_handle.flush()
            print(f"[PARALLEL] transitions={row[0]}/{self.target} rate={row[3]:.2f}/s "
                  f"DINO={row[4]:.1f}ms/robot ETA={remaining / 60:.1f}min "
                  f"success={self.successes}/{self.episodes}", flush=True)
            self.previous_steps = self.num_timesteps
            self.previous_time = now
            self.visual_ms = self.dino_ms = 0.0
            self.count = 0
        return True

    def _on_training_end(self) -> None:
        self.close_files()

    def close_files(self) -> None:
        for name in ("timing_handle", "episode_handle"):
            handle = getattr(self, name, None)
            if handle is not None and not handle.closed:
                handle.close()


def probe(env: RampChoiceEnv, steps: int) -> None:
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    for i in range(env.num_envs):
        terrain = stage.GetPrimAtPath(f"/World/envs/env_{i}/Warehouse")
        robot = stage.GetPrimAtPath(f"/World/envs/env_{i}/Spot")
        camera = stage.GetPrimAtPath(f"/World/envs/env_{i}/NavigationZEDCamera")
        if not all((terrain.IsValid(), robot.IsValid(), camera.IsValid())):
            raise RuntimeError(f"Incomplete scene for env_{i}")
    wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
    observation = wrapped.reset()
    print(f"[PROBE] envs={wrapped.num_envs} observation={observation.shape} "
          f"camera_rgb={tuple(env.camera.data.output['rgb'].shape)} "
          f"camera_depth={tuple(env.camera.data.output['distance_to_image_plane'].shape)} "
          f"valid_visual_cells={env.visual_valid.tolist()}", flush=True)
    if observation.shape != (env.num_envs, 11 + VISUAL_SIZE) or not np.isfinite(observation).all():
        raise RuntimeError("Invalid batched DINO observation")
    if not (env.visual_valid > 0).all():
        raise RuntimeError(f"At least one camera sees no valid terrain cells: {env.visual_valid.tolist()}")
    started = time.perf_counter()
    for i in range(steps):
        obs, rewards, dones, infos = wrapped.step(np.zeros((env.num_envs, 3), dtype=np.float32))
        print(f"[PROBE] action={i + 1}/{steps} reward={rewards.round(3).tolist()} "
              f"done={dones.tolist()} visual_valid={env.visual_valid.tolist()}", flush=True)
        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite observation during probe")
    elapsed = time.perf_counter() - started
    print(f"[PROBE] completed={steps} vector_actions, {steps * env.num_envs} robot_transitions "
          f"in {elapsed:.1f}s; this was not a training run", flush=True)


def main():
    validate_assets()
    if not ZED_USD.is_file():
        raise FileNotFoundError(ZED_USD)
    build_scene()
    cfg = RampChoiceCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    run_dir = None
    if args.probe_steps == 0:
        run_dir = HERE / "runs" / datetime.now().strftime(f"ramp_energy_online_{args.num_envs}env_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
        settings = {
            "mode": "online", "num_envs": args.num_envs, "seed": args.seed,
            "target_transitions": args.timesteps, "terrain_usd": str(SCENE_USD),
            "feature_atlas_for_pca_only": str(args.atlas.resolve()),
            "scene": asdict(GEOMETRY), "environment": asdict(NAV_CFG_RAMP),
            "headless": args.headless,
            "camera": f"{args.num_envs} ZED X CameraRight RGB-D 640x360, 15deg pitch",
            "dino": ("one shared frozen facebook/dinov2-small, "
                     f"microbatch={args.dino_batch_size}, input=518"),
            "visual_components": PCA_DIM, "local_visual_cells": LOCAL_CELLS,
            "ppo": {"n_steps_per_env": 256 // args.num_envs, "batch_size": 128, "n_epochs": 5,
                    "learning_rate": 3e-4, "gamma": 0.995, "net_arch": [128, 128]},
            "resume": str(args.resume) if args.resume else None,
        }
        (run_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        print(f"[PARALLEL] run_dir={run_dir}", flush=True)
    env = RampChoiceEnv(cfg, args.atlas, dino_batch_size=args.dino_batch_size)
    try:
        if args.probe_steps:
            probe(env, args.probe_steps)
            return
        wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
        if args.resume:
            model = PPO.load(str(args.resume), device="cpu")
            if model.n_envs != args.num_envs or model.n_steps != 256 // args.num_envs:
                raise RuntimeError("Resume checkpoint has a different number of environments or rollout steps")
            model.set_env(wrapped)
            remaining_steps = args.timesteps - model.num_timesteps
            if remaining_steps <= 0:
                raise ValueError("--timesteps must exceed saved model.num_timesteps")
        else:
            model = PPO(
                "MlpPolicy", wrapped, seed=args.seed, device="cpu", n_steps=256 // args.num_envs,
                batch_size=128, n_epochs=5, learning_rate=3e-4, gamma=0.995,
                policy_kwargs={"net_arch": [128, 128]}, verbose=1,
            )
            remaining_steps = args.timesteps
        logger = TrainingLog(run_dir, args.timesteps, initial_steps=model.num_timesteps)
        callbacks = CallbackList([
            logger,
            CheckpointCallback(save_freq=5120 // args.num_envs,
                               save_path=str(run_dir), name_prefix="policy_step"),
        ])
        started = time.perf_counter()
        interrupted = False
        try:
            model.learn(total_timesteps=remaining_steps, callback=callbacks,
                        reset_num_timesteps=not bool(args.resume))
        except KeyboardInterrupt:
            interrupted = True
            print("[PARALLEL] Ctrl+C received; saving current policy", flush=True)
        logger.close_files()
        wall = time.perf_counter() - started
        output = run_dir / (f"policy_interrupted_{model.num_timesteps}_steps" if interrupted else "policy_final")
        model.save(str(output))
        (run_dir / "summary.json").write_text(json.dumps({
            "interrupted": interrupted, "num_timesteps": model.num_timesteps,
            "wall_seconds": wall, "dino_load_seconds": env.vision.load_seconds,
            "checkpoint": str(output) + ".zip", "episodes": logger.episodes,
            "successes": logger.successes,
        }, indent=2), encoding="utf-8")
        print(f"[PARALLEL] checkpoint={output}.zip transitions={model.num_timesteps} "
              f"wall={wall:.1f}s", flush=True)
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

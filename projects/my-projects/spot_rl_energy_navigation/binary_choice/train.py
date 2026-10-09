"""Train one shared Spot policy on a named 3x3 binary-choice case.

The probe performs a few simulator steps but does not train or save a policy.
The user launches all probes and training runs; this module is never run on import.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", default="asphalt_rocks", help="module name under binary_choice/cases")
parser.add_argument("--atlas", type=Path, default=HERE.parent / "runs" / "ice_choice_atlas.npz",
                    help="old atlas supplies only PCA axes, never cached observations")
parser.add_argument("--timesteps", type=int, default=204_800,
                    help="total transitions gathered across all Spot environments")
parser.add_argument("--num-envs", type=int, choices=(8, 16, 32), default=32)
parser.add_argument("--dino-batch-size", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--probe-steps", type=int, default=0,
                    help="technical simulator check only; 0 runs PPO training")
parser.add_argument("--resume", type=Path, default=None,
                    help="checkpoint to continue; --timesteps is the new total target")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.atlas.is_file():
    parser.error(f"Missing PCA basis: {args.atlas}")
if args.timesteps < args.num_envs * 32 or args.timesteps % (args.num_envs * 32):
    parser.error("--timesteps must be a positive multiple of --num-envs * 32")
if args.probe_steps < 0:
    parser.error("--probe-steps cannot be negative")
if args.dino_batch_size is None:
    args.dino_batch_size = min(4, args.num_envs)
if not 1 <= args.dino_batch_size <= args.num_envs:
    parser.error("--dino-batch-size must be between 1 and --num-envs")
if args.resume is not None and not args.resume.is_file():
    parser.error(f"Missing checkpoint: {args.resume}")
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback, CallbackList, CheckpointCallback,
)

from binary_choice.config import (  # noqa: E402
    NAV_CFG_BINARY, PPO_CFG, TERRAIN_LABELS, load_case,
)
from binary_choice.env import BinaryChoiceEnv  # noqa: E402
from binary_choice.scene import BinaryChoiceCfg  # noqa: E402
from binary_choice.terrain import read_terrain_codes  # noqa: E402
from binary_choice.visual_features import (  # noqa: E402
    LOCAL_CELLS, PCA_DIM, VISUAL_SIZE, LOOK_X_M, LOOK_Y_M, RESOLUTION_M, X0_M, Y0_M,
)
from rl_config import validate_assets  # noqa: E402
from zed_sensor import ZED_USD  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrainingLog(BaseCallback):
    def __init__(self, run_dir: Path, target: int, initial_steps: int = 0):
        super().__init__()
        self.run_dir = run_dir
        self.target = target
        self.previous_steps = initial_steps
        self.start_time = self.previous_time = 0.0
        self.visual_ms = self.dino_ms = 0.0
        self.count = self.episodes = self.successes = 0

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
                                      "reason", "energy_j", "path_length_m", "final_distance_m",
                                      *(f"{label}_entry" for label in TERRAIN_LABELS)))

    @staticmethod
    def value(info, key):
        value = info[key]
        return float(value.item()) if hasattr(value, "item") else float(value)

    def _on_step(self) -> bool:
        for env_id, info in enumerate(self.locals["infos"]):
            self.visual_ms += self.value(info, "visual_ms")
            self.dino_ms += self.value(info, "dino_ms")
            self.count += 1
            episode = info.get("episode")
            if episode:
                success = self.value(info, "is_success")
                self.episodes += 1
                self.successes += int(success)
                reason = ("goal" if success else
                          "timeout" if info.get("TimeLimit.truncated") else "unsafe")
                self.episode_writer.writerow((
                    self.num_timesteps, env_id, episode["r"], episode["l"], int(success), reason,
                    self.value(info, "episode_energy_j"), self.value(info, "path_length_m"),
                    self.value(info, "final_distance_m"),
                    *(int(self.value(info, f"{label}_entry")) for label in TERRAIN_LABELS),
                ))
                self.episode_handle.flush()
        if self.num_timesteps - self.previous_steps >= 1024:
            now = time.perf_counter()
            elapsed = now - self.previous_time
            rate = (self.num_timesteps - self.previous_steps) / elapsed
            remaining = max(0, self.target - self.num_timesteps) / rate
            row = (self.num_timesteps, round(now - self.start_time, 2), round(elapsed, 2),
                   round(rate, 3), round(self.dino_ms / self.count, 2),
                   round(self.visual_ms / self.count, 2), round(remaining))
            self.timing.writerow(row)
            self.timing_handle.flush()
            print(f"[BINARY] transitions={row[0]}/{self.target} rate={row[3]:.2f}/s "
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


def probe(env: BinaryChoiceEnv, steps: int) -> None:
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    case = env.case
    for i in range(env.num_envs):
        base = f"/World/envs/env_{i}"
        paths = (
            f"{base}/Warehouse", f"{base}/Spot", f"{base}/NavigationZEDCamera",
            f"{base}/Warehouse/Grid/Cell_0_0_{case.grid[0][0]}",
            f"{base}/Warehouse/Grid/Cell_2_2_{case.grid[2][2]}",
        )
        if not all(stage.GetPrimAtPath(path).IsValid() for path in paths):
            raise RuntimeError(f"Incomplete binary-choice scene in env_{i}: {paths}")
    wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
    observation = wrapped.reset()
    print(f"[PROBE] envs={wrapped.num_envs} observation={observation.shape} "
          f"camera_rgb={tuple(env.camera.data.output['rgb'].shape)} "
          f"camera_depth={tuple(env.camera.data.output['distance_to_image_plane'].shape)} "
          f"valid_visual_cells={env.visual_valid.tolist()}", flush=True)
    if observation.shape != (env.num_envs, 11 + VISUAL_SIZE) or not np.isfinite(observation).all():
        raise RuntimeError("Invalid binary-choice DINO observation")
    if not bool((env.visual_valid > 0).all()):
        raise RuntimeError("At least one Spot has no valid visible terrain sample")
    started = time.perf_counter()
    for i in range(steps):
        obs, rewards, dones, _ = wrapped.step(np.zeros((env.num_envs, 3), dtype=np.float32))
        print(f"[PROBE] action={i + 1}/{steps} reward={rewards.round(3).tolist()} "
              f"done={dones.tolist()} visual_valid={env.visual_valid.tolist()}", flush=True)
        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite observation during probe")
    elapsed = time.perf_counter() - started
    print(f"[PROBE] completed={steps} vector_actions, {steps * env.num_envs} "
          f"robot_transitions in {elapsed:.1f}s; no training was run", flush=True)


def main() -> None:
    validate_assets()
    if not ZED_USD.is_file():
        raise FileNotFoundError(ZED_USD)
    case = load_case(args.case)
    read_terrain_codes(case)
    geometry = case.geometry
    cfg = BinaryChoiceCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = NAV_CFG_BINARY.max_episode_seconds
    cfg.robot_cfg.init_state.pos = (*case.start_xy, NAV_CFG_BINARY.spawn_height_m)
    half = geometry.start_yaw_rad / 2.0
    cfg.robot_cfg.init_state.rot = (math.cos(half), 0.0, 0.0, math.sin(half))
    cfg.viewer.eye = (8.0, 9.0, 11.0)
    cfg.viewer.lookat = (2.0, 2.0, 0.0)
    run_dir = None
    if args.probe_steps == 0:
        run_dir = HERE / "runs" / datetime.now().strftime(
            f"{case.name}_online_{args.num_envs}env_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
        settings = {
            "mode": "online", "case": case.name, "num_envs": args.num_envs,
            "seed": args.seed, "target_transitions": args.timesteps,
            "terrain_usd": str(case.terrain_usd.resolve()),
            "terrain_sha256": sha256(case.terrain_usd),
            "feature_atlas_for_pca_only": str(args.atlas.resolve()),
            "atlas_sha256": sha256(args.atlas),
            "scene": json.loads(json.dumps(asdict(geometry))),
            "environment": asdict(NAV_CFG_BINARY),
            "headless": args.headless,
            "camera": f"{args.num_envs} ZED X CameraRight RGB-D 640x360, 15deg pitch",
            "dino": ("one shared frozen facebook/dinov2-small, "
                     f"microbatch={args.dino_batch_size}, input=518"),
            "visual_components": PCA_DIM, "local_visual_cells": LOCAL_CELLS,
            "visual_sampling": {"look_x_m": LOOK_X_M, "look_y_m": LOOK_Y_M,
                                "resolution_m": RESOLUTION_M, "x0_m": X0_M, "y0_m": Y0_M},
            "ppo": {"n_steps_per_env": PPO_CFG.n_steps_per_env,
                    "batch_size": PPO_CFG.batch_size, "n_epochs": PPO_CFG.n_epochs,
                    "learning_rate": PPO_CFG.learning_rate, "gamma": PPO_CFG.gamma,
                    "net_arch": list(PPO_CFG.net_arch)},
            "resume": str(args.resume) if args.resume else None,
        }
        (run_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        print(f"[BINARY] run_dir={run_dir}", flush=True)
    env = BinaryChoiceEnv(cfg, case, args.atlas, dino_batch_size=args.dino_batch_size)
    try:
        if args.probe_steps:
            probe(env, args.probe_steps)
            return
        wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
        if args.resume:
            old_config = args.resume.parent / "config.json"
            if not old_config.is_file():
                raise RuntimeError(f"Missing resume configuration: {old_config}")
            saved = json.loads(old_config.read_text(encoding="utf-8"))
            for key in ("case", "terrain_sha256", "atlas_sha256", "scene", "environment",
                        "ppo", "visual_sampling"):
                if saved.get(key) != json.loads(json.dumps(settings[key])):
                    raise RuntimeError(f"Cannot resume: {key} differs from checkpoint configuration")
            if saved.get("num_envs") != args.num_envs:
                raise RuntimeError("Cannot resume with a different --num-envs")
            model = PPO.load(str(args.resume), device="cpu")
            if model.n_envs != args.num_envs or model.n_steps != PPO_CFG.n_steps_per_env:
                raise RuntimeError("Checkpoint uses different vectorization or PPO rollout steps")
            model.set_env(wrapped)
            remaining_steps = args.timesteps - model.num_timesteps
            if remaining_steps <= 0:
                raise ValueError("--timesteps must exceed saved model.num_timesteps")
        else:
            model = PPO(
                "MlpPolicy", wrapped, seed=args.seed, device="cpu",
                n_steps=PPO_CFG.n_steps_per_env, batch_size=PPO_CFG.batch_size,
                n_epochs=PPO_CFG.n_epochs, learning_rate=PPO_CFG.learning_rate,
                gamma=PPO_CFG.gamma,
                policy_kwargs={"net_arch": list(PPO_CFG.net_arch)}, verbose=1,
            )
            remaining_steps = args.timesteps
        logger = TrainingLog(run_dir, args.timesteps, initial_steps=model.num_timesteps)
        callbacks = CallbackList([
            logger,
            CheckpointCallback(save_freq=max(1, 10240 // args.num_envs),
                               save_path=str(run_dir), name_prefix="policy_step"),
        ])
        started = time.perf_counter()
        interrupted = False
        try:
            model.learn(total_timesteps=remaining_steps, callback=callbacks,
                        reset_num_timesteps=not bool(args.resume))
        except KeyboardInterrupt:
            interrupted = True
            print("[BINARY] Ctrl+C received; saving current policy", flush=True)
        logger.close_files()
        wall = time.perf_counter() - started
        output = run_dir / (f"policy_interrupted_{model.num_timesteps}_steps"
                            if interrupted else "policy_final")
        model.save(str(output))
        (run_dir / "summary.json").write_text(json.dumps({
            "interrupted": interrupted, "num_timesteps": model.num_timesteps,
            "wall_seconds": wall, "dino_load_seconds": env.vision.load_seconds,
            "checkpoint": str(output) + ".zip", "episodes": logger.episodes,
            "successes": logger.successes,
        }, indent=2), encoding="utf-8")
        print(f"[BINARY] checkpoint={output}.zip transitions={model.num_timesteps} "
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

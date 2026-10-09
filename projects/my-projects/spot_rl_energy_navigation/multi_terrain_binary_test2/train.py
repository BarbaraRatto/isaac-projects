"""Continue the successful binary policy on two new terrain pairs.

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
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--atlas", type=Path, default=HERE.parent / "runs" / "ice_choice_atlas.npz",
                    help="old atlas supplies only PCA axes, never cached observations")
parser.add_argument("--timesteps", type=int, default=204_800,
                    help="additional transitions across all Spot environments")
parser.add_argument("--source-checkpoint", type=Path, default=(
    HERE.parent / "binary_choice" / "runs" /
    "asphalt_rocks_online_32env_20261007_220136" / "policy_final.zip"),
    help="successful binary-choice checkpoint from which to continue")
parser.add_argument("--dino-batch-size", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--probe-steps", type=int, default=0,
                    help="technical simulator check only; 0 runs PPO training")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.atlas.is_file():
    parser.error(f"Missing PCA basis: {args.atlas}")
NUM_ENVS = 32
if args.timesteps < NUM_ENVS * 32 or args.timesteps % (NUM_ENVS * 32):
    parser.error("--timesteps must be a positive multiple of 1024 (32 envs * 32 steps)")
if args.probe_steps < 0:
    parser.error("--probe-steps cannot be negative")
if args.dino_batch_size is None:
    args.dino_batch_size = 4
if not 1 <= args.dino_batch_size <= NUM_ENVS:
    parser.error("--dino-batch-size must be between 1 and 32")
if not args.source_checkpoint.is_file():
    parser.error(f"Missing source checkpoint: {args.source_checkpoint}")
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback, CallbackList, CheckpointCallback,
)

from binary_choice.config import TERRAIN_LABELS  # noqa: E402
from binary_choice.terrain import read_terrain_codes  # noqa: E402
from binary_choice.visual_features import (  # noqa: E402
    LOCAL_CELLS, PCA_DIM, VISUAL_SIZE, LOOK_X_M, LOOK_Y_M, RESOLUTION_M, X0_M, Y0_M,
)
from rl_config import validate_assets  # noqa: E402
from zed_sensor import ZED_USD  # noqa: E402
from binary_choice.config import NAV_CFG_BINARY, PPO_CFG  # noqa: E402
from multi_terrain_binary_choice.env import MultiChoiceEnv  # noqa: E402
from multi_terrain_binary_test2.layouts import training_conditions  # noqa: E402
from multi_terrain_binary_choice.scene import MultiChoiceCfg  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_summary(path: Path) -> dict:
    """Summarize the saved episodes by condition without another simulation."""
    groups = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            groups[f"{row['layout']}__{row['geometry']}"].append(row)
    summary = {}
    for name, rows in groups.items():
        arrivals = [row for row in rows if row["success"] == "1"]
        summary[name] = {
            "episodes": len(rows), "arrivals": len(arrivals),
            "mean_arrival_energy_j": (sum(float(row["energy_j"]) for row in arrivals)
                                      / len(arrivals) if arrivals else None),
            "mean_arrival_path_m": (sum(float(row["path_length_m"]) for row in arrivals)
                                    / len(arrivals) if arrivals else None),
        }
    return summary


class TrainingLog(BaseCallback):
    def __init__(self, run_dir: Path, target: int, conditions, initial_steps: int = 0):
        super().__init__()
        self.run_dir = run_dir
        self.target = target
        self.conditions = conditions
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
        self.episode_writer.writerow(("timesteps", "env_id", "layout", "geometry",
                                      "pair", "costly_side", "reward", "actions", "success",
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
                condition = self.conditions[env_id]
                success = self.value(info, "is_success")
                self.episodes += 1
                self.successes += int(success)
                reason = ("goal" if success else
                          "timeout" if info.get("TimeLimit.truncated") else "unsafe")
                self.episode_writer.writerow((
                    self.num_timesteps, env_id, condition.layout.name, condition.geometry,
                    "/".join(condition.layout.pair), condition.layout.costly_side,
                    episode["r"], episode["l"], int(success), reason,
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
            print(f"[TEST2] transitions={row[0]}/{self.target} rate={row[3]:.2f}/s "
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


def probe(env: MultiChoiceEnv, steps: int) -> None:
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    for condition in env.conditions:
        base = f"/World/envs/env_{condition.env_id}"
        for path in (f"{base}/Warehouse", f"{base}/Spot",
                     f"{base}/NavigationZEDCamera"):
            if not stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"Missing prim in {condition.name}: {path}")
        grid_path = f"{base}/Warehouse/Grid"
        actual = {child.GetName() for child in stage.GetPrimAtPath(grid_path).GetChildren()}
        expected = {f"Cell_{row}_{col}_{terrain}"
                    for row, cells in enumerate(condition.layout.grid)
                    for col, terrain in enumerate(cells)}
        if actual != expected:
            raise RuntimeError(f"Wrong terrain in {condition.name}: {actual ^ expected}")
    wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
    observation = wrapped.reset()
    print(f"[PROBE] verified_conditions={len(env.conditions)} "
          f"envs={wrapped.num_envs} observation={observation.shape} "
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
    conditions = training_conditions()
    for condition in conditions:
        read_terrain_codes(condition.case)
    cfg = MultiChoiceCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = NUM_ENVS
    cfg.episode_length_s = NAV_CFG_BINARY.max_episode_seconds
    cfg.robot_cfg.init_state.pos = (0.0, 0.0, NAV_CFG_BINARY.spawn_height_m)
    half = math.pi / 8.0  # initial yaw is 45 degrees in every environment
    cfg.robot_cfg.init_state.rot = (math.cos(half), 0.0, 0.0, math.sin(half))
    cfg.viewer.eye = (8.0, 9.0, 11.0)
    cfg.viewer.lookat = (2.0, 2.0, 0.0)
    run_dir = None
    if args.probe_steps == 0:
        run_dir = HERE / "runs" / datetime.now().strftime(
            "test2_curriculum_32env_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
        terrain_hashes = {item.layout.name: sha256(item.layout.usd)
                          for item in conditions}
        manifest = [{"env_id": item.env_id, "layout": item.layout.name,
                     "geometry": item.geometry, "pair": list(item.layout.pair),
                     "costly_side": item.layout.costly_side,
                     "start_xy": list(item.start_xy), "goal_xy": list(item.goal_xy)}
                    for item in conditions]
        settings = {
            "mode": "online", "num_envs": NUM_ENVS,
            "seed": args.seed, "additional_transitions": args.timesteps,
            "source_checkpoint": str(args.source_checkpoint.resolve()),
            "source_checkpoint_sha256": sha256(args.source_checkpoint),
            "train_layouts": manifest, "terrain_sha256": terrain_hashes,
            "feature_atlas_for_pca_only": str(args.atlas.resolve()),
            "atlas_sha256": sha256(args.atlas),
            "environment": json.loads(json.dumps(asdict(NAV_CFG_BINARY))),
            "headless": args.headless,
            "camera": "32 ZED X CameraRight RGB-D 640x360, 15deg pitch",
            "dino": ("one shared frozen facebook/dinov2-small, "
                     f"microbatch={args.dino_batch_size}, input=518"),
            "visual_components": PCA_DIM, "local_visual_cells": LOCAL_CELLS,
            "visual_sampling": {"look_x_m": LOOK_X_M, "look_y_m": LOOK_Y_M,
                                "resolution_m": RESOLUTION_M, "x0_m": X0_M, "y0_m": Y0_M},
            "ppo": {"n_steps_per_env": PPO_CFG.n_steps_per_env,
                    "batch_size": PPO_CFG.batch_size,
                    "n_epochs": PPO_CFG.n_epochs,
                    "learning_rate": PPO_CFG.learning_rate,
                    "gamma": PPO_CFG.gamma,
                    "net_arch": list(PPO_CFG.net_arch)},
        }
        (run_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        print(f"[TEST2] run_dir={run_dir}", flush=True)
    env = MultiChoiceEnv(cfg, conditions, args.atlas, dino_batch_size=args.dino_batch_size)
    try:
        if args.probe_steps:
            probe(env, args.probe_steps)
            return
        wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
        old_config = args.source_checkpoint.parent / "config.json"
        if not old_config.is_file():
            raise RuntimeError(f"Missing source training configuration: {old_config}")
        saved = json.loads(old_config.read_text(encoding="utf-8"))
        for key in ("atlas_sha256", "environment", "ppo", "visual_sampling"):
            # JSON stores the LOOK_X/Y tuples as lists; compare the serialized
            # forms so equal sampling settings are not rejected by type alone.
            if saved.get(key) != json.loads(json.dumps(settings[key])):
                raise RuntimeError(f"Source checkpoint uses different {key}")
        if saved.get("case") != "asphalt_rocks" or saved.get("num_envs") != NUM_ENVS:
            raise RuntimeError("Source must be the successful 32-robot asphalt/rocks run")
        if saved.get("terrain_sha256") != settings["terrain_sha256"]["original_asphalt_rocks"]:
            raise RuntimeError("Original rehearsal USD differs from source training")
        model = PPO.load(str(args.source_checkpoint), device="cpu")
        if (model.n_envs != NUM_ENVS or model.n_steps != PPO_CFG.n_steps_per_env
                or model.observation_space != wrapped.observation_space
                or model.action_space != wrapped.action_space):
            raise RuntimeError("Source checkpoint has incompatible PPO or observation spaces")
        model.set_env(wrapped)
        settings["source_transitions"] = model.num_timesteps
        settings["target_transitions"] = model.num_timesteps + args.timesteps
        (run_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        print(f"[TEST2] source={args.source_checkpoint.resolve()} "
              f"source_transitions={model.num_timesteps} "
              f"additional_transitions={args.timesteps} "
              f"target_transitions={settings['target_transitions']}", flush=True)
        logger = TrainingLog(run_dir, model.num_timesteps + args.timesteps, conditions,
                             initial_steps=model.num_timesteps)
        callbacks = CallbackList([
            logger,
            CheckpointCallback(save_freq=10240 // NUM_ENVS,
                               save_path=str(run_dir), name_prefix="policy_step"),
        ])
        started = time.perf_counter()
        interrupted = False
        try:
            model.learn(total_timesteps=args.timesteps, callback=callbacks,
                        reset_num_timesteps=False)
        except KeyboardInterrupt:
            interrupted = True
            print("[TEST2] Ctrl+C received; saving current policy", flush=True)
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
            "per_condition": episode_summary(run_dir / "episodes.csv"),
        }, indent=2), encoding="utf-8")
        print(f"[TEST2] checkpoint={output}.zip transitions={model.num_timesteps} "
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

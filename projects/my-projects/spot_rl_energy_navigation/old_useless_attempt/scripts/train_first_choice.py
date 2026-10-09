"""Train the first terrain-choice policy using pose-aligned cached DINO features."""

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
DEFAULT_ATLAS = HERE / "runs" / "first_choice_feature_atlas.npz"
parser = argparse.ArgumentParser()
parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
parser.add_argument("--timesteps", type=int, default=30_720)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--check-only", action="store_true")
parser.add_argument("--visible-camera", action="store_true")
parser.add_argument("--augment", action="store_true")
parser.add_argument("--resume", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.timesteps < 256 or args.timesteps % 256:
    parser.error("--timesteps must be a positive multiple of 256")
if not args.atlas.is_file():
    parser.error(f"Feature atlas missing: {args.atlas}")
if args.resume is not None and not args.resume.is_file():
    parser.error(f"Checkpoint missing: {args.resume}")
args.enable_cameras = args.visible_camera
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from first_choice_env import FirstChoiceEnv, FIRST_CHOICE_CFG  # noqa: E402
from rl_config import validate_assets  # noqa: E402


def main():
    validate_assets()
    run_dir = HERE / "runs" / datetime.now().strftime("first_choice_%Y%m%d_%H%M%S")
    if not args.check_only:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(json.dumps({
            "atlas": str(args.atlas), "timesteps": args.timesteps,
            "seed": args.seed, "environment": asdict(FIRST_CHOICE_CFG),
            "visual_mode": "precomputed DINO world field",
            "visible_camera": args.visible_camera, "augment": args.augment,
            "resume": str(args.resume) if args.resume else None,
        }, indent=2), encoding="utf-8")
    env = FirstChoiceEnv(simulation_app, args.atlas, online=False,
                         cached_visible=args.visible_camera, augment=args.augment,
                         device=args.device, render=not args.headless)
    try:
        if args.check_only:
            obs, info = env.reset(seed=args.seed)
            print(f"[CHECK] obs_shape={obs.shape} valid_visual={info['visual_valid']} "
                  f"goal_distance={info['goal_distance_m']:.2f} "
                  f"root={env.robot.data.root_pos_w[0].tolist()} "
                  f"quat={env.robot.data.root_quat_w[0].tolist()} "
                  f"mask={obs[11:].reshape(-1,13)[:,-1].tolist()}", flush=True)
            for i in range(3):
                obs, reward, term, trunc, info = env.step(np.array([0.5, 0.0, 0.0], dtype=np.float32))
                print(f"[CHECK] step={i+1} reward={reward:.2f} "
                      f"valid_visual={info['visual_valid']} reason={info['termination_reason']}", flush=True)
                if term or trunc:
                    break
            return
        monitored = Monitor(
            env, filename=str(run_dir / "episodes.monitor.csv"),
            info_keywords=("is_success", "episode_energy_j", "final_distance_m",
                           "route", "path_length_m", "visual_valid"),
        )
        model = (PPO.load(str(args.resume), env=monitored, device="cpu")
                 if args.resume is not None else PPO(
                     "MlpPolicy", monitored, seed=args.seed, device="cpu",
                     n_steps=256, batch_size=128, n_epochs=5,
                     learning_rate=3e-4, gamma=0.99,
                     policy_kwargs={"net_arch": [128, 128]}, verbose=1,
                 ))
        callback = CheckpointCallback(save_freq=5120, save_path=str(run_dir),
                                      name_prefix="policy_step")
        model.learn(total_timesteps=args.timesteps, callback=callback,
                    reset_num_timesteps=args.resume is None)
        model.save(str(run_dir / "policy_final"))
        print(f"[TRAIN-CHOICE] saved={run_dir / 'policy_final.zip'}", flush=True)
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

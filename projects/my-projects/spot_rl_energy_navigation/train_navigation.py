"""Train the first high-level Spot navigation policy with PPO.

This experiment has one robot and one asphalt cell. It does not use DINOv2.
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train Spot to reach a goal using velocity commands.")
parser.add_argument("--timesteps", type=int, default=10_000, help="High-level RL transitions.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.timesteps < 1:
    parser.error("--timesteps must be positive")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from navigation_config import NAV_CFG  # noqa: E402
from navigation_env import SpotNavigationEnv  # noqa: E402
from rl_config import validate_assets  # noqa: E402


def main() -> None:
    validate_assets()
    run_dir = (Path(__file__).resolve().parent / "runs"
               / datetime.now().strftime("navigation_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps({"environment": asdict(NAV_CFG), "timesteps": args.timesteps,
                    "seed": args.seed}, indent=2),
        encoding="utf-8",
    )
    print(f"[TRAIN] Logs and checkpoints: {run_dir}", flush=True)

    env = SpotNavigationEnv(simulation_app, device=args.device, render=not args.headless)
    monitored_env = Monitor(
        env, filename=str(run_dir / "episodes.monitor.csv"),
        info_keywords=("is_success", "episode_energy_j", "final_distance_m"),
    )
    try:
        model = PPO(
            "MlpPolicy", monitored_env, seed=args.seed, device="cpu",
            n_steps=128, batch_size=64, n_epochs=5,
            learning_rate=3e-4, gamma=0.99,
            verbose=1,
        )
        callback = CheckpointCallback(
            save_freq=2048, save_path=str(run_dir), name_prefix="policy_step"
        )
        model.learn(total_timesteps=args.timesteps, callback=callback)
        output = run_dir / "policy_final"
        model.save(str(output))
        print(f"[TRAIN] Saved policy: {output}.zip", flush=True)
    finally:
        monitored_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=args.headless)

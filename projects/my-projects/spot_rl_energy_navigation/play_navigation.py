"""Evaluate a saved high-level navigation policy in the same Isaac Lab scene."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay a trained Spot navigation policy.")
parser.add_argument("--checkpoint", type=Path, required=True, help="PPO policy .zip file.")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1:
    parser.error("--episodes must be positive")
if not args.checkpoint.is_file():
    parser.error(f"Checkpoint not found: {args.checkpoint}")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


from stable_baselines3 import PPO  # noqa: E402

from navigation_env import SpotNavigationEnv  # noqa: E402
from rl_config import validate_assets  # noqa: E402


def main() -> None:
    validate_assets()
    model = PPO.load(str(args.checkpoint), device="cpu")
    env = SpotNavigationEnv(simulation_app, device=args.device, render=not args.headless)
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            episode_reward = 0.0
            while simulation_app.is_running():
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    print(
                        f"[PLAY] episode={episode + 1} reason={info['termination_reason']} "
                        f"distance={info['final_distance_m']:.2f}m "
                        f"energy={info['episode_energy_j']:.1f}J "
                        f"reward={episode_reward:.2f}",
                        flush=True,
                    )
                    break
            if not simulation_app.is_running():
                break
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=args.headless)

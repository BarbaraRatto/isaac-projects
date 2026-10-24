"""Test7: fresh SAC, raw online DINO and two-branch visual/state encoders.

Run from my-projects. --probe-steps launches a technical check without training.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from isaaclab.app import AppLauncher
from multi_terrain_test7.config import NAV_CFG, SAC_CFG, experiment_signature

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--timesteps", type=int, default=SAC_CFG.timesteps,
                    help="training transitions summed over all 32 robots")
parser.add_argument("--dino-batch-size", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--probe-steps", type=int, default=0)
parser.add_argument("--agent-device", default="cpu", help="SAC device; independent of simulator --device")
parser.add_argument("--torch-threads", type=int, default=4, help="CPU threads used by PyTorch")
parser.add_argument("--buffer-size", type=int, default=SAC_CFG.buffer_size,
                    help="total replay transitions across all environments")
parser.add_argument("--gradient-steps", type=int, default=SAC_CFG.gradient_steps,
                    help="SAC minibatch updates after each 32-robot vector step")
parser.add_argument("--eval-every", type=int, default=SAC_CFG.eval_every,
                    help="training transitions between deterministic evaluations; 0 disables")
parser.add_argument("--checkpoint-every", type=int, default=SAC_CFG.checkpoint_every)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
NUM_ENVS = SAC_CFG.num_envs
if args.timesteps < NUM_ENVS or args.timesteps % NUM_ENVS:
    parser.error("--timesteps must be a positive multiple of 32")
if args.probe_steps < 0 or not 1 <= args.dino_batch_size <= NUM_ENVS:
    parser.error("--probe-steps must be nonnegative and --dino-batch-size between 1 and 32")
if args.buffer_size < SAC_CFG.batch_size or args.buffer_size % NUM_ENVS:
    parser.error("--buffer-size must be at least 256 and a multiple of 32")
if args.gradient_steps < 1 or args.torch_threads < 1:
    parser.error("--gradient-steps and --torch-threads must be positive")
if args.eval_every < 0 or args.eval_every % NUM_ENVS:
    parser.error("--eval-every must be zero or a positive multiple of 32")
if args.checkpoint_every < NUM_ENVS or args.checkpoint_every % NUM_ENVS:
    parser.error("--checkpoint-every must be a positive multiple of 32")
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402

from multi_terrain_test7.assets import validate_assets  # noqa: E402
from multi_terrain_test7.env import Test7Env  # noqa: E402
from multi_terrain_test7.evaluation import PeriodicEvaluation  # noqa: E402
from multi_terrain_test7.layouts import training_conditions  # noqa: E402
from multi_terrain_test7.logging_callbacks import TrainingLog, episode_summary  # noqa: E402
from multi_terrain_test7.networks import policy_kwargs  # noqa: E402
from multi_terrain_test7.probe import probe  # noqa: E402
from multi_terrain_test7.scene import Test7Cfg  # noqa: E402
from multi_terrain_test7.terrain import read_terrain_codes  # noqa: E402
from multi_terrain_test7.utils import terrain_hashes, replay_bytes  # noqa: E402


def main() -> None:
    validate_assets()
    torch.set_num_threads(args.torch_threads)
    conditions = training_conditions()
    for condition in conditions:
        read_terrain_codes(condition.case)
    cfg = Test7Cfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    env = Test7Env(cfg, conditions, dino_batch_size=args.dino_batch_size)
    try:
        if args.probe_steps:
            probe(env, args.probe_steps)
            return
        run_dir = HERE / "runs" / datetime.now().strftime("test7_sac_32env_%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        settings = {
            "algorithm": "SAC", "initialization": "fresh_random_policy",
            "signature": experiment_signature(), "num_envs": NUM_ENVS,
            "seed": args.seed, "target_transitions": args.timesteps,
            "terrain_sha256": terrain_hashes(),
            "train_layouts": [{"env_id": item.env_id, "layout": item.layout.name,
                               "start_xy": item.start_xy, "goal_xy": item.goal_xy}
                              for item in conditions],
            "sac": {**asdict(SAC_CFG), "buffer_size": args.buffer_size,
                    "gradient_steps": args.gradient_steps, "timesteps": args.timesteps,
                    "eval_every": args.eval_every, "checkpoint_every": args.checkpoint_every},
            "replay_estimated_gib": replay_bytes(args.buffer_size, NUM_ENVS) / 1024**3,
            "sim_device": args.device, "agent_device": args.agent_device,
            "torch_threads": args.torch_threads, "headless": args.headless,
            "dino_batch_size": args.dino_batch_size,
            "evaluation_seed": args.seed + 10_000,
        }
        config_file = run_dir / "config.json"
        config_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        (run_dir / "sources").mkdir()
        for source in HERE.glob("*.py"):
            (run_dir / "sources" / source.name).write_bytes(source.read_bytes())
        print(f"[TEST7] run_dir={run_dir} replay≈{settings['replay_estimated_gib']:.2f}GiB "
              f"agent={args.agent_device} gradient_steps={args.gradient_steps}/vector_step", flush=True)
        wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
        model = SAC(
            "MlpPolicy", wrapped, seed=args.seed, device=args.agent_device,
            learning_rate=SAC_CFG.learning_rate, buffer_size=args.buffer_size,
            learning_starts=SAC_CFG.learning_starts, batch_size=SAC_CFG.batch_size,
            gamma=SAC_CFG.gamma, tau=SAC_CFG.tau,
            train_freq=(SAC_CFG.train_freq, "step"), gradient_steps=args.gradient_steps,
            ent_coef=SAC_CFG.ent_coef, target_entropy=SAC_CFG.target_entropy,
            policy_kwargs=policy_kwargs(), verbose=1,
        )
        model.set_logger(configure(str(run_dir / "sb3"), ["stdout", "csv"]))
        settings["policy_parameters_including_targets"] = sum(p.numel() for p in model.policy.parameters())
        config_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        (run_dir / "policy_architecture.txt").write_text(str(model.policy), encoding="utf-8")
        logger = TrainingLog(run_dir, args.timesteps, conditions)
        evaluator = PeriodicEvaluation(run_dir, conditions, args.eval_every, args.seed + 10_000)
        callbacks = CallbackList([
            logger,
            CheckpointCallback(save_freq=args.checkpoint_every // NUM_ENVS,
                               save_path=str(run_dir), name_prefix="policy_step"),
            evaluator,
        ])
        interrupted = False
        started = time.perf_counter()
        try:
            model.learn(total_timesteps=args.timesteps, callback=callbacks, log_interval=100)
            if args.eval_every:
                evaluator.evaluate_now(phase="final")
        except KeyboardInterrupt:
            interrupted = True
            print("[TEST7] Ctrl+C received; saving current SAC policy", flush=True)
        finally:
            logger.close_files()
        output = run_dir / (f"policy_interrupted_{model.num_timesteps}_steps"
                            if interrupted else "policy_final")
        model.save(str(output))
        summary = {
            "interrupted": interrupted, "num_timesteps": model.num_timesteps,
            "wall_seconds": time.perf_counter() - started,
            "dino_load_seconds": env.vision.load_seconds,
            "checkpoint": str(output) + ".zip", "episodes": logger.episodes,
            "successes": logger.successes,
            "per_condition": episode_summary(run_dir / "episodes.csv"),
            "best_checkpoint": str(run_dir / "policy_best.zip") if evaluator.best_score is not None else None,
            "evaluation_in_replay": False, "replay_buffer_saved": False,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[TEST7] checkpoint={output}.zip transitions={model.num_timesteps} "
              f"wall={summary['wall_seconds']:.1f}s", flush=True)
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

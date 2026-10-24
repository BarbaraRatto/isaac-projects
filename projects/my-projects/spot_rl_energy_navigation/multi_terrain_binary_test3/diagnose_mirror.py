"""Diagnose mirrored-board failures and check the right/gravel route.

This is an evaluation only: it neither trains nor changes a checkpoint.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("policy", "guided", "both"), default="both")
parser.add_argument("--checkpoint", type=Path, default=(
    HERE / "runs" / "test3_mirrored_32env_20261010_233033" / "policy_final.zip"))
parser.add_argument("--atlas", type=Path, default=HERE.parent / "runs" / "ice_choice_atlas.npz")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--speed", type=float, default=0.65,
                    help="forward speed in m/s for the guided run")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1 or not 0 < args.speed <= 1.6:
    parser.error("--episodes must be positive and --speed must be in (0, 1.6]")
if not args.atlas.is_file() or (args.mode != "guided" and not args.checkpoint.is_file()):
    parser.error("Missing PCA atlas or checkpoint")
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from binary_choice.config import NAV_CFG_BINARY  # noqa: E402
from binary_choice.visual_features import LOOK_X_M, LOOK_Y_M, RESOLUTION_M, X0_M, Y0_M  # noqa: E402
from multi_terrain_binary_choice.env import MultiChoiceEnv  # noqa: E402
from multi_terrain_binary_choice.scene import MultiChoiceCfg  # noqa: E402
from multi_terrain_binary_test3.layouts import evaluation_condition, training_conditions  # noqa: E402
from rl_config import validate_assets  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(info: dict, key: str) -> float:
    value = info[key]
    return float(value.item()) if hasattr(value, "item") else float(value)


class DiagnosticEnv(MultiChoiceEnv):
    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        pos = self.robot.data.root_pos_w - self.scene.env_origins
        gravity = self.robot.data.projected_gravity_b[:, 2]
        bounds_x = self.geometry.world_x_bounds
        bounds_y = self.geometry.world_y_bounds
        self.extras.update({
            "body_z_m": pos[:, 2].clone(),
            "gravity_z": gravity.clone(),
            "unsafe_low_height": (pos[:, 2] < self.nav.minimum_body_height_m).clone(),
            "unsafe_tilt": (gravity > -0.5).clone(),
            "unsafe_x_min": (pos[:, 0] < bounds_x[0]).clone(),
            "unsafe_x_max": (pos[:, 0] > bounds_x[1]).clone(),
            "unsafe_y_min": (pos[:, 1] < bounds_y[0]).clone(),
            "unsafe_y_max": (pos[:, 1] > bounds_y[1]).clone(),
        })
        return reward


UNSAFE_KEYS = ("unsafe_low_height", "unsafe_tilt", "unsafe_x_min",
               "unsafe_x_max", "unsafe_y_min", "unsafe_y_max")


def guided_action(env: DiagnosticEnv, waypoint: tuple[float, float]) -> np.ndarray:
    local = (env.robot.data.root_pos_w[0, :2] - env.scene.env_origins[0, :2]).detach().cpu().numpy()
    q = env.robot.data.root_quat_w[0].detach().cpu().numpy()
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                     1 - 2 * (q[2] * q[2] + q[3] * q[3]))
    direction = math.atan2(waypoint[1] - float(local[1]), waypoint[0] - float(local[0]))
    error = math.atan2(math.sin(direction - yaw), math.cos(direction - yaw))
    turn = float(np.clip(2.0 * error, -NAV_CFG_BINARY.max_yaw_rad_s,
                         NAV_CFG_BINARY.max_yaw_rad_s))
    forward = args.speed * max(0.0, math.cos(error))
    return np.array([[forward / NAV_CFG_BINARY.max_forward_m_s, 0.0,
                      turn / NAV_CFG_BINARY.max_yaw_rad_s]], dtype=np.float32)


def main() -> None:
    validate_assets()
    condition = evaluation_condition("rocks_gravel__right_gravel", "shifted")
    if args.mode != "guided":
        config_file = args.checkpoint.parent / "config.json"
        if not config_file.is_file():
            raise RuntimeError(f"Missing checkpoint configuration: {config_file}")
        saved = json.loads(config_file.read_text(encoding="utf-8"))
        hashes = {item.layout.name: sha256(item.layout.usd) for item in training_conditions()}
        visual = {"look_x_m": LOOK_X_M, "look_y_m": LOOK_Y_M,
                  "resolution_m": RESOLUTION_M, "x0_m": X0_M, "y0_m": Y0_M}
        if (saved.get("num_envs") != 32 or saved.get("terrain_sha256") != hashes
                or saved.get("atlas_sha256") != sha256(args.atlas)
                or saved.get("environment") != json.loads(json.dumps(asdict(NAV_CFG_BINARY)))
                or saved.get("visual_sampling") != json.loads(json.dumps(visual))):
            raise RuntimeError("Checkpoint configuration differs from test3")
    cfg = MultiChoiceCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.episode_length_s = NAV_CFG_BINARY.max_episode_seconds
    cfg.robot_cfg.init_state.pos = (0.0, 0.0, NAV_CFG_BINARY.spawn_height_m)
    half = math.pi / 8.0
    cfg.robot_cfg.init_state.rot = (math.cos(half), 0.0, 0.0, math.sin(half))
    cfg.viewer.eye = (8.0, 9.0, 11.0)
    cfg.viewer.lookat = (2.0, 2.0, 0.0)
    env = DiagnosticEnv(cfg, (condition,), args.atlas, dino_batch_size=1)
    try:
        wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
        model = None
        if args.mode != "guided":
            model = PPO.load(str(args.checkpoint), device="cpu")
            if model.observation_space != wrapped.observation_space or model.action_space != wrapped.action_space:
                raise RuntimeError("Checkpoint observation/action spaces differ from environment")
        rows = []
        trajectory = []
        for mode in (("policy", "guided") if args.mode == "both" else (args.mode,)):
            obs = wrapped.reset()
            for episode in range(1, args.episodes + 1):
                # Stay in the top/right gravel cells, avoiding the center ice.
                waypoints = ((2.0, 0.0), (3.7, 0.0), (3.7, 2.0), condition.goal_xy)
                waypoint_index = 0
                closest = float("inf")
                for decision in range(1, env.max_episode_length + 2):
                    if mode == "policy":
                        action, _ = model.predict(obs, deterministic=True)
                    else:
                        local = (env.robot.data.root_pos_w[0, :2]
                                 - env.scene.env_origins[0, :2]).detach().cpu().tolist()
                        while (waypoint_index < len(waypoints) - 1
                               and math.dist(local, waypoints[waypoint_index]) < 0.4):
                            waypoint_index += 1
                        action = guided_action(env, waypoints[waypoint_index])
                    obs, _, dones, infos = wrapped.step(action)
                    info = infos[0]
                    distance = number(info, "final_distance_m")
                    closest = min(closest, distance)
                    flags = [key.removeprefix("unsafe_") for key in UNSAFE_KEYS
                             if bool(number(info, key))]
                    xy = (number(info, "body_x_m"), number(info, "body_y_m"))
                    trajectory.append({
                        "mode": mode, "episode": episode, "decision": decision,
                        "x_m": round(xy[0], 4), "y_m": round(xy[1], 4),
                        "z_m": round(number(info, "body_z_m"), 4),
                        "gravity_z": round(number(info, "gravity_z"), 4),
                        "goal_distance_m": round(distance, 4),
                        "unsafe_flags": "+".join(flags),
                    })
                    if bool(dones[0]):
                        success = bool(number(info, "is_success"))
                        reason = "goal" if success else (
                            "timeout" if info.get("TimeLimit.truncated") else "+".join(flags) or "unsafe_unknown")
                        row = {
                            "mode": mode, "episode": episode, "reason": reason,
                            "success": int(success), "decisions": decision,
                            "final_x_m": round(xy[0], 4), "final_y_m": round(xy[1], 4),
                            "final_z_m": round(number(info, "body_z_m"), 4),
                            "final_gravity_z": round(number(info, "gravity_z"), 4),
                            "final_goal_distance_m": round(distance, 4),
                            "closest_goal_distance_m": round(closest, 4),
                            "gravel_entry": int(number(info, "gravel_entry")),
                            "rocks_entry": int(number(info, "rocks_entry")),
                            "ice_entry": int(number(info, "ice_entry")),
                            "energy_j": round(number(info, "episode_energy_j"), 2),
                            "path_length_m": round(number(info, "path_length_m"), 3),
                        }
                        rows.append(row)
                        print(f"[DIAG] {row}", flush=True)
                        break
                else:
                    raise RuntimeError("Episode exceeded configured maximum actions")
        output = args.output or HERE / "results" / f"mirror_diagnosis_{datetime.now():%Y%m%d_%H%M%S}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        trace_output = output.with_name(output.stem + "_trajectory.csv")
        with trace_output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(trajectory[0]))
            writer.writeheader()
            writer.writerows(trajectory)
        print(f"[DIAG] episodes={output.resolve()} trajectory={trace_output.resolve()}", flush=True)
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

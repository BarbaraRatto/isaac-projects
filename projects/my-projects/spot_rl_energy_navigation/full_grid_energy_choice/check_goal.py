"""Compare guided route energy on the full terrain; no RL training."""

import argparse
import csv
from dataclasses import replace
import math
import statistics
import sys
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--speed", type=float, default=0.8, help="maximum commanded forward speed in m/s")
parser.add_argument("--scenario", choices=("original", "border"), default="original",
                    help="original start/goal or symmetric border comparison")
parser.add_argument("--routes", nargs="+", choices=("direct", "asphalt", "rocks", "center"),
                    default=None, help="routes to compare; defaults depend on scenario")
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1 or not 0.0 < args.speed <= 1.6:
    parser.error("Require episodes >= 1 and speed in (0, 1.6] m/s")
args.enable_cameras = False
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from full_grid_energy_choice.config import GEOMETRY, NAV_CFG_FULL  # noqa: E402
from full_grid_energy_choice.env import FullGridChoiceEnv  # noqa: E402
from full_grid_energy_choice.scene import FullGridChoiceCfg  # noqa: E402
from rl_config import validate_assets  # noqa: E402


def number(info, key):
    value = info[key]
    return float(value.item()) if hasattr(value, "item") else float(value)


def guided_action(env, target_xy, final_target):
    local = (env.robot.data.root_pos_w[0, :2] - env.scene.env_origins[0, :2]).detach().cpu().numpy()
    q = env.robot.data.root_quat_w[0].detach().cpu().numpy()
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                     1 - 2 * (q[2] * q[2] + q[3] * q[3]))
    dx = target_xy[0] - float(local[0])
    dy = target_xy[1] - float(local[1])
    distance = math.hypot(dx, dy)
    error = math.atan2(math.sin(math.atan2(dy, dx) - yaw),
                       math.cos(math.atan2(dy, dx) - yaw))
    turn = float(np.clip(2.0 * error, -NAV_CFG_FULL.max_yaw_rad_s, NAV_CFG_FULL.max_yaw_rad_s))
    # Decelerating in proportion to distance minus goal radius approaches the
    # boundary asymptotically and can time out just outside the goal. Keep the
    # selected speed for the final target until the environment declares success.
    # One decision lasts 0.2 s; at 0.8 m/s this advances about 0.16 m.
    forward = (args.speed if final_target else min(args.speed, max(0.4, distance)))
    forward *= max(0.0, math.cos(error))
    return np.array([[forward / NAV_CFG_FULL.max_forward_m_s, 0.0,
                      turn / NAV_CFG_FULL.max_yaw_rad_s]], dtype=np.float32)


def main():
    validate_assets()
    cfg = FullGridChoiceCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    geometry = (replace(GEOMETRY, start_xy=(3.0, 6.0), goal_xy=(3.0, 10.5),
                        start_jitter_m=0.0) if args.scenario == "border" else GEOMETRY)
    cfg.robot_cfg.init_state.pos = (*geometry.start_xy, NAV_CFG_FULL.spawn_height_m)
    routes = args.routes or (("asphalt", "rocks") if args.scenario == "border"
                             else ("direct", "asphalt"))
    if args.scenario == "original" and any(route in ("rocks", "center") for route in routes):
        parser.error("The rocks and center routes are available only for --scenario border")
    if args.scenario == "border" and "direct" in routes:
        parser.error("The direct route is available only for --scenario original")
    env = FullGridChoiceEnv(cfg, atlas_file=None, geometry=geometry)
    try:
        wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
        wrapped.reset()
        rows = []
        if args.scenario == "border":
            # Equal geometric lengths: about 5.11 m for either side.
            waypoints = {
                "asphalt": ((2.2, 7.8), (2.2, 10.0)),
                "rocks": ((3.8, 7.8), (3.8, 10.0)),
                "center": (),  # Goal at x=3, so the body follows the cell boundary.
            }
        else:
            waypoints = {
                "direct": (),
                "asphalt": ((2.1, 7.8), (2.1, 10.3)),
            }
        for route in routes:
            for episode in range(1, args.episodes + 1):
                targets = waypoints[route] + (geometry.goal_xy,)
                waypoint_index = 0
                actions = 0
                middle_asphalt_m = 0.0
                middle_rocks_m = 0.0
                middle_center_m = 0.0
                middle_total_m = 0.0
                middle_max_abs_x_error_m = 0.0
                middle_steps = 0
                middle_mixed_foot_steps = 0
                while True:
                    local = (env.robot.data.root_pos_w[0, :2]
                             - env.scene.env_origins[0, :2]).detach().cpu().numpy()
                    if (waypoint_index < len(targets) - 1
                            and math.dist(local, targets[waypoint_index]) < 0.25):
                        waypoint_index += 1
                    _, _, dones, infos = wrapped.step(
                        guided_action(env, targets[waypoint_index], waypoint_index == len(targets) - 1))
                    actions += 1
                    if bool(dones[0]):
                        # The wrapper has already reset this robot; its new
                        # position is not part of the finished trajectory.
                        break
                    next_local = (env.robot.data.root_pos_w[0, :2]
                                  - env.scene.env_origins[0, :2]).detach().cpu().numpy()
                    midpoint = (local + next_local) / 2.0
                    segment_m = math.dist(local, next_local)
                    if 7.5 <= midpoint[1] < 10.5:
                        middle_total_m += segment_m
                        x_error = abs(float(midpoint[0]) - 3.0)
                        middle_max_abs_x_error_m = max(middle_max_abs_x_error_m, x_error)
                        if x_error <= 0.35:
                            middle_center_m += segment_m
                        feet = (env.robot.data.body_pos_w[0, env.foot_ids, :2]
                                - env.scene.env_origins[0, :2]).detach().cpu().numpy()
                        in_row = (feet[:, 1] >= 7.5) & (feet[:, 1] < 10.5)
                        middle_mixed_foot_steps += int(bool((in_row & (feet[:, 0] < 3.0)).any()
                                                            and (in_row & (feet[:, 0] >= 3.0)).any()))
                        middle_steps += 1
                        if -3.0 <= midpoint[0] < 3.0:
                            middle_asphalt_m += segment_m
                        elif 3.0 <= midpoint[0] < 9.0:
                            middle_rocks_m += segment_m
                info = infos[0]
                success = int(number(info, "is_success"))
                # Check the actual body trajectory. Foot-contact flags remain
                # descriptive: feet may touch a border even if the body stays
                # on the intended side of it.
                if route == "center":
                    route_followed = (middle_total_m >= 1.0
                                      and middle_center_m / middle_total_m >= 0.8
                                      and middle_max_abs_x_error_m <= 0.4)
                elif route in ("direct", "rocks"):
                    route_followed = (waypoint_index == len(waypoints[route])
                                      and middle_rocks_m >= 1.0 and middle_asphalt_m < 0.2)
                else:
                    route_followed = (waypoint_index == len(waypoints[route])
                                      and middle_asphalt_m >= 1.0 and middle_rocks_m < 0.2)
                row = {
                    "scenario": args.scenario,
                    "start_x_m": geometry.start_xy[0],
                    "start_y_m": geometry.start_xy[1],
                    "goal_x_m": geometry.goal_xy[0],
                    "goal_y_m": geometry.goal_xy[1],
                    "route": route,
                    "episode": episode,
                    "success": success,
                    "route_followed": int(route_followed),
                    "reason": "goal" if success else ("timeout" if info.get("TimeLimit.truncated") else "unsafe"),
                    "command_speed_cap_m_s": args.speed,
                    "energy_j": round(number(info, "episode_energy_j"), 2),
                    "path_length_m": round(number(info, "path_length_m"), 3),
                    "middle_asphalt_body_m": round(middle_asphalt_m, 3),
                    "middle_rocks_body_m": round(middle_rocks_m, 3),
                    "middle_center_body_m": round(middle_center_m, 3),
                    "middle_total_body_m": round(middle_total_m, 3),
                    "middle_max_abs_x_error_m": round(middle_max_abs_x_error_m, 3),
                    "middle_mixed_foot_steps": middle_mixed_foot_steps,
                    "middle_steps": middle_steps,
                    "asphalt_entry": int(number(info, "asphalt_entry")),
                    "rocks_entry": int(number(info, "rocks_entry")),
                    "obstacle_entry": int(number(info, "obstacle_entry")),
                    "final_distance_m": round(number(info, "final_distance_m"), 3),
                    "actions": actions,
                }
                rows.append(row)
                print(f"[GUIDED] {row}", flush=True)
        output = args.output or HERE / "results" / f"guided_{args.scenario}_check_{datetime.now():%Y%m%d_%H%M%S}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[GUIDED] saved={output.resolve()}", flush=True)
        for route in routes:
            selected = [row for row in rows if row["route"] == route]
            valid_arrivals = [row for row in selected if row["success"] and row["route_followed"]]
            if valid_arrivals:
                mean_energy = statistics.fmean(row["energy_j"] for row in valid_arrivals)
                mean_distance = statistics.fmean(row["path_length_m"] for row in valid_arrivals)
                details = ""
                if route == "center":
                    mean_center_fraction = statistics.fmean(
                        row["middle_center_body_m"] / row["middle_total_body_m"]
                        for row in valid_arrivals if row["middle_total_body_m"] > 0
                    )
                    mean_mixed_fraction = statistics.fmean(
                        row["middle_mixed_foot_steps"] / row["middle_steps"]
                        for row in valid_arrivals if row["middle_steps"] > 0
                    )
                    details = (f" center_fraction={mean_center_fraction:.2f}"
                               f" mixed_feet_fraction={mean_mixed_fraction:.2f}")
                print(f"[GUIDED-SUMMARY] route={route} arrivals={sum(row['success'] for row in selected)}"
                      f"/{len(selected)} valid_arrivals={len(valid_arrivals)}"
                      f" mean_energy={mean_energy:.2f}J mean_path={mean_distance:.3f}m{details}", flush=True)
            else:
                print(f"[GUIDED-SUMMARY] route={route} no valid arrivals; "
                      "no energy comparison possible", flush=True)
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

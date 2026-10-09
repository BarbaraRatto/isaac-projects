"""Measure Spot's mechanical energy on the three user-drawn guided routes.

This is an evaluation with prescribed XY centerlines. It does not train a
policy or use DINO. Only successful trials that follow their prescribed route
are included in the energy summary.
"""

import argparse
import bisect
import csv
import math
import statistics
import sys
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=3, help="trials per route")
parser.add_argument("--speed", type=float, default=0.8, help="common forward speed cap, m/s")
parser.add_argument("--lookahead", type=float, default=0.8, help="path tracking lookahead, m")
parser.add_argument("--routes", nargs="+", choices=("pink", "orange", "yellow"),
                    default=("pink", "orange", "yellow"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1 or not 0.0 < args.speed <= 1.6 or not 0.2 <= args.lookahead <= 2.0:
    parser.error("Require episodes >= 1, speed in (0, 1.6], lookahead in [0.2, 2.0]")
args.enable_cameras = False
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from full_grid_energy_choice.scene import FullGridChoiceCfg  # noqa: E402
from medium_distance_choice.env import MediumDistanceEnv  # noqa: E402
from medium_distance_choice.task_config import (  # noqa: E402
    GEOMETRY_MEDIUM, NAV_CFG_MEDIUM, START_XY, START_YAW_RAD,
    TERRAIN_LABELS, read_terrain_codes,
)
from medium_distance_choice.three_route_plan import GOAL, route_points  # noqa: E402
from rl_config import validate_assets  # noqa: E402


def number(info, key):
    value = info[key]
    return float(value.item()) if hasattr(value, "item") else float(value)


def path_lengths(points):
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.dist(a, b))
    return lengths


def nearest_on_path(points, lengths, xy, minimum_s=0.0):
    """Return along-path distance and cross-track error at the closest point."""
    best = (float("inf"), 0.0)
    for index, (a, b) in enumerate(zip(points, points[1:])):
        if lengths[index + 1] < minimum_s - 0.6:
            continue
        vx, vy = b[0] - a[0], b[1] - a[1]
        segment_sq = vx * vx + vy * vy
        if segment_sq == 0.0:
            continue
        fraction = max(0.0, min(1.0,
                       ((xy[0] - a[0]) * vx + (xy[1] - a[1]) * vy) / segment_sq))
        px, py = a[0] + fraction * vx, a[1] + fraction * vy
        error = math.hypot(xy[0] - px, xy[1] - py)
        along = lengths[index] + fraction * (lengths[index + 1] - lengths[index])
        if error < best[0]:
            best = (error, along)
    if not math.isfinite(best[0]):
        raise RuntimeError("Could not project Spot onto the planned route")
    return best[1], best[0]


def point_at_distance(points, lengths, along):
    if along >= lengths[-1]:
        return points[-1]
    index = max(0, bisect.bisect_right(lengths, along) - 1)
    if index >= len(points) - 1:
        return points[-1]
    span = lengths[index + 1] - lengths[index]
    t = (along - lengths[index]) / span if span else 0.0
    a, b = points[index], points[index + 1]
    return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])


def command(env, xy, target):
    q = env.robot.data.root_quat_w[0].detach().cpu().tolist()
    yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                     1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))
    heading = math.atan2(target[1] - xy[1], target[0] - xy[0])
    error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
    turn = float(np.clip(2.0 * error, -env.nav.max_yaw_rad_s, env.nav.max_yaw_rad_s))
    forward = 0.0 if abs(error) > 1.0 else args.speed * max(0.0, math.cos(error))
    return np.array([[forward / env.nav.max_forward_m_s, 0.0,
                      turn / env.nav.max_yaw_rad_s]], dtype=np.float32)


def main():
    validate_assets()
    read_terrain_codes()
    if math.dist(START_XY, GOAL) < 1.0:
        raise RuntimeError("Invalid start/goal")
    geometry = replace(GEOMETRY_MEDIUM, start_jitter_m=0.0)
    cfg = FullGridChoiceCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.episode_length_s = NAV_CFG_MEDIUM.max_episode_seconds
    cfg.robot_cfg.init_state.pos = (*geometry.start_xy, NAV_CFG_MEDIUM.spawn_height_m)
    half = START_YAW_RAD / 2.0
    cfg.robot_cfg.init_state.rot = (math.cos(half), 0.0, 0.0, math.sin(half))
    cfg.viewer.eye = (31.0, 24.0, 23.0)
    cfg.viewer.lookat = (12.0, 9.0, 0.0)
    env = MediumDistanceEnv(cfg, atlas_file=None, geometry=geometry)
    try:
        wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
        rows = []
        tracks = []
        for route_name in args.routes:
            points = route_points(route_name)
            lengths = path_lengths(points)
            for episode in range(1, args.episodes + 1):
                wrapped.reset()
                progress = 0.0
                cross_errors = []
                body_points = []
                within_one_m = 0
                steps = 0
                start_xy = (env.robot.data.root_pos_w[0, :2]
                            - env.scene.env_origins[0, :2]).detach().cpu().tolist()
                tracks.append({"route": route_name, "episode": episode, "decision": 0,
                               "time_s": 0.0, "x_m": round(start_xy[0], 5),
                               "y_m": round(start_xy[1], 5), "energy_j": 0.0,
                               "along_path_m": 0.0, "cross_track_m": 0.0})
                for steps in range(1, env.max_episode_length + 2):
                    xy = (env.robot.data.root_pos_w[0, :2]
                          - env.scene.env_origins[0, :2]).detach().cpu().tolist()
                    closest_s, _ = nearest_on_path(points, lengths, xy, progress)
                    progress = max(progress, closest_s)
                    target = point_at_distance(points, lengths, progress + args.lookahead)
                    action = command(env, xy, target)
                    _, _, dones, infos = wrapped.step(action)
                    info = infos[0]
                    # The terminal position in info was captured before auto-reset.
                    final_xy = (number(info, "body_x_m"), number(info, "body_y_m"))
                    closest_s, cross_error = nearest_on_path(points, lengths, final_xy, progress)
                    progress = max(progress, closest_s)
                    cross_errors.append(cross_error)
                    body_points.append(final_xy)
                    within_one_m += int(cross_error <= 1.0)
                    tracks.append({"route": route_name, "episode": episode,
                                   "decision": steps, "time_s": round(steps * env.step_dt, 3),
                                   "x_m": round(final_xy[0], 5), "y_m": round(final_xy[1], 5),
                                   "energy_j": round(number(info, "episode_energy_j"), 3),
                                   "along_path_m": round(progress, 3),
                                   "cross_track_m": round(cross_error, 3)})
                    if bool(dones[0]):
                        break
                success = bool(number(info, "is_success"))
                followed = (success and progress >= lengths[-1] - 0.9
                            and within_one_m / steps >= 0.85
                            and max(cross_errors) <= 1.7)
                seam_points = [(x, y) for x, y in body_points if 9.0 <= x <= 15.0]
                seam_max_error = (max(abs(y - 10.5) for _, y in seam_points)
                                  if seam_points else float("inf"))
                obstacle_center_error = (max(min(math.dist(point, center)
                                                 for point in body_points)
                                             for center in ((18.0, 9.0), (6.0, 12.0)))
                                         if body_points else float("inf"))
                if route_name == "pink":
                    followed = (followed and bool(seam_points)
                                and seam_max_error <= 0.5
                                and obstacle_center_error <= 1.0
                                and not number(info, "ice_entry"))
                if route_name == "orange":
                    followed = (followed
                                and not any(number(info, f"{label}_entry")
                                            for label in ("ice", "stairs", "obstacles")))
                row = {
                    "route": route_name, "episode": episode, "success": int(success),
                    "route_followed": int(followed),
                    "reason": "goal" if success else
                              ("timeout" if info.get("TimeLimit.truncated") else "unsafe"),
                    "speed_cap_m_s": args.speed, "lookahead_m": args.lookahead,
                    "planned_length_m": round(lengths[-1], 3),
                    "energy_j": round(number(info, "episode_energy_j"), 2),
                    "path_length_m": round(number(info, "path_length_m"), 3),
                    "final_distance_m": round(number(info, "final_distance_m"), 3),
                    "decisions": steps,
                    "mean_cross_track_m": round(statistics.fmean(cross_errors), 3),
                    "max_cross_track_m": round(max(cross_errors), 3),
                    "fraction_within_1m": round(within_one_m / steps, 3),
                    "pink_seam_max_error_m": (round(seam_max_error, 3)
                                              if route_name == "pink" else ""),
                    "pink_obstacle_center_max_error_m": (round(obstacle_center_error, 3)
                                                         if route_name == "pink" else ""),
                }
                row.update({f"{label}_entry": int(number(info, f"{label}_entry"))
                            for label in TERRAIN_LABELS})
                rows.append(row)
                print(f"[THREE-ROUTES] {row}", flush=True)
        output = args.output or HERE / "results" / f"guided_three_routes_{datetime.now():%Y%m%d_%H%M%S}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        track_output = output.with_name(output.stem + "_trajectory.csv")
        with track_output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(tracks[0]))
            writer.writeheader()
            writer.writerows(tracks)
        print(f"[THREE-ROUTES] summary={output.resolve()}", flush=True)
        print(f"[THREE-ROUTES] trajectory={track_output.resolve()}", flush=True)
        for route_name in args.routes:
            selected = [row for row in rows if row["route"] == route_name]
            valid = [row for row in selected if row["route_followed"]]
            if valid:
                print(f"[THREE-ROUTES-SUMMARY] route={route_name} "
                      f"valid_arrivals={len(valid)}/{len(selected)} "
                      f"mean_energy={statistics.fmean(row['energy_j'] for row in valid):.2f}J "
                      f"mean_path={statistics.fmean(row['path_length_m'] for row in valid):.3f}m",
                      flush=True)
            else:
                print(f"[THREE-ROUTES-SUMMARY] route={route_name} "
                      "no valid arrivals; energy comparison unavailable", flush=True)
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

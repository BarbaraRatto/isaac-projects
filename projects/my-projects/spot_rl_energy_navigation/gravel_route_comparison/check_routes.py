"""Guided energy comparison on the complete USD; this does not train an RL policy."""

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import math
from pathlib import Path
import statistics
import sys
import traceback

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=10, help="trials per route")
parser.add_argument("--speed", type=float, default=0.8, help="maximum commanded forward speed in m/s")
parser.add_argument("--routes", nargs="+", choices=("direct", "avoid", "partial"),
                    default=("direct", "avoid", "partial"))
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1 or not 0.0 < args.speed <= 1.6:
    parser.error("Require episodes >= 1 and speed in (0, 1.6] m/s")
if len(set(args.routes)) != len(args.routes):
    parser.error("Each route can be specified only once")
args.enable_cameras = False
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from full_grid_energy_choice.config import GEOMETRY, NAV_CFG_FULL  # noqa: E402
from full_grid_energy_choice.env import FullGridChoiceEnv  # noqa: E402
from full_grid_energy_choice.scene import FullGridChoiceCfg  # noqa: E402
from rl_config import TERRAIN_USD, validate_assets  # noqa: E402

START = (26.0, 15.0)  # Cell_5_4_t1_asphalt
GOAL = (28.5, 9.0)   # Cell_3_5_t3_ramp
GRAVEL_X = (27.0, 33.0)  # Cell_4_5_t4_fine_gravel
GRAVEL_Y = (10.5, 13.5)
ROUTES = {
    "direct": (),
    # Remain in row-4 ramp (x<27); enter row-3 asphalt before turning to goal.
    "avoid": ((26.4, 10.3),),
    # Enter gravel near its lower edge, then descend into row-3 ramp.
    "partial": ((26.4, 11.4), (27.6, 10.9)),
}
EXPECTED_CELLS = {
    "Cell_5_4_t1_asphalt": ((21.0, 27.0), (13.5, 16.5)),
    "Cell_4_4_t3_ramp": ((21.0, 27.0), (10.5, 13.5)),
    "Cell_4_5_t4_fine_gravel": ((27.0, 33.0), (10.5, 13.5)),
    "Cell_3_4_t1_asphalt": ((21.0, 27.0), (7.5, 10.5)),
    "Cell_3_5_t3_ramp": ((27.0, 33.0), (7.5, 10.5)),
}


def validate_cells():
    stage = Usd.Stage.Open(str(TERRAIN_USD))
    if stage is None:
        raise RuntimeError(f"Could not open terrain USD: {TERRAIN_USD}")
    box_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for name, (expected_x, expected_y) in EXPECTED_CELLS.items():
        prim = stage.GetPrimAtPath(f"/World/Grid/{name}")
        if not prim.IsValid():
            raise RuntimeError(f"The full terrain USD lacks the expected cell: {name}")
        box = box_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        lo, hi = box.GetMin(), box.GetMax()
        if any(abs(actual - expected) > 0.1 for actual, expected in
               ((lo[0], expected_x[0]), (hi[0], expected_x[1]),
                (lo[1], expected_y[0]), (hi[1], expected_y[1]))):
            raise RuntimeError(f"Unexpected bounds for {name}: {tuple(lo)}..{tuple(hi)}")
    for label, point, cell in (("start", START, "Cell_5_4_t1_asphalt"),
                               ("goal", GOAL, "Cell_3_5_t3_ramp")):
        xb, yb = EXPECTED_CELLS[cell]
        if not (xb[0] + 0.5 <= point[0] <= xb[1] - 0.5
                and yb[0] + 0.5 <= point[1] <= yb[1] - 0.5):
            raise RuntimeError(f"{label}={point} is too close to the edge of {cell}")


def number(info, key):
    value = info[key]
    return float(value.item()) if hasattr(value, "item") else float(value)


def local_xy(env):
    return (env.robot.data.root_pos_w[0, :2]
            - env.scene.env_origins[0, :2]).detach().cpu().numpy()


def guided_action(env, target_xy, final_target):
    xy = local_xy(env)
    q = env.robot.data.root_quat_w[0].detach().cpu().numpy()
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                     1 - 2 * (q[2] * q[2] + q[3] * q[3]))
    dx, dy = target_xy[0] - float(xy[0]), target_xy[1] - float(xy[1])
    distance = math.hypot(dx, dy)
    error = math.atan2(math.sin(math.atan2(dy, dx) - yaw),
                       math.cos(math.atan2(dy, dx) - yaw))
    turn = float(np.clip(2.0 * error, -NAV_CFG_FULL.max_yaw_rad_s,
                         NAV_CFG_FULL.max_yaw_rad_s))
    forward = args.speed if final_target else min(args.speed, max(0.4, distance))
    forward *= max(0.0, math.cos(error))
    return np.array([[forward / NAV_CFG_FULL.max_forward_m_s, 0.0,
                      turn / NAV_CFG_FULL.max_yaw_rad_s]], dtype=np.float32)


def main():
    validate_assets()
    validate_cells()
    cfg = FullGridChoiceCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.viewer.eye = (34.0, 20.0, 15.0)
    cfg.viewer.lookat = (27.0, 12.0, 0.0)
    yaw = math.atan2(GOAL[1] - START[1], GOAL[0] - START[0])
    geometry = replace(GEOMETRY, start_xy=START, goal_xy=GOAL,
                       start_yaw_rad=yaw, start_jitter_m=0.0)
    cfg.robot_cfg.init_state.pos = (*START, NAV_CFG_FULL.spawn_height_m)
    cfg.robot_cfg.init_state.rot = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    env = FullGridChoiceEnv(cfg, atlas_file=None, geometry=geometry)
    try:
        wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
        wrapped.reset()
        rows = []
        for route in args.routes:
            targets = ROUTES[route] + (GOAL,)
            for episode in range(1, args.episodes + 1):
                waypoint_index = 0
                actions = 0
                gravel_body_m = 0.0
                ramp_body_m = 0.0
                asphalt_body_m = 0.0
                while True:
                    before = local_xy(env)
                    if (waypoint_index < len(targets) - 1
                            and math.dist(before, targets[waypoint_index]) < 0.25):
                        waypoint_index += 1
                    _, _, dones, infos = wrapped.step(
                        guided_action(env, targets[waypoint_index],
                                      waypoint_index == len(targets) - 1))
                    actions += 1
                    if bool(dones[0]):
                        # SB3 has already reset Spot; never count that reset as travel.
                        break
                    after = local_xy(env)
                    midpoint = (before + after) / 2.0
                    segment_m = math.dist(before, after)
                    if GRAVEL_Y[0] <= midpoint[1] < GRAVEL_Y[1]:
                        if GRAVEL_X[0] <= midpoint[0] < GRAVEL_X[1]:
                            gravel_body_m += segment_m
                        elif 21.0 <= midpoint[0] < 27.0:
                            ramp_body_m += segment_m
                    elif 7.5 <= midpoint[1] < 10.5 and 21.0 <= midpoint[0] < 27.0:
                        asphalt_body_m += segment_m
                info = infos[0]
                success = int(number(info, "is_success"))
                followed_waypoints = waypoint_index == len(ROUTES[route])
                if route == "direct":
                    route_followed = followed_waypoints and gravel_body_m >= 1.0
                elif route == "avoid":
                    route_followed = followed_waypoints and gravel_body_m <= 0.15
                else:
                    route_followed = followed_waypoints and 0.2 <= gravel_body_m <= 1.6
                row = {
                    "route": route,
                    "episode": episode,
                    "success": success,
                    "route_followed": int(route_followed),
                    "reason": "goal" if success else ("timeout" if info.get("TimeLimit.truncated") else "unsafe"),
                    "start_x_m": START[0], "start_y_m": START[1],
                    "goal_x_m": GOAL[0], "goal_y_m": GOAL[1],
                    "speed_cap_m_s": args.speed,
                    "energy_j": round(number(info, "episode_energy_j"), 2),
                    "path_length_m": round(number(info, "path_length_m"), 3),
                    "gravel_body_m": round(gravel_body_m, 3),
                    "row4_ramp_body_m": round(ramp_body_m, 3),
                    "row3_asphalt_body_m": round(asphalt_body_m, 3),
                    "final_distance_m": round(number(info, "final_distance_m"), 3),
                    "actions": actions,
                }
                rows.append(row)
                print(f"[GRAVEL-ROUTE] {row}", flush=True)
        output = args.output or HERE / "results" / f"gravel_routes_{datetime.now():%Y%m%d_%H%M%S}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[GRAVEL-ROUTE] saved={output.resolve()}", flush=True)
        for route in args.routes:
            selected = [row for row in rows if row["route"] == route]
            valid = [row for row in selected if row["success"] and row["route_followed"]]
            if not valid:
                print(f"[GRAVEL-SUMMARY] route={route} valid_arrivals=0/{len(selected)}; "
                      "no valid energy comparison", flush=True)
                continue
            print(f"[GRAVEL-SUMMARY] route={route} valid_arrivals={len(valid)}/{len(selected)}"
                  f" mean_energy={statistics.fmean(r['energy_j'] for r in valid):.2f}J"
                  f" mean_path={statistics.fmean(r['path_length_m'] for r in valid):.3f}m"
                  f" mean_gravel_body={statistics.fmean(r['gravel_body_m'] for r in valid):.3f}m"
                  f" mean_final_distance={statistics.fmean(r['final_distance_m'] for r in valid):.3f}m",
                  flush=True)
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

"""Measure complete direct and bypass routes on the controlled two-lane course.

This is an empirical oracle for evaluation, never an observation or label for
Spot's future RL policy. It uses the frozen IsaacRobotics joint controller.
"""

import argparse
import csv
import math
import os
import sys
import traceback
from statistics import mean
from time import perf_counter

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Measure complete-route mechanical work.")
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--speeds", type=float, nargs="+", default=(0.6, 0.9),
                    help="Maximum commanded forward speeds in m/s.")
parser.add_argument("--max-seconds", type=float, default=None)
parser.add_argument("--direct-speed", type=float, default=None,
                    help="For a matched comparison, direct-route speed in m/s.")
parser.add_argument("--bypass-speed", type=float, default=None,
                    help="For a matched comparison, bypass-route speed in m/s.")
parser.add_argument("--routes", nargs="+", choices=("direct", "bypass"),
                    default=("direct", "bypass"))
parser.add_argument("--output", type=str, default=None, help="CSV path, default results/route_baselines.csv")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
chosen_speeds = [*args.speeds, *(v for v in (args.direct_speed, args.bypass_speed) if v is not None)]
if args.repeats < 1 or any(speed <= 0 or speed > 0.9 for speed in chosen_speeds):
    parser.error("--repeats must be positive and each speed must be in (0, 0.9]")
if (args.direct_speed is None) != (args.bypass_speed is None):
    parser.error("--direct-speed and --bypass-speed must be given together")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat  # noqa: E402
from pxr import UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

from locomotion import SpotVelocityController, create_spot, reset_spot  # noqa: E402
from rl_config import PHYSICS_DT, validate_assets  # noqa: E402
from route_config import ROUTE_CFG  # noqa: E402
from two_route_scene import build_two_route_scene  # noqa: E402


FIELDS = (
    "direct_terrain", "route", "trial", "lateral_offset_m", "speed_cap_m_s", "status", "walk_s", "distance_m",
    "displacement_m", "energy_j", "energy_j_per_m", "mean_speed_m_s",
    "goal_distance_m", "reached_waypoints", "start_x_m", "start_y_m",
    "end_x_m", "end_y_m", "end_z_m",
)
VIEW_CAMERA_PATH = "/World/RouteComparisonCamera"


def step_robot(sim, robot, controller, command: torch.Tensor) -> None:
    if not simulation_app.is_running():
        raise RuntimeError("Isaac Sim was closed during the route comparison")
    controller.forward(command.view(1, 3))
    sim.step(render=not args.headless)
    robot.update(PHYSICS_DT)


def power_w(robot) -> float:
    effort = robot.root_physx_view.get_dof_projected_joint_forces()
    speed = robot.data.joint_vel
    if effort.shape != speed.shape or speed.shape != (1, robot.num_joints):
        raise RuntimeError("Measured joint efforts and velocities do not match")
    power = torch.sum(torch.abs(effort * speed))
    if not bool(torch.isfinite(power)):
        raise RuntimeError("Non-finite measured joint power")
    return float(power.item())


def velocity_to_waypoint(robot, waypoint: tuple[float, float], speed_cap: float) -> torch.Tensor:
    data = robot.data
    pos = data.root_pos_w[0]
    delta_w = torch.tensor((waypoint[0] - float(pos[0]), waypoint[1] - float(pos[1]), 0.0),
                           device=pos.device, dtype=pos.dtype)
    delta_b = quat_apply_inverse(yaw_quat(data.root_quat_w), delta_w.view(1, 3))[0]
    forward_w = quat_apply(yaw_quat(data.root_quat_w),
                           torch.tensor([[1.0, 0.0, 0.0]], device=pos.device))[0]
    heading = math.atan2(float(forward_w[1]), float(forward_w[0]))
    vx = max(-0.15, min(speed_cap, 0.8 * float(delta_b[0])))
    vy = max(-ROUTE_CFG.max_lateral_m_s,
             min(ROUTE_CFG.max_lateral_m_s, 0.8 * float(delta_b[1])))
    yaw_rate = max(-ROUTE_CFG.max_yaw_rad_s,
                   min(ROUTE_CFG.max_yaw_rad_s, -1.5 * heading))
    return torch.tensor((vx, vy, yaw_rate), device=pos.device)


def run_trial(sim, robot, controller, route: str, trial: int, speed_cap: float,
              max_seconds: float) -> dict:
    lateral_offset = (0.0, -0.15, 0.15)[(trial - 1) % 3]
    start = (ROUTE_CFG.start_xyz[0], ROUTE_CFG.start_xyz[1] + lateral_offset,
             ROUTE_CFG.start_xyz[2])
    reset_spot(robot, start)
    controller.reset()
    robot.update(PHYSICS_DT)
    zero = torch.zeros(3, device=sim.device)
    for _ in range(round(ROUTE_CFG.settle_seconds / PHYSICS_DT)):
        step_robot(sim, robot, controller, zero)
    start_xy = robot.data.root_pos_w[0, :2].clone()
    prev_xy = start_xy.clone()
    prev_power = power_w(robot)
    energy_j = 0.0
    distance_m = 0.0
    waypoints = ROUTE_CFG.waypoints(route)
    waypoint_index = 0
    status = "timeout"
    steps = 0
    for steps in range(1, round(max_seconds / PHYSICS_DT) + 1):
        pos = robot.data.root_pos_w[0]
        while waypoint_index < len(waypoints) - 1:
            target = waypoints[waypoint_index]
            if math.hypot(float(pos[0]) - target[0], float(pos[1]) - target[1]) > ROUTE_CFG.waypoint_radius_m:
                break
            waypoint_index += 1
        command = velocity_to_waypoint(robot, waypoints[waypoint_index], speed_cap)
        step_robot(sim, robot, controller, command)
        pos = robot.data.root_pos_w[0]
        distance_m += float(torch.linalg.vector_norm(pos[:2] - prev_xy).item())
        prev_xy = pos[:2].clone()
        current_power = power_w(robot)
        energy_j += 0.5 * (prev_power + current_power) * PHYSICS_DT
        prev_power = current_power
        x, y, z = (float(v) for v in pos[:3])
        goal_distance = math.hypot(x - ROUTE_CFG.goal_xy[0], y - ROUTE_CFG.goal_xy[1])
        if z < ROUTE_CFG.min_body_height_m or float(robot.data.projected_gravity_b[0, 2]) > -0.5:
            status = "fall"
            break
        if not (-0.8 <= x <= 36.8 and -1.8 <= y <= 4.8):
            status = "left_course"
            break
        if goal_distance <= ROUTE_CFG.goal_radius_m:
            status = "ok"
            break
    end = robot.data.root_pos_w[0].detach().cpu().tolist()
    elapsed = steps * PHYSICS_DT
    net_distance = float(torch.linalg.vector_norm(robot.data.root_pos_w[0, :2] - start_xy).item())
    goal_distance = math.hypot(end[0] - ROUTE_CFG.goal_xy[0], end[1] - ROUTE_CFG.goal_xy[1])
    result = {
        "direct_terrain": "t6_obstacles",
        "route": route,
        "trial": trial,
        "lateral_offset_m": lateral_offset,
        "speed_cap_m_s": speed_cap,
        "status": status,
        "walk_s": round(elapsed, 4),
        "distance_m": round(distance_m, 4),
        "displacement_m": round(net_distance, 4),
        "energy_j": round(energy_j, 4),
        "energy_j_per_m": round(energy_j / distance_m, 4) if distance_m else "",
        "mean_speed_m_s": round(distance_m / elapsed, 4) if elapsed else "",
        "goal_distance_m": round(goal_distance, 4),
        "reached_waypoints": waypoint_index,
        "start_x_m": round(float(start_xy[0]), 4),
        "start_y_m": round(float(start_xy[1]), 4),
        "end_x_m": round(end[0], 4),
        "end_y_m": round(end[1], 4),
        "end_z_m": round(end[2], 4),
    }
    return result


def main() -> None:
    validate_assets()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device))
    original_bounds = build_two_route_scene()
    if not args.headless:
        UsdGeom.Camera.Define(omni.usd.get_context().get_stage(), VIEW_CAMERA_PATH)
    robot = create_spot()
    controller = SpotVelocityController(robot, sim.device)
    sim.reset()
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = VIEW_CAMERA_PATH
        sim.set_camera_view((18.0, 18.0, 13.0), (18.0, 1.0, 0.0), camera_prim_path=VIEW_CAMERA_PATH)
    print(f"[ROUTES] Source bounds: {original_bounds}", flush=True)
    direct_length = ROUTE_CFG.nominal_length("direct")
    bypass_length = ROUTE_CFG.nominal_length("bypass")
    print(f"[ROUTES] direct={direct_length:.2f}m nominal; bypass={bypass_length:.2f}m "
          f"(+{100 * (bypass_length / direct_length - 1):.1f}%)", flush=True)
    print("[ROUTES] Energy excludes 2s initial settling, includes all steering to goal.", flush=True)
    rows = []
    started = perf_counter()
    max_seconds = args.max_seconds or ROUTE_CFG.max_walk_seconds
    jobs = ([("direct", args.direct_speed), ("bypass", args.bypass_speed)]
            if args.direct_speed is not None else
            [(route, speed) for speed in args.speeds for route in args.routes])
    for trial in range(1, args.repeats + 1):
        for route, speed in jobs:
            row = run_trial(sim, robot, controller, route, trial, speed, max_seconds)
            rows.append(row)
            print(f"[ROUTES] {route} speed={speed:.2f} trial={trial} "
                  f"status={row['status']} walked={row['distance_m']:.2f}m "
                  f"goal_error={row['goal_distance_m']:.2f}m "
                  f"energy={row['energy_j']:.1f}J", flush=True)
    from pathlib import Path
    csv_path = Path(args.output).resolve() if args.output else ROUTE_CFG.csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pending = csv_path.with_suffix(".csv.tmp")
    try:
        with pending.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(pending, csv_path)
    finally:
        pending.unlink(missing_ok=True)
    print(f"[ROUTES] CSV: {csv_path}", flush=True)
    for route, speed in jobs:
        completed = [r for r in rows if r["route"] == route and r["speed_cap_m_s"] == speed and r["status"] == "ok"]
        if completed:
            print(f"[ROUTES] {route} speed={speed:.2f}: {len(completed)}/{args.repeats} "
                  f"completed, mean energy={mean(r['energy_j'] for r in completed):.1f}J", flush=True)
    print(f"[ROUTES] Wall time: {perf_counter() - started:.1f}s", flush=True)
    sim.clear_instance()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        if args.headless:
            sys.stderr.flush()
            os._exit(1)
        simulation_app.close()
        raise
    else:
        simulation_app.close(skip_cleanup=args.headless)

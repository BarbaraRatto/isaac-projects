"""Compare mechanical walking energy across the complete terrain USD and several speeds.

Uses the existing Spot controller. This is not RL training and does not use DINO.
"""

import argparse
import csv
import math
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


parser = argparse.ArgumentParser(description="Compare measured Spot energy at multiple speeds on the complete terrain USD.")
parser.add_argument("--speeds", nargs="+", type=float, default=(1.1, 1.3, 1.6), help="Commanded forward speeds in m/s; default tests only new speeds above 0.9.")
parser.add_argument("--terrains", nargs="+", default=None, help="Subset of terrain names (default: all seven).")
parser.add_argument("--repeats", type=int, default=3, help="Trials per terrain and speed.")
parser.add_argument("--distance", type=float, default=2.0, help="Measured walking distance per trial in meters.")
parser.add_argument("--output", type=Path, default=None, help="Trial CSV path; default is timestamped in results/.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

from locomotion import SpotVelocityController, create_spot, reset_spot  # noqa: E402
from rl_config import (  # noqa: E402
    BENCHMARK_EDGE_MARGIN_M,
    BENCHMARK_FOOT_MARGIN_M,
    BENCHMARK_MIN_BODY_HEIGHT_M,
    BENCHMARK_SETTLE_SECONDS,
    BENCHMARK_SPAWN_HEIGHT_M,
    BENCHMARK_START_BACK_M,
    PHYSICS_DT,
    STOP_SECONDS,
    validate_assets,
)
from scene import build_scene  # noqa: E402


VIEW_CAMERA_PATH = "/World/BenchmarkCamera"
ALL_TERRAINS = ("t1_asphalt", "t2_slippery", "t3_ramp", "t4_fine_gravel",
                "t5_large_rocks", "t6_obstacles", "t7_stairs")

FIELDS = (
    "terrain", "cell_prim", "trial", "status", "command_vx_m_s", "settle_s",
    "walk_s", "target_distance_m", "lateral_offset_m", "settle_speed_m_s",
    "distance_m", "displacement_m", "estimated_energy_j", "estimated_energy_j_per_m",
    "measured_energy_j", "measured_energy_j_per_m", "mean_speed_m_s",
    "estimated_mean_power_w", "measured_mean_power_w",
    "estimated_mean_abs_torque_nm", "measured_mean_abs_torque_nm",
    "start_x_m", "start_y_m",
    "end_x_m", "end_y_m", "end_z_m",
)


def find_terrain_cells(terrains: tuple[str, ...]) -> dict[str, dict]:
    """Find one fixed cell for each requested terrain in the loaded full USD."""
    stage = omni.usd.get_context().get_stage()
    cells = []
    for prim in stage.Traverse():
        match = re.fullmatch(r"Cell_(\d+)_(\d+)_(.+)", prim.GetName())
        if match and match.group(3) in terrains:
            cells.append((int(match.group(1)), int(match.group(2)), match.group(3), prim))
    missing = sorted(set(terrains) - {item[2] for item in cells})
    if missing:
        raise RuntimeError(f"Terrain cells missing from the loaded USD: {missing}")
    selected = [min((item for item in cells if item[2] == kind), key=lambda x: (x[0], x[1]))
                for kind in terrains]

    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    result = {}
    for _row, _col, kind, prim in selected:
        box = bounds.ComputeWorldBound(prim).ComputeAlignedBox()
        low = tuple(float(value) for value in box.GetMin())
        high = tuple(float(value) for value in box.GetMax())
        result[kind] = {"path": str(prim.GetPath()), "low": low, "high": high}
    return result


def step_robot(sim, robot, controller, command: torch.Tensor) -> None:
    controller.forward(command)
    sim.step(render=not args.headless)
    robot.update(PHYSICS_DT)


def instantaneous_metrics(robot) -> tuple[float, float, float, float]:
    """Return power and mean absolute torque from the same joint samples."""
    estimated_effort = robot.data.applied_torque
    measured_effort = robot.root_physx_view.get_dof_projected_joint_forces()
    speed = robot.data.joint_vel
    if (estimated_effort is None or measured_effort is None or speed is None
            or estimated_effort.shape != speed.shape or measured_effort.shape != speed.shape
            or speed.shape != (1, robot.num_joints)):
        raise RuntimeError("Missing or inconsistent estimated/measured effort and joint velocity samples")
    estimated_power = torch.sum(torch.abs(estimated_effort * speed))
    measured_power = torch.sum(torch.abs(measured_effort * speed))
    if not bool(torch.isfinite(estimated_power) and torch.isfinite(measured_power)):
        raise RuntimeError("Non-finite estimated or measured joint power")
    return (float(estimated_power.item()), float(measured_power.item()),
            float(estimated_effort.abs().mean().item()), float(measured_effort.abs().mean().item()))


def run_trial(sim, robot, controller, terrain: str, cell: dict, trial: int, target_m: float,
              lateral_offset_m: float, command_vx: float) -> dict:
    low, high = cell["low"], cell["high"]
    center_x = (low[0] + high[0]) / 2.0
    center_y = (low[1] + high[1]) / 2.0
    start_position = (center_x - BENCHMARK_START_BACK_M, center_y + lateral_offset_m,
                      high[2] + BENCHMARK_SPAWN_HEIGHT_M)
    reset_spot(robot, start_position)
    controller.reset()
    robot.update(PHYSICS_DT)
    if not args.headless:
        sim.set_camera_view((center_x + 2.5, center_y + 4.5, 3.5),
                            (center_x, center_y, 0.0), camera_prim_path=VIEW_CAMERA_PATH)

    zero = torch.zeros((1, 3), device=sim.device)
    forward = torch.tensor([[command_vx, 0.0, 0.0]], device=sim.device, dtype=torch.float32)
    for _ in range(round(BENCHMARK_SETTLE_SECONDS / PHYSICS_DT)):
        if not simulation_app.is_running():
            raise RuntimeError("Simulation closed during the settling period")
        step_robot(sim, robot, controller, zero)

    start_xy = robot.data.root_pos_w[0, :2].clone()
    settle_speed = float(torch.linalg.vector_norm(robot.data.root_lin_vel_b[0, :2]).item())
    previous_xy = start_xy.clone()
    previous_estimated_power, previous_measured_power, _, _ = instantaneous_metrics(robot)
    foot_ids = [index for index, name in enumerate(robot.body_names) if name.endswith("_foot")]
    if len(foot_ids) != 4:
        raise RuntimeError(f"Expected four Spot feet, found: {foot_ids}")
    walked_m = 0.0
    estimated_energy_j = 0.0
    measured_energy_j = 0.0
    estimated_abs_torque_sum = 0.0
    measured_abs_torque_sum = 0.0
    status = "timeout"
    steps = 0
    max_walk_s = max(10.0, 3.0 * target_m / command_vx)
    for steps in range(1, round(max_walk_s / PHYSICS_DT) + 1):
        if not simulation_app.is_running():
            raise RuntimeError("Simulation closed during walking")
        step_robot(sim, robot, controller, forward)
        position = robot.data.root_pos_w[0].clone()
        walked_m += float(torch.linalg.vector_norm(position[:2] - previous_xy).item())
        previous_xy = position[:2].clone()
        estimated_power, measured_power, estimated_abs_torque, measured_abs_torque = instantaneous_metrics(robot)
        estimated_abs_torque_sum += estimated_abs_torque
        measured_abs_torque_sum += measured_abs_torque
        estimated_energy_j += 0.5 * (previous_estimated_power + estimated_power) * PHYSICS_DT
        measured_energy_j += 0.5 * (previous_measured_power + measured_power) * PHYSICS_DT
        previous_estimated_power = estimated_power
        previous_measured_power = measured_power

        x, y, z = (float(value) for value in position[:3])
        if z < high[2] + BENCHMARK_MIN_BODY_HEIGHT_M or float(robot.data.projected_gravity_b[0, 2]) > -0.5:
            status = "fall"
            break
        margin = BENCHMARK_EDGE_MARGIN_M
        if not (low[0] + margin <= x <= high[0] - margin and low[1] + margin <= y <= high[1] - margin):
            status = "left_cell"
            break
        feet_xy = robot.data.body_pos_w[0, foot_ids, :2]
        foot_margin = BENCHMARK_FOOT_MARGIN_M
        if bool(((feet_xy[:, 0] < low[0] + foot_margin) | (feet_xy[:, 0] > high[0] - foot_margin)
                 | (feet_xy[:, 1] < low[1] + foot_margin) | (feet_xy[:, 1] > high[1] - foot_margin)).any()):
            status = "foot_left_cell"
            break
        if walked_m >= target_m:
            status = "ok"
            break

    end_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
    net_m = float(torch.linalg.vector_norm(robot.data.root_pos_w[0, :2] - start_xy).item())
    walk_s = steps * PHYSICS_DT
    row = {
        "terrain": terrain,
        "cell_prim": cell["path"],
        "trial": trial,
        "status": status,
        "command_vx_m_s": command_vx,
        "settle_s": BENCHMARK_SETTLE_SECONDS,
        "walk_s": round(walk_s, 4),
        "target_distance_m": target_m,
        "lateral_offset_m": round(lateral_offset_m, 4),
        "settle_speed_m_s": round(settle_speed, 4),
        "distance_m": round(walked_m, 4),
        "displacement_m": round(net_m, 4),
        "estimated_energy_j": round(estimated_energy_j, 4),
        "estimated_energy_j_per_m": round(estimated_energy_j / walked_m, 4) if walked_m > 0 else "",
        "measured_energy_j": round(measured_energy_j, 4),
        "measured_energy_j_per_m": round(measured_energy_j / walked_m, 4) if walked_m > 0 else "",
        "mean_speed_m_s": round(walked_m / walk_s, 4),
        "estimated_mean_power_w": round(estimated_energy_j / walk_s, 4),
        "measured_mean_power_w": round(measured_energy_j / walk_s, 4),
        "estimated_mean_abs_torque_nm": round(estimated_abs_torque_sum / steps, 4),
        "measured_mean_abs_torque_nm": round(measured_abs_torque_sum / steps, 4),
        "start_x_m": round(float(start_xy[0]), 4),
        "start_y_m": round(float(start_xy[1]), 4),
        "end_x_m": round(end_pos[0], 4),
        "end_y_m": round(end_pos[1], 4),
        "end_z_m": round(end_pos[2], 4),
    }

    # Stop before teleporting to the next cell; this period is not included in energy.
    for _ in range(round(STOP_SECONDS / PHYSICS_DT)):
        if not simulation_app.is_running():
            break
        step_robot(sim, robot, controller, zero)
    return row


def summarize(rows: list[dict], terrains: tuple[str, ...], speeds: tuple[float, ...],
              repeats: int, output: Path) -> None:
    """Only completed walks qualify for a J/m comparison."""
    summary_path = output.with_name(output.stem + "_summary.csv")
    fields = ("terrain", "cell_prim", "command_vx_m_s", "completed", "attempted",
              "measured_j_per_m_mean", "measured_j_per_m_std",
              "estimated_j_per_m_mean", "actual_speed_m_s_mean", "walk_s_mean",
              "rankable")
    summaries = []
    for terrain in terrains:
        for speed in speeds:
            subset = [r for r in rows if r["terrain"] == terrain and r["command_vx_m_s"] == speed]
            valid = [r for r in subset if r["status"] == "ok"]
            measured = [float(r["measured_energy_j_per_m"]) for r in valid]
            summary = {
                "terrain": terrain,
                "cell_prim": subset[0]["cell_prim"] if subset else "",
                "command_vx_m_s": speed,
                "completed": len(valid),
                "attempted": len(subset),
                "measured_j_per_m_mean": round(mean(measured), 4) if measured else "",
                "measured_j_per_m_std": round(pstdev(measured), 4) if measured else "",
                "estimated_j_per_m_mean": round(mean(float(r["estimated_energy_j_per_m"]) for r in valid), 4) if valid else "",
                "actual_speed_m_s_mean": round(mean(float(r["mean_speed_m_s"]) for r in valid), 4) if valid else "",
                "walk_s_mean": round(mean(float(r["walk_s"]) for r in valid), 4) if valid else "",
                "rankable": int(len(valid) == repeats),
            }
            summaries.append(summary)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[SPEEDS] summary={summary_path}", flush=True)
    for terrain in terrains:
        candidates = [r for r in summaries if r["terrain"] == terrain and r["rankable"]]
        for r in (r for r in summaries if r["terrain"] == terrain):
            energy = r["measured_j_per_m_mean"]
            print(f"[SPEEDS] {terrain} command={r['command_vx_m_s']:.2f}m/s "
                  f"completed={r['completed']}/{repeats} measured={energy if energy != '' else 'n/a'}J/m "
                  f"actual_speed={r['actual_speed_m_s_mean'] if energy != '' else 'n/a'}m/s", flush=True)
        if candidates:
            best = min(candidates, key=lambda r: r["measured_j_per_m_mean"])
            print(f"[SPEEDS] {terrain} lowest measured J/m among fully completed "
                  f"tested speeds: {best['command_vx_m_s']:.2f}m/s", flush=True)
        else:
            print(f"[SPEEDS] {terrain} no speed completed all {repeats} trials; "
                  "no reliable minimum assigned", flush=True)


def main() -> None:
    terrains = tuple(args.terrains or ALL_TERRAINS)
    speeds = tuple(args.speeds)
    repeats = args.repeats
    target_m = args.distance
    if (repeats < 1 or not math.isfinite(target_m) or target_m <= 0
            or any(not math.isfinite(speed) or speed <= 0 or speed > 1.6 for speed in speeds)):
        raise ValueError("Require repeats >= 1, finite distance > 0 and speeds in (0, 1.6] m/s")
    if len(set(terrains)) != len(terrains) or set(terrains) - set(ALL_TERRAINS):
        raise ValueError(f"--terrains must contain unique names from {ALL_TERRAINS}")
    if len(set(speeds)) != len(speeds):
        raise ValueError("--speeds must not contain duplicates")
    if target_m > 2.5:
        raise ValueError("Distance > 2.5 m may leave a 6 m terrain cell from the standard start")
    output = args.output or (Path(__file__).resolve().parent.parent / "results" /
                              f"terrain_speeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    if output.exists() or output.with_name(output.stem + "_summary.csv").exists():
        raise FileExistsError(f"Output already exists: {output}")
    validate_assets()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device))
    build_scene()
    if not args.headless:
        UsdGeom.Camera.Define(omni.usd.get_context().get_stage(), VIEW_CAMERA_PATH)
    cells = find_terrain_cells(terrains)
    for name, cell in cells.items():
        print(f"[SPEEDS] {name}: {cell['path']} bounds={cell['low']}..{cell['high']}", flush=True)
    robot = create_spot()
    controller = SpotVelocityController(robot, sim.device)
    sim.reset()
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = VIEW_CAMERA_PATH

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        # Flush each row so a stopped run retains all completed trials.
        with output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            handle.flush()
            for trial in range(1, repeats + 1):
                offset = 0.15 * (trial - (repeats + 1) / 2.0)
                for terrain in terrains:
                    for speed in speeds:
                        row = run_trial(sim, robot, controller, terrain, cells[terrain],
                                        trial, target_m, offset, speed)
                        writer.writerow(row)
                        handle.flush()
                        rows.append(row)
                        print(f"[SPEEDS] {terrain} command={speed:.2f}m/s "
                              f"trial={trial}/{repeats} status={row['status']} "
                              f"distance={row['distance_m']:.3f}m "
                              f"measured={row['measured_energy_j_per_m']}J/m",
                              flush=True)
    finally:
        print(f"[SPEEDS] trials={output} completed_rows={len(rows)}", flush=True)
        if rows:
            summarize(rows, terrains, speeds, repeats, output)


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

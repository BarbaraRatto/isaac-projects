"""Measure the same Spot walk on asphalt, fine gravel, and large rocks.

This is a physical sanity check, not RL training or a terrain-label dataset.
"""

import argparse
import csv
import os
import re
import sys
import traceback
from pathlib import Path
from statistics import mean, pstdev

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


parser = argparse.ArgumentParser(description="Compare Spot's mechanical energy on three existing USD terrains.")
parser.add_argument("--repeats", type=int, default=None, help="Trials per terrain (default: config value).")
parser.add_argument("--distance", type=float, default=None, help="Target path length in meters (default: config value).")
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
    BENCHMARK_CSV,
    BENCHMARK_DISTANCE_M,
    BENCHMARK_EDGE_MARGIN_M,
    BENCHMARK_FOOT_MARGIN_M,
    BENCHMARK_MAX_WALK_SECONDS,
    BENCHMARK_MIN_BODY_HEIGHT_M,
    BENCHMARK_REPEATS,
    BENCHMARK_SETTLE_SECONDS,
    BENCHMARK_SPAWN_HEIGHT_M,
    BENCHMARK_START_BACK_M,
    BENCHMARK_TERRAINS,
    PHYSICS_DT,
    STOP_SECONDS,
    WALK_COMMAND,
    validate_assets,
)
from scene import build_scene  # noqa: E402


VIEW_CAMERA_PATH = "/World/BenchmarkCamera"

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


def find_terrain_cells() -> dict[str, dict]:
    """Find three named cells in the loaded USD, preferring one common row."""
    stage = omni.usd.get_context().get_stage()
    cells = []
    for prim in stage.Traverse():
        match = re.fullmatch(r"Cell_(\d+)_(\d+)_(.+)", prim.GetName())
        if match and match.group(3) in BENCHMARK_TERRAINS:
            cells.append((int(match.group(1)), int(match.group(2)), match.group(3), prim))
    for row in sorted({item[0] for item in cells}):
        if all(any(item[0] == row and item[2] == kind for item in cells) for kind in BENCHMARK_TERRAINS):
            selected = [min((item for item in cells if item[0] == row and item[2] == kind), key=lambda x: x[1])
                        for kind in BENCHMARK_TERRAINS]
            break
    else:
        selected = [min((item for item in cells if item[2] == kind), key=lambda x: (x[0], x[1]))
                    for kind in BENCHMARK_TERRAINS]

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


def run_trial(sim, robot, controller, terrain: str, cell: dict, trial: int, target_m: float, lateral_offset_m: float) -> dict:
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
    forward = torch.tensor([WALK_COMMAND], device=sim.device, dtype=torch.float32)
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
    for steps in range(1, round(BENCHMARK_MAX_WALK_SECONDS / PHYSICS_DT) + 1):
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
        "command_vx_m_s": WALK_COMMAND[0],
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


def main() -> None:
    repeats = BENCHMARK_REPEATS if args.repeats is None else args.repeats
    target_m = BENCHMARK_DISTANCE_M if args.distance is None else args.distance
    if repeats < 1 or target_m <= 0:
        raise ValueError("--repeats and --distance must be positive")
    validate_assets()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device))
    build_scene()
    if not args.headless:
        UsdGeom.Camera.Define(omni.usd.get_context().get_stage(), VIEW_CAMERA_PATH)
    cells = find_terrain_cells()
    for name, cell in cells.items():
        print(f"[COMPARE] {name}: {cell['path']} bounds={cell['low']}..{cell['high']}", flush=True)
    robot = create_spot()
    controller = SpotVelocityController(robot, sim.device)
    sim.reset()
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = VIEW_CAMERA_PATH
        else:
            print("[COMPARE] No active viewport found; measurements will continue without a GUI view.", flush=True)

    BENCHMARK_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    pending_csv = BENCHMARK_CSV.with_suffix(".csv.tmp")
    try:
        with pending_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for trial in range(1, repeats + 1):
                for terrain in BENCHMARK_TERRAINS:
                    lateral_offset_m = 0.15 * (trial - (repeats + 1) / 2.0)
                    row = run_trial(sim, robot, controller, terrain, cells[terrain],
                                    trial, target_m, lateral_offset_m)
                    writer.writerow(row)
                    handle.flush()
                    rows.append(row)
                    print(
                        f"[COMPARE] {terrain} trial={trial} status={row['status']} "
                        f"distance={row['distance_m']:.3f}m "
                        f"estimated={row['estimated_energy_j_per_m']}J/m "
                        f"measured={row['measured_energy_j_per_m']}J/m",
                        flush=True,
                    )
        os.replace(pending_csv, BENCHMARK_CSV)
    finally:
        pending_csv.unlink(missing_ok=True)
    print(f"[COMPARE] CSV: {BENCHMARK_CSV}", flush=True)
    for terrain in BENCHMARK_TERRAINS:
        completed = [row for row in rows if row["terrain"] == terrain and row["status"] == "ok"]
        if not completed:
            print(f"[COMPARE] {terrain}: no completed walks", flush=True)
            continue
        for source in ("estimated", "measured"):
            values = [float(row[f"{source}_energy_j_per_m"]) for row in completed]
            print(f"[COMPARE] {terrain}: {len(values)}/{repeats} completed, {source} "
                  f"mean={mean(values):.2f} J/m, std={pstdev(values):.2f} J/m", flush=True)


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

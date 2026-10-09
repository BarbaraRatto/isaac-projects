"""Precompute DINO features from the ORIGINAL terrain USD for the cached run.

Run this once before either training. The camera is placed at head-height poses
around the rocks, ice, ramps and goal; Spot does not have to walk to scan them.
"""

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "runs" / "ice_choice_atlas.npz")
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists() and not args.overwrite:
    parser.error(f"Atlas already exists: {args.output}; pass --overwrite to replace it")
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_mul  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402

from ice_visual_features import (COLS, RESOLUTION_M, ROWS, X0_M, Y0_M,
                                 FeatureField, camera_patches, save_atlas)  # noqa: E402
from rl_config import PHYSICS_DT, validate_assets  # noqa: E402
from scene import build_scene  # noqa: E402
from zed_sensor import create_body_zed  # noqa: E402

sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "spot_terrain_gridmap"))


def _scan_poses():
    deg = math.radians
    return (
        (18.0, 3.0, 45), (18.0, 3.0, 70), (18.0, 3.0, 90),
        (19.5, 3.0, 45), (19.5, 3.0, 75),
        (21.5, 3.0, 65), (22.5, 3.0, 90), (24.0, 3.0, 90),
        (22.5, 4.5, 90), (24.0, 4.5, 115),
        (22.5, 6.0, 90), (22.5, 6.0, 135), (24.0, 6.0, 135),
        (22.5, 7.5, 115), (24.0, 7.5, 135),
        (24.0, 9.0, 180), (24.0, 9.0, 225), (26.0, 9.0, 200),
    )


def _place_camera(camera, x: float, y: float, yaw_deg: float, device: str):
    yaw = math.radians(yaw_deg)
    rotation = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
                            device=device, dtype=torch.float32)
    pitch = math.radians(15.0)
    camera_pitch = torch.tensor([[math.cos(pitch / 2), 0.0,
                                  math.sin(pitch / 2), 0.0]],
                                device=device, dtype=torch.float32)
    # Approximate the settled Spot body height. Online capture follows the
    # actual simulated body; the atlas therefore remains an approximation.
    root = torch.tensor([[x, y, 0.55]], device=device, dtype=torch.float32)
    offset = torch.tensor([[0.42, 0.0, 0.07]], device=device, dtype=torch.float32)
    camera.set_world_poses(
        positions=root + quat_apply(rotation, offset),
        orientations=quat_mul(rotation, camera_pitch), convention="world",
    )


def main():
    started = time.perf_counter()
    validate_assets()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device))
    build_scene()  # References real_terrains.usd without changing its layout.
    camera = create_body_zed(depth=True)
    sim.reset()
    from spot_terrain_gridmap.dino_feature_extractor import DinoFeatureExtractor
    dino = DinoFeatureExtractor(model_name="facebook/dinov2-small", device="cuda", input_size=518)
    field = FeatureField()
    try:
        poses = _scan_poses()
        for index, (x, y, yaw) in enumerate(poses, 1):
            camera.reset()
            _place_camera(camera, x, y, yaw, args.device)
            sim.step(render=True)
            camera.update(PHYSICS_DT, force_recompute=True)
            points, features = camera_patches(camera, dino)
            field.insert(points, features)
            print(f"[ATLAS] {index}/{len(poses)} pose=({x:.1f}, {y:.1f}, {yaw}deg) "
                  f"patches={len(points)} occupied={int((field.count > 0).sum())}", flush=True)
        occupied_mask = field.count > 0
        relevant_cells = {
            "start_rocks": (15.0, 21.0, 1.5, 4.5),
            "ice": (15.0, 21.0, 4.5, 7.5),
            "lower_ramp": (21.0, 27.0, 1.5, 4.5),
            "upper_ramp": (21.0, 27.0, 4.5, 7.5),
            "goal_asphalt": (21.0, 27.0, 7.5, 10.5),
        }
        for name, (xlo, xhi, ylo, yhi) in relevant_cells.items():
            r0 = max(0, int((xlo - X0_M) / RESOLUTION_M))
            r1 = min(ROWS, int((xhi - X0_M) / RESOLUTION_M))
            c0 = max(0, int((ylo - Y0_M) / RESOLUTION_M))
            c1 = min(COLS, int((yhi - Y0_M) / RESOLUTION_M))
            count = int(occupied_mask[r0:r1, c0:c1].sum())
            print(f"[ATLAS] {name} occupied={count}/{(r1-r0)*(c1-c0)}", flush=True)
            if count == 0:
                raise RuntimeError(f"No visual features recorded for {name}; atlas not saved")
        occupied = save_atlas(args.output, field, time.perf_counter() - started)
        print(f"[ATLAS] saved={args.output.resolve()} occupied={occupied}", flush=True)
    finally:
        sim.clear_instance()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(skip_cleanup=args.headless)

"""Create a PCA basis from camera views around the new full-USD route.

The saved atlas supplies PCA axes for live DINO; the RL policy does not look up
precomputed terrain features during training.
"""

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.append(str(HERE.parent))
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=HERE / "runs" / "full_grid_pca_atlas.npz")
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

from full_grid_energy_choice.visual_features import (COLS, RESOLUTION_M, ROWS, X0_M, Y0_M,
                                                     FeatureField, camera_patches, save_atlas)  # noqa: E402
from rl_config import PHYSICS_DT, validate_assets  # noqa: E402
from isaacsim.core.utils.prims import create_prim  # noqa: E402
from rl_config import TERRAIN_USD  # noqa: E402
from zed_sensor import create_body_zed  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "ros2_ws" / "src" / "spot_terrain_gridmap"))


def _scan_poses():
    # Head-height views of the ramp start, asphalt, rocks, obstacles and
    # possible routes between them. These poses do not prescribe an RL route.
    return (
        (3.7, 3.2, 90), (6.0, 3.2, 90),
        (3.7, 5.0, 90), (6.0, 5.0, 90), (3.7, 6.8, 90),
        (-2.0, 5.0, 55), (0.0, 5.0, 75), (2.0, 5.0, 90),
        (-2.0, 6.5, 45), (0.0, 6.5, 75), (2.0, 6.5, 100),
        (-2.0, 8.0, 45), (0.0, 8.0, 70), (2.0, 8.0, 100),
        (4.0, 8.0, 120), (6.0, 8.0, 125),
        (-2.0, 9.5, 45), (0.0, 9.5, 70), (2.0, 9.5, 90),
        (4.0, 9.5, 110), (6.0, 9.5, 130),
        (0.0, 11.0, 45), (2.0, 11.0, 80), (4.0, 11.0, 110),
        (6.0, 11.0, 150), (4.0, 12.5, 180),
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
    create_prim("/World/Warehouse", "Xform", usd_path=str(TERRAIN_USD))
    light = sim_utils.DistantLightCfg(intensity=3000.0, angle=1.0)
    light.func("/World/defaultLight", light)
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
            print(f"[FULL-ATLAS] {index}/{len(poses)} pose=({x:.1f}, {y:.1f}, {yaw}deg) "
                  f"patches={len(points)} occupied={int((field.count > 0).sum())}", flush=True)
        occupied_mask = field.count > 0
        relevant_cells = {
            "start_ramp": (3.0, 9.0, 4.5, 7.5),
            "middle_asphalt": (-3.0, 3.0, 7.5, 10.5),
            "middle_rocks": (3.0, 9.0, 7.5, 10.5),
            "goal_obstacles": (3.0, 9.0, 10.5, 13.5),
        }
        for name, (xlo, xhi, ylo, yhi) in relevant_cells.items():
            r0 = max(0, int((xlo - X0_M) / RESOLUTION_M))
            r1 = min(ROWS, int((xhi - X0_M) / RESOLUTION_M))
            c0 = max(0, int((ylo - Y0_M) / RESOLUTION_M))
            c1 = min(COLS, int((yhi - Y0_M) / RESOLUTION_M))
            count = int(occupied_mask[r0:r1, c0:c1].sum())
            print(f"[FULL-ATLAS] {name} occupied={count}/{(r1-r0)*(c1-c0)}", flush=True)
            if count == 0:
                raise RuntimeError(f"No visual features recorded for {name}; atlas not saved")
        occupied = save_atlas(args.output, field, time.perf_counter() - started)
        print(f"[FULL-ATLAS] saved={args.output.resolve()} occupied={occupied}", flush=True)
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

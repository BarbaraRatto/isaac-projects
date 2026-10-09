"""Scan the short course with ZED X, extract DINO, and save a world feature field."""

import argparse
import math
import sys
import traceback
import os
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ZED_USD = Path("/home/isaac/isaac-sim/assets/Assets/Isaac/5.1/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc")
DEFAULT_OUTPUT = HERE / "runs" / "first_choice_feature_atlas.npz"

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--max-poses", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402

from first_choice_scene import build_first_choice_scene  # noqa: E402
from locomotion import SpotVelocityController, create_spot, reset_spot  # noqa: E402
from rl_config import PHYSICS_DT, validate_assets  # noqa: E402
from visual_features import FeatureField, camera_patches, save_atlas  # noqa: E402
from zed_sensor import create_body_zed, sync_zed_pose  # noqa: E402

sys.path.insert(0, str(ROOT / "IsaacRobotics" / "applications"))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "spot_terrain_gridmap"))


def main():
    validate_assets()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device))
    build_first_choice_scene()
    robot = create_spot()
    camera = create_body_zed(depth=True)
    controller = SpotVelocityController(robot, sim.device)
    sim.reset()
    from spot_terrain_gridmap.dino_feature_extractor import DinoFeatureExtractor
    dino = DinoFeatureExtractor(model_name="facebook/dinov2-small", device="cuda", input_size=518)
    field = FeatureField()
    poses = []
    for x in (1.5, 3.0, 4.5):
        poses.extend((x, 0.0, yaw) for yaw in (-0.45, 0.0, 0.45))
    for x in (6.5, 8.5, 10.5, 12.5, 14.5):
        poses.extend((x, 3.0, yaw) for yaw in (-0.45, 0.0, 0.45))
    for x in (12.5, 14.5, 16.0):
        poses.extend((x, 0.0, yaw) for yaw in (-0.45, 0.0, 0.45))
    zero = torch.zeros((1, 3), device=sim.device)
    for index, (x, y, yaw) in enumerate(poses[:args.max_poses], 1):
        reset_spot(robot, (x, y, 0.75))
        root_state = robot.data.default_root_state.clone()
        root_state[0, :3] = torch.tensor((x, y, 0.75), device=sim.device)
        root_state[0, 3:7] = torch.tensor((math.cos(yaw / 2), 0.0, 0.0,
                                         math.sin(yaw / 2)), device=sim.device)
        robot.write_root_state_to_sim(root_state)
        controller.reset()
        robot.update(PHYSICS_DT)
        camera.reset()
        for _ in range(3):
            controller.forward(zero)
            sim.step(render=False)
            robot.update(PHYSICS_DT)
        sync_zed_pose(camera, robot)
        controller.forward(zero)
        sim.step(render=True)
        robot.update(PHYSICS_DT)
        camera.update(4 * PHYSICS_DT, force_recompute=True)
        points, features = camera_patches(camera, dino)
        field.insert(points, features)
        if index == 1:
            depth = camera.data.output["distance_to_image_plane"][0].cpu().numpy()
            print(f"[ATLAS-DEBUG] points={len(points)} camera_pos={camera.data.pos_w[0]} "
                  f"quat={camera.data.quat_w_ros[0]} root_pos={robot.data.root_pos_w[0]} "
                  f"finite_depth={int(__import__('numpy').isfinite(depth).sum())} "
                  f"depth_shape={depth.shape}", flush=True)
            if len(points):
                print(f"[ATLAS-DEBUG] points_xy_min={points[:, :2].min(axis=0)} "
                      f"max={points[:, :2].max(axis=0)}", flush=True)
            from PIL import Image
            Image.fromarray(camera.data.output["rgb"][0].cpu().numpy()).save('/tmp/spot_zed_debug.png')
        print(f"[ATLAS] {index}/{len(poses)} pose=({x:.1f},{y:.1f},{yaw:.2f}) "
              f"ground_patches={len(points)} occupied={int((field.count>0).sum())}", flush=True)
    if args.max_poses is not None:
        sim.clear_instance()
        return
    occupied = save_atlas(args.output, field)
    print(f"[ATLAS] saved={args.output} occupied={occupied}/432", flush=True)
    sim.clear_instance()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        if args.headless:
            os._exit(1)
        simulation_app.close()
        raise
    else:
        simulation_app.close(skip_cleanup=args.headless)

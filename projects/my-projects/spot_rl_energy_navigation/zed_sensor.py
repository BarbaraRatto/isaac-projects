"""ZED X right-camera optics, synchronized explicitly to simulated Spot pose.

Isaac Sim's source ZED_X.usdc has a rigid body; for RL RGB-D rendering we
copy its CameraRight optical parameters onto a movable USD Camera prim.
"""

from pathlib import Path

import omni.usd
import torch
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import quat_apply, quat_mul
from pxr import Gf, Usd, UsdGeom


ZED_USD = Path("/home/isaac/isaac-sim/assets/Assets/Isaac/5.1/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc")
ZED_CAMERA_PRIM = "/Root/base_link/ZED_X/CameraRight"


def create_body_zed(*, depth: bool = True) -> Camera:
    if not ZED_USD.is_file():
        raise FileNotFoundError(ZED_USD)
    source_stage = Usd.Stage.Open(str(ZED_USD))
    source = UsdGeom.Camera.Get(source_stage, ZED_CAMERA_PRIM)
    if not source:
        raise RuntimeError(f"CameraRight absent in {ZED_USD}")
    stage = omni.usd.get_context().get_stage()
    path = "/World/NavigationZEDCamera"
    target = UsdGeom.Camera.Define(stage, path)
    target.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    target.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    target.CreateFocalLengthAttr(float(source.GetFocalLengthAttr().Get()))
    target.CreateHorizontalApertureAttr(float(source.GetHorizontalApertureAttr().Get()))
    target.CreateVerticalApertureAttr(float(source.GetVerticalApertureAttr().Get()))
    target.CreateClippingRangeAttr(source.GetClippingRangeAttr().Get())
    return Camera(CameraCfg(
        prim_path=path, update_period=0.0, height=360, width=640,
        update_latest_camera_pose=True,
        data_types=["rgb", "distance_to_image_plane"] if depth else ["rgb"],
        spawn=None,
    ))


def sync_zed_pose(camera: Camera, robot) -> None:
    """Copy Spot's simulated body pose to the render camera before capture."""
    root = robot.data.root_pos_w[0:1]
    rotation = robot.data.root_quat_w[0:1]
    offset = torch.tensor([[0.42, 0.0, 0.07]], device=root.device, dtype=root.dtype)
    half = torch.deg2rad(torch.tensor(7.5, device=root.device, dtype=root.dtype))
    pitch = torch.stack((torch.cos(half), half * 0, torch.sin(half), half * 0)).view(1, 4)
    camera.set_world_poses(
        positions=root + quat_apply(rotation, offset),
        orientations=quat_mul(rotation, pitch),
        convention="world",
    )

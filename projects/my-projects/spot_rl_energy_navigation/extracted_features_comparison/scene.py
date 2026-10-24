"""Isaac-only scene construction. Import only AFTER AppLauncher startup."""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from pxr import UsdGeom, UsdLux

from common import camera_optics, local_path


def quaternion_from_euler(x: float, y: float, z: float) -> tuple:
    x, y, z = (math.radians(v) / 2 for v in (x, y, z))
    cx, cy, cz, sx, sy, sz = math.cos(x), math.cos(y), math.cos(z), math.sin(x), math.sin(y), math.sin(z)
    return (cx*cy*cz + sx*sy*sz, sx*cy*cz - cx*sy*sz,
            cx*sy*cz + sx*cy*sz, cx*cy*sz - sx*sy*cz)


def build_scene(config: dict, manifest: dict, device: str):
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/60, device=device))
    stage = sim.stage
    paths = []
    for i, variant in enumerate(manifest["terrains"]):
        path = f"/World/Terrains/terrain_{i}"
        spawn = sim_utils.UsdFileCfg(usd_path=str(local_path(variant["file"])))
        spawn.func(path, spawn)
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
        paths.append(path)
    light = sim_utils.DistantLightCfg(intensity=3000.0, angle=1.0)
    light.func("/World/Light", light)
    camera_settings = config["camera"]
    optics = camera_optics(config)
    camera = Camera(CameraCfg(
        prim_path="/World/ComparisonCamera", width=camera_settings["width"],
        height=camera_settings["height"], update_period=0.0,
        update_latest_camera_pose=True,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(**optics),
    ))
    sim.reset()
    sim.set_camera_view(eye=(4.0, 4.0, 3.0), target=(0.0, 0.0, 0.0))
    return sim, camera, paths


def select_terrain_and_light(sim, paths: list[str], selected: int, lighting: dict) -> None:
    for i, path in enumerate(paths):
        imageable = UsdGeom.Imageable(sim.stage.GetPrimAtPath(path))
        imageable.MakeVisible() if i == selected else imageable.MakeInvisible()
    light = UsdLux.DistantLight.Get(sim.stage, "/World/Light")
    light.GetIntensityAttr().Set(float(lighting["intensity"]))
    xform = UsdGeom.Xformable(light.GetPrim())
    # Clear rotation only, preserving the light itself. Match training identity by default.
    xform.ClearXformOpOrder()
    xform.AddRotateXYZOp().Set(tuple(lighting["rotation_deg"]))

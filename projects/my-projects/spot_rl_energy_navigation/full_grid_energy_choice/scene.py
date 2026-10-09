"""Isolated copies of the complete original USD, Spot and ZED X optics."""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_assets import SPOT_CFG
from isaacsim.core.utils.prims import create_prim
from pxr import Usd, UsdGeom

from full_grid_energy_choice.config import GEOMETRY, NAV_CFG_FULL
from full_grid_energy_choice.visual_features import VISUAL_SIZE
from rl_config import PHYSICS_DT, ROBOT_USD, TERRAIN_USD
from zed_sensor import ZED_CAMERA_PRIM, ZED_USD


def camera_optics() -> dict:
    source_stage = Usd.Stage.Open(str(ZED_USD))
    source = UsdGeom.Camera.Get(source_stage, ZED_CAMERA_PRIM)
    if not source:
        raise RuntimeError(f"ZED X CameraRight absent in {ZED_USD}")
    clip = source.GetClippingRangeAttr().Get()
    return dict(
        focal_length=float(source.GetFocalLengthAttr().Get()),
        horizontal_aperture=float(source.GetHorizontalApertureAttr().Get()),
        vertical_aperture=float(source.GetVerticalApertureAttr().Get()),
        clipping_range=(float(clip[0]), float(clip[1])),
    )


def spot_cfg() -> ArticulationCfg:
    cfg = SPOT_CFG.copy()
    cfg.prim_path = "/World/envs/env_.*/Spot"
    cfg.spawn.usd_path = str(ROBOT_USD)
    cfg.init_state.pos = (*GEOMETRY.start_xy, NAV_CFG_FULL.spawn_height_m)
    half = GEOMETRY.start_yaw_rad / 2
    cfg.init_state.rot = (math.cos(half), 0.0, 0.0, math.sin(half))
    cfg.actuators = {
        "hips": ImplicitActuatorCfg(joint_names_expr=[".*_h[xy]"],
                                    effort_limit_sim=45.0, stiffness=60.0, damping=1.5),
        "knees": ImplicitActuatorCfg(joint_names_expr=[".*_kn"],
                                     effort_limit_sim=115.0, stiffness=60.0, damping=1.5),
    }
    return cfg


@configclass
class FullGridChoiceCfg(DirectRLEnvCfg):
    decimation = NAV_CFG_FULL.physics_steps_per_action
    episode_length_s = NAV_CFG_FULL.max_episode_seconds
    sim = sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=decimation)
    scene = InteractiveSceneCfg(num_envs=4, env_spacing=50.0, replicate_physics=True)
    robot_cfg = spot_cfg()
    camera_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/NavigationZEDCamera",
        update_period=0.0, height=360, width=640,
        update_latest_camera_pose=True,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(**camera_optics()),
    )
    action_space = 3
    observation_space = 11 + VISUAL_SIZE
    state_space = 0
    rerender_on_reset = True
    viewer = ViewerCfg(eye=(13.0, 17.0, 14.0), lookat=(4.0, 9.0, 0.0))


def create_source_terrain() -> None:
    if not TERRAIN_USD.is_file():
        raise FileNotFoundError(TERRAIN_USD)
    stage = Usd.Stage.Open(str(TERRAIN_USD))
    for name, xy in (("Cell_2_1_t3_ramp", GEOMETRY.start_xy),
                     ("Cell_4_1_t6_obstacles", GEOMETRY.goal_xy)):
        prim = stage.GetPrimAtPath("/World/Grid/" + name)
        if not prim.IsValid():
            raise RuntimeError(f"Required cell absent from original USD: {name}")
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bounds = box.ComputeWorldBound(prim).ComputeAlignedBox()
        lo, hi = bounds.GetMin(), bounds.GetMax()
        if not (lo[0] + 0.5 <= xy[0] <= hi[0] - 0.5
                and lo[1] + 0.5 <= xy[1] <= hi[1] - 0.5):
            raise RuntimeError(f"Point {xy} is too close to the edge of {name}")
    create_prim("/World/envs/env_0/Warehouse", "Xform", usd_path=str(TERRAIN_USD))
    light = sim_utils.DistantLightCfg(intensity=3000.0, angle=1.0)
    light.func("/World/defaultLight", light)

"""Local test13 scene configuration, using existing assets in read-only mode."""

import math

import gymnasium as gym
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets import SPOT_CFG
from isaacsim.core.utils.prims import create_prim
from pxr import Usd, UsdGeom

from multi_terrain_test13.terrain import read_terrain_codes
from multi_terrain_test13.assets import ROBOT_USD
from multi_terrain_test13.assets import ZED_CAMERA_PRIM, ZED_USD
from multi_terrain_test13.config import ACTION_LOW, ACTION_HIGH, NAV_CFG, OBSERVATION_SIZE, CRITIC_SIZE, FOOT_NAMES


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
    cfg.spawn.activate_contact_sensors = True
    cfg.init_state.pos = (0.0, 0.0, NAV_CFG.spawn_height_m)
    half = math.pi / 8.0
    cfg.init_state.rot = (math.cos(half), 0.0, 0.0, math.sin(half))
    cfg.actuators = {
        "hips": ImplicitActuatorCfg(joint_names_expr=[".*_h[xy]"],
                                    effort_limit_sim=45.0, stiffness=60.0, damping=1.5),
        "knees": ImplicitActuatorCfg(joint_names_expr=[".*_kn"],
                                     effort_limit_sim=115.0, stiffness=60.0, damping=1.5),
    }
    return cfg


@configclass
class Test13Cfg(DirectRLEnvCfg):
    decimation = NAV_CFG.physics_steps_per_action
    episode_length_s = NAV_CFG.max_episode_seconds
    sim = sim_utils.SimulationCfg(dt=NAV_CFG.physics_dt, render_interval=decimation,
                                  enable_scene_query_support=True)
    scene = InteractiveSceneCfg(num_envs=32, env_spacing=50.0,
                                replicate_physics=False, filter_collisions=True)
    robot_cfg = spot_cfg()
    camera_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/NavigationZEDCamera",
        update_period=0.0, height=360, width=640,
        update_latest_camera_pose=True,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(**camera_optics()),
    )
    action_space = gym.spaces.Box(np.array(ACTION_LOW, dtype=np.float32),
                                  np.array(ACTION_HIGH, dtype=np.float32))
    observation_space = OBSERVATION_SIZE
    state_space = CRITIC_SIZE
    rerender_on_reset = True
    viewer = ViewerCfg(eye=(8.0, 9.0, 11.0), lookat=(2.0, 2.0, 0.0))


def setup_scene(env) -> None:
    for condition in env.conditions:
        read_terrain_codes(condition.case)
        create_prim(f"/World/envs/env_{condition.env_id}/Warehouse", "Xform",
                    usd_path=str(condition.layout.usd))
    light = sim_utils.DistantLightCfg(intensity=3000.0, angle=1.0)
    light.func("/World/defaultLight", light)
    env.robot = Articulation(env.cfg.robot_cfg)
    env.camera = TiledCamera(env.cfg.camera_cfg)
    env.scene.articulations["spot"] = env.robot
    env.scene.sensors["zed_x"] = env.camera
    # Exactly one foot body per environment per sensor: required for filtered contacts.
    env.foot_sensors = []
    filters = [f'/World/envs/env_.*/Warehouse/Grid/Cell_{r}_{c}_.*/terrain/mesh'
               for r in range(3) for c in range(3)]
    for foot in FOOT_NAMES:
        sensor = ContactSensor(ContactSensorCfg(
            prim_path=f'/World/envs/env_.*/Spot/{foot}', update_period=0.0,
            filter_prim_paths_expr=filters, track_contact_points=True,
            max_contact_data_count_per_prim=64))
        env.scene.sensors[foot] = sensor
        env.foot_sensors.append(sensor)

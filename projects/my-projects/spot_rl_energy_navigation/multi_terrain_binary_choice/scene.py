"""Thirty-two heterogeneous 3x3 USDs, each with Spot and ZED X."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera
from isaaclab.utils import configclass
from isaacsim.core.utils.prims import create_prim

from binary_choice.terrain import read_terrain_codes
from full_grid_energy_choice.scene import FullGridChoiceCfg
from multi_terrain_binary_choice.layouts import Condition


@configclass
class MultiChoiceCfg(FullGridChoiceCfg):
    # Isaac Lab must not replicate env_0's physics into heterogeneous scenes.
    scene = InteractiveSceneCfg(num_envs=32, env_spacing=50.0,
                                replicate_physics=False, filter_collisions=True)


def create_terrains(conditions: tuple[Condition, ...]) -> None:
    for condition in conditions:
        read_terrain_codes(condition.case)
        create_prim(f"/World/envs/env_{condition.env_id}/Warehouse", "Xform",
                    usd_path=str(condition.layout.usd))
    light = sim_utils.DistantLightCfg(intensity=3000.0, angle=1.0)
    light.func("/World/defaultLight", light)


def setup_scene(env) -> None:
    create_terrains(env.conditions)
    env.robot = Articulation(env.cfg.robot_cfg)
    env.camera = TiledCamera(env.cfg.camera_cfg)
    env.scene.articulations["spot"] = env.robot
    env.scene.sensors["zed_x"] = env.camera

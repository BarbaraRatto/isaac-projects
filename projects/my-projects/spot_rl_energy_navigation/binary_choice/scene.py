"""Load one selected 3x3 USD into each isolated Isaac Lab environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaacsim.core.utils.prims import create_prim
from binary_choice.config import BinaryCase
from binary_choice.terrain import read_terrain_codes
from full_grid_energy_choice.scene import FullGridChoiceCfg


BinaryChoiceCfg = FullGridChoiceCfg  # Reuse the same Spot, actuators and ZED X.


def create_binary_terrain(case: BinaryCase) -> None:
    read_terrain_codes(case)
    create_prim("/World/envs/env_0/Warehouse", "Xform", usd_path=str(case.terrain_usd))
    light = sim_utils.DistantLightCfg(intensity=3000.0, angle=1.0)
    light.func("/World/defaultLight", light)

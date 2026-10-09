"""Build the same terrain and lighting used by the standalone Spot scene."""

import isaaclab.sim as sim_utils
from isaacsim.core.utils.prims import create_prim

from rl_config import TERRAIN_USD


def build_scene() -> None:
    """Reference the existing USD, retaining its mesh and physics materials."""
    create_prim("/World/Warehouse", "Xform", usd_path=str(TERRAIN_USD))
    light = sim_utils.DistantLightCfg(intensity=3000.0, angle=1.0)
    light.func("/World/defaultLight", light)

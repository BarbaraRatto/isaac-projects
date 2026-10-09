"""Arrange existing USD cells into two parallel routes for an energy test.

The source USD stays untouched. Local Xform overrides reposition complete cells;
physics and visual bindings are inherited from the source.
"""

import re

import omni.usd
from pxr import Gf, Usd, UsdGeom

from scene import build_scene


# Source cell -> bottom-left corner in the new course. Cells are 6 x 3 m.
COURSE_CELLS = {
    "Cell_0_2_t1_asphalt": (0.0, -1.5),    # common start
    "Cell_3_3_t6_obstacles": (6.0, -1.5),  # direct, section 1
    "Cell_4_1_t6_obstacles": (12.0, -1.5), # direct, section 2
    "Cell_6_5_t6_obstacles": (18.0, -1.5), # direct, section 3
    "Cell_5_5_t6_obstacles": (24.0, -1.5), # direct, section 4
    "Cell_0_6_t1_asphalt": (30.0, -1.5),  # common goal
    "Cell_0_4_t1_asphalt": (6.0, 1.5),    # asphalt bypass, section 1
    "Cell_2_0_t1_asphalt": (12.0, 1.5),   # asphalt bypass, section 2
    "Cell_3_0_t1_asphalt": (18.0, 1.5),   # asphalt bypass, section 3
    "Cell_3_4_t1_asphalt": (24.0, 1.5),   # asphalt bypass, section 4
    "Cell_5_4_t1_asphalt": (30.0, 1.5),   # bypass return corridor
}


def build_two_route_scene() -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    build_scene()
    stage = omni.usd.get_context().get_stage()
    grid = stage.GetPrimAtPath("/World/Warehouse/Grid")
    if not grid.IsValid():
        raise RuntimeError("Terrain USD has no /World/Warehouse/Grid")
    cells = [prim for prim in grid.GetChildren() if re.fullmatch(r"Cell_\d+_\d+_.*", prim.GetName())]
    if len(cells) != 49:
        raise RuntimeError(f"Expected 49 original terrain cells, found {len(cells)}")
    found = {prim.GetName() for prim in cells}
    missing = set(COURSE_CELLS) - found
    if missing:
        raise RuntimeError(f"Missing terrain cells: {sorted(missing)}")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bounds = {}
    for prim in cells:
        if prim.GetName() in COURSE_CELLS:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            bounds[prim.GetName()] = (tuple(float(v) for v in box.GetMin()),
                                      tuple(float(v) for v in box.GetMax()))

    for prim in cells:
        name = prim.GetName()
        if name not in COURSE_CELLS:
            prim.SetActive(False)
            continue
        source_low, source_high = bounds[name]
        target_x, target_y = COURSE_CELLS[name]
        if abs(source_high[0] - source_low[0] - 6.0) > 0.01 or abs(source_high[1] - source_low[1] - 3.0) > 0.01:
            raise RuntimeError(f"Unexpected source size for {name}: {bounds[name]}")
        root = UsdGeom.Xform.Define(stage, prim.GetPath())
        root.AddTranslateOp().Set(Gf.Vec3d(target_x - source_low[0],
                                           target_y - source_low[1], 0.0))
    return bounds

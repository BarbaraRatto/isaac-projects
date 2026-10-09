"""Short two-choice course made from existing, unchanged USD terrain cells."""

import re

import omni.usd
from pxr import Gf, Usd, UsdGeom

from scene import build_scene


# Existing meshes/materials, re-positioned only in this in-memory stage.
CELLS = {
    "Cell_0_2_t1_asphalt": (0.0, -1.5),
    "Cell_3_3_t6_obstacles": (6.0, -1.5),
    "Cell_0_4_t1_asphalt": (6.0, 1.5),
    "Cell_0_6_t1_asphalt": (12.0, -1.5),
    "Cell_5_4_t1_asphalt": (12.0, 1.5),
}


def build_first_choice_scene() -> None:
    build_scene()
    grid = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Warehouse/Grid")
    if not grid.IsValid():
        raise RuntimeError("Missing terrain grid")
    cells = [p for p in grid.GetChildren() if re.fullmatch(r"Cell_\d+_\d+_.*", p.GetName())]
    missing = set(CELLS) - {p.GetName() for p in cells}
    if missing:
        raise RuntimeError(f"Missing terrain cells: {sorted(missing)}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bounds = {}
    for prim in cells:
        if prim.GetName() in CELLS:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            bounds[prim.GetName()] = (tuple(map(float, box.GetMin())),
                                      tuple(map(float, box.GetMax())))
    for prim in cells:
        name = prim.GetName()
        if name not in CELLS:
            prim.SetActive(False)
            continue
        low, high = bounds[name]
        if abs(high[0] - low[0] - 6.0) > 0.02 or abs(high[1] - low[1] - 3.0) > 0.02:
            raise RuntimeError(f"Unexpected cell dimensions: {name}: {low}, {high}")
        x, y = CELLS[name]
        UsdGeom.Xform.Define(grid.GetStage(), prim.GetPath()).AddTranslateOp().Set(
            Gf.Vec3d(x - low[0], y - low[1], 0.0)
        )

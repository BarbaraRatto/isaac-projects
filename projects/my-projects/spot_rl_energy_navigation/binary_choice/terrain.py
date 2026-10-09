"""Read and validate the nine terrain cells of one binary-choice USD."""

import re

from pxr import Usd, UsdGeom

from binary_choice.config import BinaryCase, TERRAIN_CODES


def read_terrain_codes(case: BinaryCase) -> list[list[int]]:
    """Check that the USD has the requested nine cells at 2 m spacing."""
    if not case.terrain_usd.is_file():
        raise FileNotFoundError(case.terrain_usd)
    stage = Usd.Stage.Open(str(case.terrain_usd))
    if stage is None:
        raise RuntimeError(f"Could not open {case.terrain_usd}")
    grid = stage.GetPrimAtPath("/World/Grid")
    if not grid.IsValid():
        raise RuntimeError(f"/World/Grid is absent from {case.terrain_usd}")
    children = list(grid.GetChildren())
    if len(children) != 9:
        raise RuntimeError(f"Expected 9 terrain cells in {case.terrain_usd}, found {len(children)}")
    codes = [[-1] * 3 for _ in range(3)]
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in children:
        match = re.fullmatch(r"Cell_(\d+)_(\d+)_(.+)", prim.GetName())
        if match is None:
            raise RuntimeError(f"Unexpected cell name: {prim.GetPath()}")
        row, col, terrain = int(match[1]), int(match[2]), match[3]
        if not 0 <= row < 3 or not 0 <= col < 3:
            raise RuntimeError(f"Cell outside 3x3 grid: {prim.GetPath()}")
        if terrain != case.grid[row][col] or terrain not in TERRAIN_CODES:
            raise RuntimeError(f"Case {case.name}: unexpected terrain at ({row}, {col}): {terrain}")
        if codes[row][col] != -1:
            raise RuntimeError(f"Duplicate terrain cell ({row}, {col})")
        box = bounds.ComputeWorldBound(prim).ComputeAlignedBox()
        lo, hi = box.GetMin(), box.GetMax()
        expected = (col * 2 - 1, col * 2 + 1, row * 2 - 1, row * 2 + 1)
        actual = (lo[0], hi[0], lo[1], hi[1])
        if any(abs(float(a) - e) > 0.005 for a, e in zip(actual, expected)):
            raise RuntimeError(f"Cell {prim.GetPath()} has XY bounds {actual}, expected {expected}")
        codes[row][col] = TERRAIN_CODES[terrain]
    if any(code < 0 for row in codes for code in row):
        raise RuntimeError("The 3x3 terrain grid is incomplete")
    for point, label in ((case.start_xy, "start"), (case.goal_xy, "goal")):
        col = int((point[0] + 1) // 2)
        row = int((point[1] + 1) // 2)
        if not (0 <= row < 3 and 0 <= col < 3
                and abs(point[0] - 2 * col) <= 0.5
                and abs(point[1] - 2 * row) <= 0.5):
            raise RuntimeError(f"{label} {point} must be at least 0.5 m inside a cell")
    return codes


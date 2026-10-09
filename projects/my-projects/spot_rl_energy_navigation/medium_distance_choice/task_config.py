"""Task geometry and reward for the medium-distance fixed-goal experiment."""

import math
import re
from dataclasses import replace

from pxr import Usd, UsdGeom

from full_grid_energy_choice.config import FullGridGeometry, GEOMETRY, NAV_CFG_FULL
from rl_config import TERRAIN_USD

START_XY = (24.0, 3.0)  # row 1, column 4: ramp
GOAL_XY = (0.0, 15.0)  # row 5, column 0: rocks
START_YAW_RAD = math.atan2(GOAL_XY[1] - START_XY[1], GOAL_XY[0] - START_XY[0])

GEOMETRY_MEDIUM: FullGridGeometry = replace(
    GEOMETRY,
    start_xy=START_XY,
    goal_xy=GOAL_XY,
    start_yaw_rad=START_YAW_RAD,
    start_jitter_m=0.05,
)
NAV_CFG_MEDIUM = replace(
    NAV_CFG_FULL,
    max_episode_seconds=120.0,
    energy_penalty_per_j=0.01,
    success_bonus=100.0,
    failure_penalty=25.0,
)

TERRAIN_NAMES = (
    "t1_asphalt", "t2_slippery", "t3_ramp", "t4_fine_gravel",
    "t5_large_rocks", "t6_obstacles", "t7_stairs",
)
TERRAIN_LABELS = ("asphalt", "ice", "ramp", "gravel", "rocks", "obstacles", "stairs")
TERRAIN_CODES = {name: index for index, name in enumerate(TERRAIN_NAMES)}
EXPECTED_ENDPOINTS = {
    (1, 4): ("t3_ramp", START_XY),
    (5, 0): ("t5_large_rocks", GOAL_XY),
}


def read_terrain_codes() -> list[list[int]]:
    """Read the actual USD cell types; no terrain labels enter the policy input."""
    stage = Usd.Stage.Open(str(TERRAIN_USD))
    if stage is None:
        raise RuntimeError(f"Could not open terrain USD: {TERRAIN_USD}")
    grid = stage.GetPrimAtPath("/World/Grid")
    if not grid.IsValid():
        raise RuntimeError("/World/Grid is missing from terrain USD")
    codes = [[-1] * 7 for _ in range(7)]
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in grid.GetChildren():
        match = re.fullmatch(r"Cell_(\d+)_(\d+)_(.+)", prim.GetName())
        if not match:
            continue
        row, col, name = int(match[1]), int(match[2]), match[3]
        if not (0 <= row < 7 and 0 <= col < 7 and name in TERRAIN_CODES):
            raise RuntimeError(f"Unexpected cell: {prim.GetPath()}")
        if codes[row][col] != -1:
            raise RuntimeError(f"Duplicate terrain cell ({row}, {col})")
        codes[row][col] = TERRAIN_CODES[name]
        if (row, col) in EXPECTED_ENDPOINTS:
            expected_name, point = EXPECTED_ENDPOINTS[(row, col)]
            if name != expected_name:
                raise RuntimeError(f"Expected {expected_name} at ({row}, {col}), found {name}")
            box = bounds.ComputeWorldBound(prim).ComputeAlignedBox()
            lo, hi = box.GetMin(), box.GetMax()
            if not (lo[0] + 0.5 <= point[0] <= hi[0] - 0.5
                    and lo[1] + 0.5 <= point[1] <= hi[1] - 0.5):
                raise RuntimeError(f"Endpoint {point} is outside the interior of {prim.GetPath()}")
    if any(code < 0 for row in codes for code in row):
        raise RuntimeError("The terrain USD does not contain all 49 named cells")
    return codes

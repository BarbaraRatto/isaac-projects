"""Three training layouts and two unseen left/right swaps for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from binary_choice.config import BinaryCase
from binary_choice.cases.asphalt_rocks import CASE as ORIGINAL_CASE
from multi_terrain_binary_choice.layouts import TERRAIN


ROOT = Path(__file__).resolve().parents[2]
USD_DIR = ROOT / "terrain_generator" / "multi_terrain_binary_test2"
ASPHALT = TERRAIN["asphalt"]
ICE = TERRAIN["ice"]
SHIFT_M = 0.25


def route_grid(right: str, left: str) -> tuple[tuple[str, ...], ...]:
    """Right route crosses the top and right edge; left crosses left and bottom."""
    return ((ASPHALT, TERRAIN[right], TERRAIN[right]),
            (TERRAIN[left], ICE, TERRAIN[right]),
            (TERRAIN[left], TERRAIN[left], ASPHALT))


@dataclass(frozen=True)
class Layout:
    name: str
    split: str
    pair: tuple[str, str]
    costly_side: str
    grid: tuple[tuple[str, ...], ...]
    usd: Path

    def case(self, start_xy: tuple[float, float],
             goal_xy: tuple[float, float]) -> BinaryCase:
        return BinaryCase(self.name, self.usd, self.grid, start_xy, goal_xy)


OLD = Layout("original_asphalt_rocks", "train", ("asphalt", "rocks"), "right",
             ORIGINAL_CASE.grid, ORIGINAL_CASE.terrain_usd)
ROCKS = Layout("rocks_gravel__right_rocks", "train", ("rocks", "gravel"), "right",
               route_grid("rocks", "gravel"),
               USD_DIR / "train" / "rocks_gravel__right_rocks.usd")
OBSTACLES = Layout("obstacles_ramp__right_obstacles", "train",
                   ("obstacles", "ramp"), "right", route_grid("obstacles", "ramp"),
                   USD_DIR / "train" / "obstacles_ramp__right_obstacles.usd")
TRAIN_LAYOUTS = (OLD, ROCKS, OBSTACLES)
TEST_LAYOUTS = (
    Layout("rocks_gravel__right_gravel", "test", ("rocks", "gravel"), "left",
           route_grid("gravel", "rocks"),
           USD_DIR / "test" / "rocks_gravel__right_gravel.usd"),
    Layout("obstacles_ramp__right_ramp", "test", ("obstacles", "ramp"), "left",
           route_grid("ramp", "obstacles"),
           USD_DIR / "test" / "obstacles_ramp__right_ramp.usd"),
)
BY_NAME = {layout.name: layout for layout in TRAIN_LAYOUTS + TEST_LAYOUTS}


@dataclass(frozen=True)
class Condition:
    env_id: int
    layout: Layout
    geometry: str
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]

    @property
    def name(self) -> str:
        return f"{self.layout.name}__{self.geometry}__env{self.env_id}"

    @property
    def case(self) -> BinaryCase:
        return self.layout.case(self.start_xy, self.goal_xy)


def geometry_points(layout: Layout, geometry: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if layout is OLD:
        if geometry != "original":
            raise ValueError("Original map keeps its original start and goal")
        return ORIGINAL_CASE.start_xy, ORIGINAL_CASE.goal_xy
    if geometry == "centered":
        return (0.0, 0.0), (4.0, 4.0)
    if geometry == "shifted":
        # The nominally costly route is geometrically shorter on either side.
        shift = SHIFT_M if layout.costly_side == "right" else -SHIFT_M
        return (shift, 0.0), (4.0 + shift, 4.0)
    raise ValueError(f"Unknown geometry: {geometry}")


def training_conditions() -> tuple[Condition, ...]:
    # 12 rehearsal environments retain the already learned task; 10 each
    # explore the two new terrain pairs. All 32 share one PPO policy.
    layouts = (OLD,) * 12 + (ROCKS,) * 10 + (OBSTACLES,) * 10
    result = []
    for env_id, layout in enumerate(layouts):
        geometry = "original" if layout is OLD else "shifted"
        start, goal = geometry_points(layout, geometry)
        result.append(Condition(env_id, layout, geometry, start, goal))
    return tuple(result)


def evaluation_condition(layout_name: str, geometry: str) -> Condition:
    layout = BY_NAME[layout_name]
    start, goal = geometry_points(layout, geometry)
    return Condition(0, layout, geometry, start, goal)

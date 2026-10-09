"""Fixed, auditable train/test split for the multi-terrain binary choice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from binary_choice.config import BinaryCase


ROOT = Path(__file__).resolve().parents[2]
TERRAIN_DIR = ROOT / "terrain_generator" / "multi_terrain_binary_choice"
TERRAIN = {
    "asphalt": "t1_asphalt", "ice": "t2_slippery", "ramp": "t3_ramp",
    "gravel": "t4_fine_gravel", "rocks": "t5_large_rocks",
    "obstacles": "t6_obstacles", "stairs": "t7_stairs",
}
START_TERRAIN = TERRAIN["gravel"]
ICE = TERRAIN["ice"]

# The costly label is provisional: it comes from the previous J/m survey.
# Stairs are treated as costly because of fall risk, not their measured J/m.
PAIRS = (
    ("rocks", "gravel", "rocks"),
    ("ramp", "gravel", "gravel"),
    ("obstacles", "gravel", "gravel"),
    ("asphalt", "ramp", "ramp"),
    ("asphalt", "rocks", "rocks"),
    ("stairs", "rocks", "stairs"),
    ("ramp", "rocks", "rocks"),
    ("rocks", "obstacles", "rocks"),
)


def grid_from_routes(right: tuple[str, str, str],
                     left: tuple[str, str, str]) -> tuple[tuple[str, ...], ...]:
    """Right goes across the top then down; left goes down then across."""
    return (
        (START_TERRAIN, TERRAIN[right[0]], TERRAIN[right[1]]),
        (TERRAIN[left[0]], ICE, TERRAIN[right[2]]),
        (TERRAIN[left[1]], TERRAIN[left[2]], START_TERRAIN),
    )


@dataclass(frozen=True)
class Layout:
    name: str
    split: str
    pair: tuple[str, str]
    costly: str
    costly_side: str
    grid: tuple[tuple[str, ...], ...]

    @property
    def usd(self) -> Path:
        return TERRAIN_DIR / self.split / f"{self.name}.usd"

    def case(self, start_xy: tuple[float, float],
             goal_xy: tuple[float, float]) -> BinaryCase:
        return BinaryCase(self.name, self.usd, self.grid, start_xy, goal_xy)


def _train_layouts() -> tuple[Layout, ...]:
    result = []
    for first, second, costly in PAIRS:
        for right, left in ((first, second), (second, first)):
            result.append(Layout(
                name=f"{first}_{second}__right_{right}", split="train",
                pair=(first, second), costly=costly,
                costly_side="right" if right == costly else "left",
                grid=grid_from_routes((right,) * 3, (left,) * 3),
            ))
    return tuple(result)


TRAIN_LAYOUTS = _train_layouts()

# Held-out arrangements use the same terrain pairs but different cell order.
# They are never referenced by training_conditions().
TEST_LAYOUTS = (
    Layout("asphalt_rocks__mixed", "test", ("asphalt", "rocks"), "rocks", "left",
           grid_from_routes(("asphalt", "rocks", "asphalt"),
                            ("rocks", "asphalt", "rocks"))),
    Layout("ramp_gravel__mixed", "test", ("ramp", "gravel"), "gravel", "right",
           grid_from_routes(("gravel", "ramp", "gravel"),
                            ("ramp", "gravel", "ramp"))),
    Layout("obstacles_gravel__mixed", "test", ("obstacles", "gravel"), "gravel", "left",
           grid_from_routes(("obstacles", "gravel", "obstacles"),
                            ("gravel", "obstacles", "gravel"))),
    Layout("stairs_rocks__mixed", "test", ("stairs", "rocks"), "stairs", "right",
           grid_from_routes(("stairs", "stairs", "rocks"),
                            ("rocks", "rocks", "rocks"))),
)
ALL_LAYOUTS = TRAIN_LAYOUTS + TEST_LAYOUTS
BY_NAME = {layout.name: layout for layout in ALL_LAYOUTS}


@dataclass(frozen=True)
class Condition:
    env_id: int
    layout: Layout
    geometry: str
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]

    @property
    def name(self) -> str:
        return f"{self.layout.name}__{self.geometry}"

    @property
    def case(self) -> BinaryCase:
        return self.layout.case(self.start_xy, self.goal_xy)


def geometry_points(layout: Layout, geometry: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if geometry == "centered":
        return (0.0, 0.0), (4.0, 4.0)
    if geometry != "short_costly":
        raise ValueError(f"Unknown geometry: {geometry}")
    shift = 0.5 if layout.costly_side == "right" else -0.5
    return (shift, 0.0), (4.0 + shift, 4.0)


def training_conditions() -> tuple[Condition, ...]:
    result = []
    for layout in TRAIN_LAYOUTS:
        for geometry in ("centered", "short_costly"):
            start, goal = geometry_points(layout, geometry)
            result.append(Condition(len(result), layout, geometry, start, goal))
    assert len(result) == 32 and len({item.name for item in result}) == 32
    return tuple(result)


def evaluation_condition(layout_name: str, geometry: str) -> Condition:
    layout = BY_NAME[layout_name]
    start, goal = geometry_points(layout, geometry)
    return Condition(0, layout, geometry, start, goal)

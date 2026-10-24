"""The two existing test4 USD maps, with a local test7 assignment manifest."""

from dataclasses import dataclass
from pathlib import Path

from multi_terrain_test7.config import BinaryCase


USD_DIR = Path(__file__).resolve().parents[2] / "terrain_generator" / "multi_terrain_binary_test2"
START_XY = (0.0, 0.0)
GOAL_XY = (4.0, 4.0)


@dataclass(frozen=True)
class Layout:
    name: str
    grid: tuple[tuple[str, ...], ...]
    usd: Path
    costly_side: str
    pair: tuple[str, str] = ("rocks", "gravel")
    split: str = "train"


@dataclass(frozen=True)
class Condition:
    env_id: int
    layout: Layout
    geometry: str = "centered"
    start_xy: tuple[float, float] = START_XY
    goal_xy: tuple[float, float] = GOAL_XY

    @property
    def name(self) -> str:
        return f"{self.layout.name}__{self.geometry}__env{self.env_id}"

    @property
    def case(self) -> BinaryCase:
        return BinaryCase(self.layout.name, self.layout.usd, self.layout.grid,
                          self.start_xy, self.goal_xy)


def route_grid(right: str, left: str) -> tuple[tuple[str, ...], ...]:
    return (("t1_asphalt", right, right),
            (left, "t2_slippery", right),
            (left, left, "t1_asphalt"))


ROCKS = Layout(
    "rocks_gravel__right_rocks", route_grid("t5_large_rocks", "t4_fine_gravel"),
    USD_DIR / "train" / "rocks_gravel__right_rocks.usd", "right",
)
MIRRORED = Layout(
    "rocks_gravel__right_gravel", route_grid("t4_fine_gravel", "t5_large_rocks"),
    USD_DIR / "test" / "rocks_gravel__right_gravel.usd", "left",
)
TRAIN_LAYOUTS = (ROCKS, MIRRORED)
BY_NAME = {layout.name: layout for layout in TRAIN_LAYOUTS}


def training_conditions() -> tuple[Condition, ...]:
    return tuple(Condition(i, layout)
                 for i, layout in enumerate((ROCKS,) * 16 + (MIRRORED,) * 16))


def evaluation_condition(layout_name: str) -> Condition:
    return Condition(0, BY_NAME[layout_name])

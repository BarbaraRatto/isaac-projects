"""The two small mirrored USD boards, both explicitly assigned to training."""

from dataclasses import dataclass
from pathlib import Path

from multi_terrain_test11.config import BinaryCase


USD_DIR = Path(__file__).resolve().parent / "maps"
START_XY = (0.0, 0.0)
GOAL_XY = (4.0, 4.0)


@dataclass(frozen=True)
class Layout:
    name: str
    grid: tuple[tuple[str, ...], ...]
    usd: Path
    costly_side: str
    pair: tuple[str, str] = ("rocks", "asphalt")
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
    "rocks_asphalt__right_rocks", route_grid("t5_large_rocks", "t1_asphalt"),
    USD_DIR / "rocks_right.usd", "right",
)
MIRRORED = Layout(
    "rocks_asphalt__right_asphalt", route_grid("t1_asphalt", "t5_large_rocks"),
    USD_DIR / "rocks_left.usd", "left",
)
TRAIN_LAYOUTS = (ROCKS, MIRRORED)
BY_NAME = {layout.name: layout for layout in TRAIN_LAYOUTS}


def training_conditions(num_envs: int = 32) -> tuple[Condition, ...]:
    if num_envs < 2 or num_envs % 2:
        raise ValueError('Use an even number of environments to balance mirrored maps')
    return tuple(Condition(i, TRAIN_LAYOUTS[i % 2]) for i in range(num_envs))


def evaluation_condition(layout_name: str) -> Condition:
    return Condition(0, BY_NAME[layout_name])

"""Shared physics, reward, PPO and case selection for binary terrain choices."""

from __future__ import annotations

import importlib
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

from navigation_config import NAV_CFG, NavigationConfig


@dataclass(frozen=True)
class BinaryGeometry:
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    start_yaw_rad: float
    start_jitter_m: float = 0.05
    world_x_bounds: tuple[float, float] = (-0.8, 4.8)
    world_y_bounds: tuple[float, float] = (-0.8, 4.8)


@dataclass(frozen=True)
class BinaryCase:
    name: str
    terrain_usd: Path
    grid: tuple[tuple[str, ...], ...]
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    cell_size_m: float = 2.0
    start_jitter_m: float = 0.05

    @property
    def geometry(self) -> BinaryGeometry:
        width = len(self.grid[0]) * self.cell_size_m
        height = len(self.grid) * self.cell_size_m
        return BinaryGeometry(
            start_xy=self.start_xy,
            goal_xy=self.goal_xy,
            start_yaw_rad=math.atan2(self.goal_xy[1] - self.start_xy[1],
                                     self.goal_xy[0] - self.start_xy[0]),
            start_jitter_m=self.start_jitter_m,
            world_x_bounds=(-self.cell_size_m / 2 + 0.2,
                            width - self.cell_size_m / 2 - 0.2),
            world_y_bounds=(-self.cell_size_m / 2 + 0.2,
                            height - self.cell_size_m / 2 - 0.2),
        )


@dataclass(frozen=True)
class PPOSettings:
    timesteps: int = 204_800
    num_envs: int = 32
    n_steps_per_env: int = 32
    batch_size: int = 256
    n_epochs: int = 5
    learning_rate: float = 3e-4
    gamma: float = 0.999
    net_arch: tuple[int, int] = (128, 128)


PPO_CFG = PPOSettings()
NAV_CFG_BINARY: NavigationConfig = replace(
    NAV_CFG,
    max_episode_seconds=40.0,
    goal_radius_m=0.55,
    max_forward_m_s=1.6,
    max_lateral_m_s=0.9,
    max_yaw_rad_s=1.5,
    progress_reward_per_m=3.0,
    energy_penalty_per_j=0.02,
    time_penalty_per_step=0.01,
    success_bonus=60.0,
    failure_penalty=25.0,
)

TERRAIN_NAMES = (
    "t1_asphalt", "t2_slippery", "t3_ramp", "t4_fine_gravel",
    "t5_large_rocks", "t6_obstacles", "t7_stairs",
)
TERRAIN_LABELS = ("asphalt", "ice", "ramp", "gravel", "rocks", "obstacles", "stairs")
TERRAIN_CODES = dict(zip(TERRAIN_NAMES, range(len(TERRAIN_NAMES))))


def load_case(name: str) -> BinaryCase:
    """Load a named case without changing shared training code."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"Invalid case name: {name!r}")
    try:
        module = importlib.import_module(f"binary_choice.cases.{name}")
    except ModuleNotFoundError as exc:
        if exc.name == f"binary_choice.cases.{name}":
            raise ValueError(f"Unknown binary-choice case {name!r}") from exc
        raise
    case = module.CASE
    if not isinstance(case, BinaryCase) or case.name != name:
        raise RuntimeError(f"Case module {name!r} must define CASE with the same name")
    if len(case.grid) != 3 or any(len(row) != 3 for row in case.grid):
        raise RuntimeError(f"Case {name!r} must use a 3x3 grid")
    if case.cell_size_m != 2.0:
        raise RuntimeError(f"Case {name!r} must use 2 m cells")
    if any(terrain not in TERRAIN_CODES for row in case.grid for terrain in row):
        raise RuntimeError(f"Case {name!r} contains an unsupported terrain")
    return case

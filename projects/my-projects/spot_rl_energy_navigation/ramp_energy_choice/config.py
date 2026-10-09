"""Fixed rocks-to-rocks task with a ramp and an asphalt detour.

Cell coordinates are centres in metres. The direct route runs along +X.
The asphalt detour lies to Spot's right (negative Y) at the initial yaw.
"""

from dataclasses import dataclass, replace

from navigation_config import NAV_CFG, NavigationConfig


@dataclass(frozen=True)
class RampChoiceGeometry:
    start_xy: tuple[float, float] = (19.5, 3.0)  # inside Cell_1_3_t5_large_rocks
    goal_xy: tuple[float, float] = (28.5, 3.0)   # inside Cell_1_5_t5_large_rocks
    start_yaw_rad: float = 0.0
    start_jitter_m: float = 0.10
    world_x_bounds: tuple[float, float] = (15.1, 32.9)
    world_y_bounds: tuple[float, float] = (-1.4, 4.4)
    ramp_x_bounds: tuple[float, float] = (21.0, 27.0)
    ramp_y_bounds: tuple[float, float] = (1.5, 4.5)
    asphalt_x_bounds: tuple[float, float] = (15.0, 33.0)
    asphalt_y_bounds: tuple[float, float] = (-1.5, 1.5)


GEOMETRY = RampChoiceGeometry()
NAV_CFG_RAMP: NavigationConfig = replace(
    NAV_CFG,
    max_episode_seconds=80.0,
    goal_radius_m=0.55,
    max_forward_m_s=0.9,
    success_bonus=25.0,
    failure_penalty=12.0,
    # Keep goal progress/safety, while making total mechanical energy
    # more influential than a small difference in travel time.
    energy_penalty_per_j=0.005,
    time_penalty_per_step=0.01,
)

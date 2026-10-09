"""New fixed start/goal task on the complete original terrain USD.

Coordinates use the USD's world X/Y axes. Cell names use zero-based indices:
Cell_2_1 is the ramp; Cell_4_1 is the obstacles cell in the image.
The older experiments keep their original command limits.
"""

import math
from dataclasses import dataclass, replace

from navigation_config import NAV_CFG, NavigationConfig


@dataclass(frozen=True)
class FullGridGeometry:
    # 0.7 m inside the ramp's left/top edges: X[3,9], Y[4.5,7.5].
    start_xy: tuple[float, float] = (3.7, 6.8)
    # 0.7 m inside the obstacles cell's left/bottom edges:
    # Cell_4_1_t6_obstacles has X[3,9], Y[10.5,13.5].
    goal_xy: tuple[float, float] = (3.7, 11.2)
    start_yaw_rad: float = math.pi / 2  # Toward the goal, along +Y.
    start_jitter_m: float = 0.05
    # The complete 7x7 terrain stays present. These bounds only mark the
    # physical edge of the board, with a small safety inset.
    world_x_bounds: tuple[float, float] = (-2.8, 38.8)
    world_y_bounds: tuple[float, float] = (-1.3, 19.3)
    asphalt_x_bounds: tuple[float, float] = (-3.0, 3.0)  # Cell_3_0
    asphalt_y_bounds: tuple[float, float] = (7.5, 10.5)
    rocks_x_bounds: tuple[float, float] = (3.0, 9.0)    # Cell_3_1
    rocks_y_bounds: tuple[float, float] = (7.5, 10.5)
    obstacle_x_bounds: tuple[float, float] = (3.0, 9.0) # Cell_4_1
    obstacle_y_bounds: tuple[float, float] = (10.5, 13.5)


GEOMETRY = FullGridGeometry()
NAV_CFG_FULL: NavigationConfig = replace(
    NAV_CFG,
    max_episode_seconds=80.0,
    goal_radius_m=0.55,
    max_forward_m_s=1.6,  # symmetric: -1.6 to +1.6 m/s
    max_lateral_m_s=0.9,  # -0.9 to +0.9 m/s
    max_yaw_rad_s=1.5,   # -1.5 to +1.5 rad/s
    success_bonus=25.0,
    failure_penalty=12.0,
    energy_penalty_per_j=0.005,
    time_penalty_per_step=0.01,
)

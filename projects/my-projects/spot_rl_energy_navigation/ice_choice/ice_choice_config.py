"""Shared geometry and training settings for the original-USD ice choice task."""

from dataclasses import dataclass, replace

from navigation_config import NAV_CFG, NavigationConfig


@dataclass(frozen=True)
class IceChoiceConfig:
    start_xy: tuple[float, float] = (18.0, 3.0)   # Cell_1_3_t5_large_rocks
    goal_xy: tuple[float, float] = (24.0, 9.0)    # Cell_3_4_t1_asphalt
    start_yaw_rad: float = 0.7853981633974483    # face between ice and ramps
    start_jitter_m: float = 0.10
    # Exploration remains in the region covered by the visual feature atlas.
    world_x_bounds: tuple[float, float] = (12.5, 29.5)
    world_y_bounds: tuple[float, float] = (-1.0, 13.0)
    ice_x_bounds: tuple[float, float] = (15.0, 21.0)
    ice_y_bounds: tuple[float, float] = (4.5, 7.5)


ICE_GEOMETRY = IceChoiceConfig()
ICE_NAV_CFG: NavigationConfig = replace(
    NAV_CFG,
    max_episode_seconds=80.0,
    goal_radius_m=0.55,
    max_forward_m_s=0.9,
    success_bonus=25.0,
    failure_penalty=12.0,
    time_penalty_per_step=0.03,
)

"""Fixed geometry and guidance paths for the first two-route energy test."""

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TwoRouteConfig:
    start_xyz: tuple[float, float, float] = (1.5, 0.0, 0.75)
    goal_xy: tuple[float, float] = (34.5, 0.0)
    goal_radius_m: float = 0.4
    settle_seconds: float = 2.0
    max_walk_seconds: float = 90.0
    min_body_height_m: float = 0.25
    max_lateral_m_s: float = 0.45
    max_yaw_rad_s: float = 0.8
    waypoint_radius_m: float = 0.35
    # x increases toward the goal. The bypass stays on the asphalt cells above
    # the demanding central cell.
    direct_waypoints: tuple[tuple[float, float], ...] = (
        (5.0, 0.0), (7.0, 0.0), (11.0, 0.0), (13.0, 0.0),
        (17.0, 0.0), (19.0, 0.0), (23.0, 0.0), (25.0, 0.0),
        (29.0, 0.0), (31.0, 0.0), (34.5, 0.0)
    )
    bypass_waypoints: tuple[tuple[float, float], ...] = (
        (5.0, 0.0), (7.0, 2.0), (31.0, 2.0), (33.0, 0.0), (34.5, 0.0)
    )
    csv_path: Path = Path(__file__).resolve().parent / "results" / "route_baselines.csv"

    def nominal_length(self, route: str) -> float:
        points = (self.start_xyz[:2], *self.waypoints(route))
        return sum(math.dist(a, b) for a, b in zip(points, points[1:]))

    def waypoints(self, route: str) -> tuple[tuple[float, float], ...]:
        if route == "direct":
            return self.direct_waypoints
        if route == "bypass":
            return self.bypass_waypoints
        raise ValueError(f"Unknown route: {route}")


ROUTE_CFG = TwoRouteConfig()

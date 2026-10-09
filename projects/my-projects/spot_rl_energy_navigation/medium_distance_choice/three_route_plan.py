"""Provisional XY centerlines transcribed from the user's three-color sketch.

These are route proposals for review. They have not been simulated or measured.
All coordinates are local to one complete copy of real_terrains.usd.
"""

import math

START = (24.0, 3.0)  # r1c4 ramp
GOAL = (0.0, 15.0)  # r5c0 rocks

# Pink takes the high, nearly shortest corridor. The bend in r3c4 keeps
# its body center off r2c3 ice before it reaches the obstacle-cell center
# (18, 9). It follows the y=10.5 boundary between stair rows, then crosses
# the center of r4c1 obstacles (6, 12). Foot clearance is unverified.
PINK_KNOTS = (
    START,
    (22.0, 7.8),
    (18.0, 9.0),
    (15.0, 10.5),
    (9.0, 10.5),
    (6.0, 12.0),
    GOAL,
)

# Orange cuts west along row 1, then diagonally through the ramp in r2c1
# and rocks in r3c1. It enters column 0 before row 4 to avoid r4c1 obstacles.
# The corner at (2.5, 10.3) gives the body center a small 0.5 m clearance
# from the obstacle-cell boundary; actual foot clearance needs simulation.
ORANGE_KNOTS = (START, (8.4, 3.0), (2.5, 10.3), GOAL)

# Yellow shares the westward leg, then curves through r2c1 ramp toward the
# asphalt in column 0 and climbs to the same goal.
YELLOW_WEST_END = (6.0, 3.0)
YELLOW_CURVES = (
    ((6.0, 3.0), (5.1, 3.3), (3.8, 5.4)),
    ((3.8, 5.4), (1.9, 7.1), (1.6, 8.6)),
)

ROUTE_NAMES = ("pink", "orange", "yellow")


def _sample_line(start, end, spacing):
    steps = max(1, math.ceil(math.dist(start, end) / spacing))
    return [tuple(a + (b - a) * index / steps for a, b in zip(start, end))
            for index in range(1, steps + 1)]


def _sample_quadratic(start, control, end, spacing):
    length_estimate = math.dist(start, control) + math.dist(control, end)
    steps = max(1, math.ceil(length_estimate / spacing))
    return [((1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0],
             (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1])
            for index in range(1, steps + 1) for t in (index / steps,)]


def route_points(name: str, spacing: float = 0.15) -> list[tuple[float, float]]:
    """Return a dense XY polyline suitable for drawing or later path following."""
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if name not in ROUTE_NAMES:
        raise ValueError(f"Unknown route {name!r}; choose from {ROUTE_NAMES}")
    points = [START]
    if name == "yellow":
        points += _sample_line(START, YELLOW_WEST_END, spacing)
        for start, control, end in YELLOW_CURVES:
            points += _sample_quadratic(start, control, end, spacing)
        points += _sample_line(YELLOW_CURVES[-1][-1], GOAL, spacing)
    else:
        knots = PINK_KNOTS if name == "pink" else ORANGE_KNOTS
        for start, end in zip(knots, knots[1:]):
            points += _sample_line(start, end, spacing)
    assert points[0] == START and points[-1] == GOAL
    return points

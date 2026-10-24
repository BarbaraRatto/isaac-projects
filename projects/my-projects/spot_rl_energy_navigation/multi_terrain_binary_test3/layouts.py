"""Reuse the existing USDs: two mirrored rocks/gravel boards plus rehearsal."""

from __future__ import annotations

from dataclasses import replace

from multi_terrain_binary_test2.layouts import (
    Condition, OLD, ROCKS, TEST_LAYOUTS, geometry_points,
)


# The mirrored board was held out in test2. It is deliberately promoted to a
# training board in test3; its USD file remains at its original path.
ROCKS_MIRRORED = replace(TEST_LAYOUTS[0], split="train")
TRAIN_LAYOUTS = (OLD, ROCKS, ROCKS_MIRRORED)
BY_NAME = {layout.name: layout for layout in TRAIN_LAYOUTS}


def training_conditions() -> tuple[Condition, ...]:
    layouts = (OLD,) * 4 + (ROCKS,) * 14 + (ROCKS_MIRRORED,) * 14
    conditions = []
    for env_id, layout in enumerate(layouts):
        geometry = "original" if layout is OLD else "shifted"
        start, goal = geometry_points(layout, geometry)
        conditions.append(Condition(env_id, layout, geometry, start, goal))
    assert len(conditions) == 32
    return tuple(conditions)


def evaluation_condition(layout_name: str, geometry: str) -> Condition:
    layout = BY_NAME[layout_name]
    start, goal = geometry_points(layout, geometry)
    return Condition(0, layout, geometry, start, goal)

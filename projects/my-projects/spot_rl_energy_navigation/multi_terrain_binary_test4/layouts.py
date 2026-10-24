"""Balanced, centered rocks/gravel layouts for a fresh shared policy."""

from __future__ import annotations

from dataclasses import replace

from multi_terrain_binary_test2.layouts import Condition, ROCKS, TEST_LAYOUTS


ROCKS_MIRRORED = replace(TEST_LAYOUTS[0], split="train")
TRAIN_LAYOUTS = (ROCKS, ROCKS_MIRRORED)
BY_NAME = {layout.name: layout for layout in TRAIN_LAYOUTS}
START_XY = (0.0, 0.0)
GOAL_XY = (4.0, 4.0)


def training_conditions() -> tuple[Condition, ...]:
    """Sixteen Spots per layout, with the same start, goal and yaw."""
    layouts = (ROCKS,) * 16 + (ROCKS_MIRRORED,) * 16
    return tuple(Condition(i, layout, "centered", START_XY, GOAL_XY)
                 for i, layout in enumerate(layouts))


def evaluation_condition(layout_name: str) -> Condition:
    """Evaluate exactly one of the two conditions seen in training."""
    layout = BY_NAME[layout_name]
    return Condition(0, layout, "centered", START_XY, GOAL_XY)

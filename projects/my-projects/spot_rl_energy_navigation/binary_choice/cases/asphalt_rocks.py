"""Case 1: asphalt detour versus rocks, with ice on the direct diagonal."""

from pathlib import Path

from binary_choice.config import BinaryCase


CASE = BinaryCase(
    name="asphalt_rocks",
    terrain_usd=(Path(__file__).resolve().parents[3]
                 / "terrain_generator" / "binary_choice_asphalt_rocks.usd"),
    # Row 0 has y=0 m; row 2 has y=4 m. Column 0 has x=0 m.
    grid=(
        ("t4_fine_gravel", "t5_large_rocks", "t5_large_rocks"),
        ("t1_asphalt", "t2_slippery", "t5_large_rocks"),
        ("t1_asphalt", "t1_asphalt", "t4_fine_gravel"),
    ),
    start_xy=(0.0, 0.0),
    goal_xy=(4.0, 4.0),
)

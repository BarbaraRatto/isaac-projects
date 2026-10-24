"""Evaluation-only mixed layouts. Never added to the training manifest."""

from pathlib import Path

from multi_terrain_test6.layouts import Condition, Layout


MAP_DIR = Path(__file__).resolve().parent / "maps" / "heldout"
A, R, G, I = "t1_asphalt", "t5_large_rocks", "t4_fine_gravel", "t2_slippery"
MIXED_1 = Layout(
    "rocks_gravel__mixed_1",
    ((A, R, G), (G, I, G), (R, R, A)),
    MAP_DIR / "rocks_gravel__mixed_1.usd",
    costly_side="unmeasured", split="heldout",
)
MIXED_2 = Layout(
    "rocks_gravel__mixed_2",
    ((A, G, R), (R, I, R), (G, G, A)),
    MAP_DIR / "rocks_gravel__mixed_2.usd",
    costly_side="unmeasured", split="heldout",
)
HELDOUT_BY_NAME = {layout.name: layout for layout in (MIXED_1, MIXED_2)}


def heldout_condition(name: str) -> Condition:
    return Condition(0, HELDOUT_BY_NAME[name])

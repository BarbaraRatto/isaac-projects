"""Create the two train and two held-out 2 m/cell USD boards; no simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spot_rl_energy_navigation"))

from create_binary_choice_asphalt_rocks import generate  # noqa: E402
from binary_choice.terrain import read_terrain_codes  # noqa: E402
from multi_terrain_binary_test2.layouts import ROCKS, OBSTACLES, TEST_LAYOUTS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for layout in (ROCKS, OBSTACLES, *TEST_LAYOUTS):
        if layout.usd.is_file() and not args.overwrite:
            read_terrain_codes(layout.case((0.0, 0.0), (4.0, 4.0)))
            print(f"[USD] already valid: {layout.usd}")
            continue
        generate(HERE / "real_terrains.usd", layout.usd,
                 overwrite=layout.usd.exists(), grid=layout.grid)
        read_terrain_codes(layout.case((0.0, 0.0), (4.0, 4.0)))
        print(f"[USD] ready: {layout.usd}")


if __name__ == "__main__":
    main()

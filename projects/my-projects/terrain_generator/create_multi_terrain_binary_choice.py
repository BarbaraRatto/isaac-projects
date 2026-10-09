"""Build the fixed train/test 3x3 USD catalog without starting Isaac Sim.

Use Isaac's Python because it includes pxr. The terrain surfaces and material
bindings come from real_terrains.usd via create_binary_choice_asphalt_rocks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "spot_rl_energy_navigation"))

from create_binary_choice_asphalt_rocks import generate  # noqa: E402
from binary_choice.terrain import read_terrain_codes  # noqa: E402
from multi_terrain_binary_choice.layouts import TEST_LAYOUTS, TRAIN_LAYOUTS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    layouts = (TRAIN_LAYOUTS + TEST_LAYOUTS if args.split == "all" else
               TRAIN_LAYOUTS if args.split == "train" else TEST_LAYOUTS)
    for index, layout in enumerate(layouts, 1):
        if layout.usd.is_file() and not args.overwrite:
            try:
                read_terrain_codes(layout.case((0.0, 0.0), (4.0, 4.0)))
            except Exception as exc:
                print(f"[USD] rebuilding incomplete {layout.name}: {exc}", flush=True)
            else:
                print(f"[USD] {index}/{len(layouts)} already valid: {layout.usd}", flush=True)
                continue
        print(f"[USD] {index}/{len(layouts)} building {layout.split}/{layout.name}", flush=True)
        generate(HERE / "real_terrains.usd", layout.usd, overwrite=layout.usd.exists(),
                 grid=layout.grid)
    print(f"[USD] ready={len(layouts)} split={args.split}", flush=True)


if __name__ == "__main__":
    main()

"""Launch 20 test6 episodes on new mixed map 1: A R G / G I G / R R A."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multi_terrain_test6.launch_heldout import main


if __name__ == "__main__":
    main("rocks_gravel__mixed_1")

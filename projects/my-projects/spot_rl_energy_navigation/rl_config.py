"""Paths and parameters for the single-Spot Isaac Lab locomotion check."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ISAAC_ROBOTICS = PROJECT_ROOT / "IsaacRobotics"

TERRAIN_USD = PROJECT_ROOT / "terrain_generator" / "real_terrains.usd"
ROBOT_USD = ISAAC_ROBOTICS / "assets" / "spot.usd"
POLICY_WEIGHTS = ISAAC_ROBOTICS / "policies" / "spot" / "models" / "spot_policy.pt"
POLICY_PARAMS = ISAAC_ROBOTICS / "policies" / "spot" / "params" / "env.yaml"

# Match spot_warehouse.py for the first comparison with the existing simulation.
PHYSICS_DT = 1.0 / 200.0
SPAWN_POSITION = (-1.3, 8.25, 0.75)

STAND_SECONDS = 1.0
WALK_SECONDS = 3.0
STOP_SECONDS = 1.0
WALK_COMMAND = (0.9, 0.0, 0.0)  # vx, vy, yaw rate


# Three representative surfaces for the first energy benchmark.
BENCHMARK_TERRAINS = ("t1_asphalt", "t4_fine_gravel", "t5_large_rocks")
BENCHMARK_REPEATS = 3
BENCHMARK_SETTLE_SECONDS = 2.0
BENCHMARK_DISTANCE_M = 2.0
BENCHMARK_MAX_WALK_SECONDS = 5.0
BENCHMARK_START_BACK_M = 1.3
BENCHMARK_EDGE_MARGIN_M = 0.35
BENCHMARK_FOOT_MARGIN_M = 0.05
BENCHMARK_MIN_BODY_HEIGHT_M = 0.25
BENCHMARK_SPAWN_HEIGHT_M = 0.75
BENCHMARK_CSV = Path(__file__).resolve().parent / "results" / "terrain_comparison.csv"


def validate_assets() -> None:
    """Fail before launching a long simulation if an input asset is missing."""
    paths = (TERRAIN_USD, ROBOT_USD, POLICY_WEIGHTS, POLICY_PARAMS)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing assets:\n" + "\n".join(missing))

"""Read-only references to existing robot, camera, locomotion and terrain assets."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD = PROJECT_ROOT / "IsaacRobotics" / "assets" / "spot.usd"
POLICY_WEIGHTS = PROJECT_ROOT / "IsaacRobotics" / "policies" / "spot" / "models" / "spot_policy.pt"
ZED_USD = Path("/home/isaac/isaac-sim/assets/Assets/Isaac/5.1/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc")
ZED_CAMERA_PRIM = "/Root/base_link/ZED_X/CameraRight"


def validate_assets() -> None:
    from multi_terrain_test10.layouts import TRAIN_LAYOUTS
    paths = (ROBOT_USD, POLICY_WEIGHTS, ZED_USD, *(item.usd for item in TRAIN_LAYOUTS))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing assets:\n" + "\n".join(missing))

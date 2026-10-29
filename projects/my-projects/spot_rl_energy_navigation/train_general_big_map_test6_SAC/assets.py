"""Read-only references to existing robot, camera, locomotion and terrain assets."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD = PROJECT_ROOT / "IsaacRobotics" / "assets" / "spot.usd"
POLICY_WEIGHTS = PROJECT_ROOT / "IsaacRobotics" / "policies" / "spot" / "models" / "spot_policy.pt"
_LOCAL_ZED = PROJECT_ROOT / ".cloud_assets" / "ZED_X.usdc"
_ORIGINAL_ZED = Path("/home/isaac/isaac-sim/assets/Assets/Isaac/5.1/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc")
ZED_USD = Path(os.environ.get("SPOT_ZED_USD", str(_LOCAL_ZED if _LOCAL_ZED.is_file() else _ORIGINAL_ZED)))
ZED_CAMERA_PRIM = "/Root/base_link/ZED_X/CameraRight"


def validate_assets() -> None:
    from train_general_big_map_test6_SAC.layouts import TRAIN_LAYOUTS
    paths = (ROBOT_USD, POLICY_WEIGHTS, ZED_USD, *(item.usd for item in TRAIN_LAYOUTS))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing assets:\n" + "\n".join(missing))

    import hashlib
    import json
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'artifacts/manifest.json').read_text())
    for name, expected in manifest['identity']['terrain'].items():
        layout = next(item for item in TRAIN_LAYOUTS if item.name == name)
        if hashlib.sha256(layout.usd.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen terrain changed; rebuild this experiment deliberately instead of mixing assets')
    for name, expected in manifest['identity']['missions'].items():
        if hashlib.sha256((root/'artifacts'/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen mission artifact changed: {name}')

"""Local paths and metadata; importing this module never starts a simulator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = ("asphalt", "ice", "ramp", "gravel", "rocks", "obstacles", "stairs")


def local_path(value: str | Path) -> Path:
    path = Path(value)
    path = (HERE / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(HERE):
        raise ValueError(f"Path must stay inside {HERE}: {path}")
    return path


def read_config(path: str | Path = "config.json") -> dict:
    return json.loads(local_path(path).read_text())


def camera_optics(config: dict) -> dict:
    """Convert JSON values to the scalar/vector types expected by USD cameras."""
    camera = config["camera"]
    return {
        "focal_length": float(camera["focal_length"]),
        "horizontal_aperture": float(camera["horizontal_aperture"]),
        "vertical_aperture": float(camera["vertical_aperture"]),
        # USD accepts a tuple for GfVec2f, but rejects the list read from JSON.
        "clipping_range": tuple(float(v) for v in camera["clipping_range"]),
    }


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local_assets() -> dict:
    manifest = json.loads((HERE / "assets/manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = local_path(name)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Missing or modified bundled file: {name}")
    return manifest

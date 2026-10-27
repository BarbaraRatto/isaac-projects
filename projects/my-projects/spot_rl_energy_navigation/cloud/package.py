#!/usr/bin/env python3
"""Package selected experiments and exact local assets, including Git-ignored files.

No credentials, git history, logs, or previous training runs are included.
The resulting archive is self-contained: cloning the repo on the GPU is optional.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
NAV = HERE.parent
EXCLUDED = {"runs", "results", "analysis", "__pycache__", ".pytest_cache", "dist", ".git"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_files(experiments, zed):
    """Use an allowlist of roots; reject symlinks escaping those roots."""
    selected = {}
    roots = [HERE, PROJECT / "IsaacRobotics/assets",
             PROJECT / "IsaacRobotics/policies/spot/models/spot_policy.pt",
             NAV / "extracted_features_comparison_v2/models/dinov2"]
    for name in experiments:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError(f"Invalid experiment name: {name}")
        root = NAV / name
        if not (root / "train.py").is_file():
            raise FileNotFoundError(f"Experiment missing: {root}/train.py")
        roots.append(root)
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Required asset missing: {root}")
        paths = root.rglob("*") if root.is_dir() else [root]
        for path in paths:
            relative = path.relative_to(PROJECT)
            if any(part in EXCLUDED for part in relative.parts):
                continue
            if path.name in {"env.sh", "bundle-manifest.json"} or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError(f"Copy this symlink to a real file before packaging: {path}")
            if path.is_file():
                selected[relative.as_posix()] = path
    if not zed.is_file():
        raise FileNotFoundError(f"ZED asset missing: {zed}; supply --zed /path/to/ZED_X.usdc")
    # This existing camera asset is copied verbatim, rather than using a new revision.
    selected[".cloud_assets/ZED_X.usdc"] = zed
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", help="Repeat to add experiments; default: test3")
    parser.add_argument("--zed", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=HERE / "dist/spot-cloud-training.tar.gz")
    args = parser.parse_args()
    experiments = args.experiment or ["train_general_big_map_test3"]
    local_zed = PROJECT / ".cloud_assets/ZED_X.usdc"
    original_zed = Path("/home/isaac/isaac-sim/assets/Assets/Isaac/5.1/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc")
    zed = args.zed or (local_zed if local_zed.is_file() else original_zed)
    files = collect_files(experiments, zed)
    # Fail before writing an archive if the exact frozen inputs are incomplete.
    from verify import verify_files
    verify_files(PROJECT, zed=zed)
    manifest = {"schema": 1, "experiments": experiments,
                "files": {name: sha256(path) for name, path in sorted(files.items())}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=3) as archive:
            for name, path in sorted(files.items()):
                archive.add(path, arcname="spot-cloud-training/" + name, recursive=False)
            data = (json.dumps(manifest, indent=2) + "\n").encode()
            info = tarfile.TarInfo("spot-cloud-training/spot_rl_energy_navigation/cloud/bundle-manifest.json")
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Ready: {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)")
    print("Upload this archive to the GPU, extract it, then run cloud/setup.sh (see README).")


if __name__ == "__main__":
    main()

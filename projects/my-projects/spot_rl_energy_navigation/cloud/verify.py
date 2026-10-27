#!/usr/bin/env python3
"""Check transferred inputs before any large download, then optionally CUDA/DINO/PCA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
EXPERIMENT = "train_general_big_map_test3"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(project, zed=None):
    nav = project / "spot_rl_energy_navigation"
    experiment = nav / EXPERIMENT
    dino = nav / "extracted_features_comparison_v2/models/dinov2"
    zed = zed or Path(os.environ.get("SPOT_ZED_USD", str(project / ".cloud_assets/ZED_X.usdc")))
    required = [project / "IsaacRobotics/assets/spot.usd",
                project / "IsaacRobotics/policies/spot/models/spot_policy.pt", zed,
                experiment / "train.py", experiment / "artifacts/pca32.npz",
                experiment / "artifacts/manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing files. Create/upload the portable package first:\n" + "\n".join(missing))
    identity = json.loads((experiment / "artifacts/manifest.json").read_text())["identity"]
    checks = {dino / name: digest for name, digest in identity["dino"].items()}
    checks.update({experiment / "artifacts" / name: digest for name, digest in identity["missions"].items()})
    checks.update({experiment / "maps" / (name + ".usd"): digest for name, digest in identity["terrain"].items()})
    for path, digest in checks.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing or changed frozen input: {path}")
    print("[OK] Robot, policy, camera, frozen map, missions and exact DINO weights.", flush=True)


def verify_bundle(project):
    manifest_path = project / "spot_rl_energy_navigation/cloud/bundle-manifest.json"
    if not manifest_path.exists():
        return  # A full repo checkout may be used instead of the portable archive.
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        path = project / name
        if not path.resolve().is_relative_to(project.resolve()):
            raise ValueError(f"Invalid manifest path: {name}")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Package file missing/changed: {name}. Recreate the package on the source machine.")
    print(f"[OK] Package checksums: {len(manifest['files'])} files.", flush=True)


def verify_runtime():
    import importlib.metadata as metadata
    import torch
    import cv2
    from transformers import AutoImageProcessor, AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA. Check the rented instance/host driver.")
    name = torch.cuda.get_device_name(0)
    if not any(token in name.upper() for token in ("RTX", "L40", "A40")):
        raise RuntimeError(f"GPU {name} is outside the supported RTX/L40/A40 profile of this setup.")
    # Actually execute a kernel; CUDA enumeration alone is insufficient.
    value = torch.ones((64, 64), device="cuda")
    assert (value @ value).sum().item() == 64**3
    print(f"[OK] CUDA kernel on {name}; {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB VRAM.")
    sys.path.insert(0, str(HERE.parent))
    from train_general_big_map_test3.config import LOCAL_DINO, DEFAULT_PCA
    from train_general_big_map_test3.artifacts import source_identity, pca_compatible
    from train_general_big_map_test3.perception import FrozenPCA
    identity = source_identity(LOCAL_DINO)
    pca = FrozenPCA.load(DEFAULT_PCA)
    if not pca_compatible(pca.metadata.get("identity", {}), identity):
        raise ValueError("Frozen PCA and DINO weights/preprocessing do not match.")
    processor = AutoImageProcessor.from_pretrained(str(LOCAL_DINO), local_files_only=True)
    model = AutoModel.from_pretrained(str(LOCAL_DINO), local_files_only=True).to("cuda").eval()
    import numpy as np
    inputs = processor(images=np.zeros((518, 518, 3), dtype=np.uint8), return_tensors="pt",
                       do_resize=False, do_center_crop=False).to("cuda")
    with torch.no_grad():
        output = model(**inputs).last_hidden_state
    if tuple(output.shape) != (1, 1370, 384):
        raise ValueError(f"Unexpected DINO output shape: {tuple(output.shape)}")
    print(f"[OK] Frozen PCA and DINO CUDA inference; OpenCV {cv2.__version__}.")
    versions = {name: metadata.version(name) for name in
                ("isaacsim", "isaaclab", "torch", "torchvision", "numpy", "transformers")}
    print(json.dumps(versions, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--check-bundle", action="store_true")
    args = parser.parse_args()
    verify_files(PROJECT)
    if args.check_bundle:
        verify_bundle(PROJECT)
    if args.runtime:
        verify_runtime()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"[ERROR] {error}")

#!/usr/bin/env python3
"""Check transferred inputs before any large download, then optionally CUDA/DINO/PCA."""
import argparse
import hashlib
import importlib
import json
import os
import re
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


def selected_experiment(project, explicit=None):
    name = explicit
    manifest = project / 'spot_rl_energy_navigation/cloud/bundle-manifest.json'
    if name is None and manifest.is_file():
        bundle = json.loads(manifest.read_text())
        name = bundle.get('primary_experiment') or (bundle.get('experiments') or [EXPERIMENT])[0]
    name = name or EXPERIMENT
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
        raise ValueError(f'Invalid experiment name: {name}')
    return name


def verify_files(project, zed=None, experiment=None):
    nav = project / "spot_rl_energy_navigation"
    experiment = nav / selected_experiment(project, experiment)
    dino = nav / "extracted_features_comparison_v2/models/dinov2"
    zed = zed or Path(os.environ.get("SPOT_ZED_USD", str(project / ".cloud_assets/ZED_X.usdc")))
    required = [project / "IsaacRobotics/assets/spot.usd",
                project / "IsaacRobotics/policies/spot/models/spot_policy.pt", zed,
                experiment / "train.py",
                experiment / "artifacts/manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing files. Create/upload the portable package first:\n" + "\n".join(missing))
    manifest = json.loads((experiment / "artifacts/manifest.json").read_text())
    identity = manifest['identity']
    pca = experiment / manifest.get('pca_source', 'artifacts/pca32.npz')
    pending=(manifest.get('pca_status')=='pending_new_lab_calibration'
             and manifest.get('pca_lifecycle')=='calibrate_on_first_training')
    if not pca.resolve().is_relative_to(experiment.resolve()) or (not pca.is_file() and not pending):
        raise FileNotFoundError(f'Missing/invalid frozen PCA: {pca}')
    checks = {dino / name: digest for name, digest in identity["dino"].items()}
    checks.update({experiment / "artifacts" / name: digest for name, digest in identity["missions"].items()})
    for name,digest in identity['terrain'].items():
        path=experiment/manifest.get('terrain_files',{}).get(name,'maps/'+name+'.usd')
        if not path.resolve().is_relative_to(experiment.resolve()):
            raise ValueError('Invalid terrain path: '+str(path))
        checks[path]=digest
    for name,digest in manifest.get('runtime_input_files',{}).items():
        path=experiment/name
        if not path.resolve().is_relative_to(experiment.resolve()):
            raise ValueError('Invalid runtime input path: '+str(path))
        checks[path]=digest
    if manifest.get('pca_sha256'):
        checks[pca] = manifest['pca_sha256']
    for name, digest in manifest.get('portable_texture_sha256', {}).items():
        path = project/name
        if not path.resolve().is_relative_to(project.resolve()):
            raise ValueError(f'Invalid texture path: {name}')
        checks[path] = digest
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


def verify_runtime(experiment=None):
    import ctypes
    import importlib.metadata as metadata
    # Report these prerequisites before a missing renderer library causes a
    # native crash or Kit discovers a missing core HDF5 dependency at startup.
    ctypes.CDLL("libGLU.so.1")
    import h5py
    import hydra
    import torch
    import cv2
    from transformers import AutoImageProcessor, AutoModel

    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected the cu128 PyTorch build, got torch={torch.__version__}, CUDA={torch.version.cuda}. Rerun the updated setup.")
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
    experiment = selected_experiment(PROJECT, experiment)
    config = importlib.import_module(experiment+'.config')
    artifacts = importlib.import_module(experiment+'.artifacts')
    perception = importlib.import_module(experiment+'.perception')
    identity = artifacts.source_identity(config.LOCAL_DINO)
    manifest=json.loads((HERE.parent/experiment/'artifacts/manifest.json').read_text())
    pending=(manifest.get('pca_status')=='pending_new_lab_calibration'
             and manifest.get('pca_lifecycle')=='calibrate_on_first_training')
    if not config.DEFAULT_PCA.exists() and pending:
        print('[OK] PCA pending: will be fitted and validated before the first training run.')
    else:
        pca = perception.FrozenPCA.load(config.DEFAULT_PCA)
        if not artifacts.pca_compatible(pca.metadata.get("identity", {}), identity):
            raise ValueError("Frozen PCA and DINO weights/preprocessing do not match.")
        calibration_name = experiment+'.calibration'
        if importlib.util.find_spec(calibration_name) is not None:
            importlib.import_module(calibration_name).validate_calibration(pca, identity)
    processor = AutoImageProcessor.from_pretrained(str(config.LOCAL_DINO), local_files_only=True)
    model = AutoModel.from_pretrained(str(config.LOCAL_DINO), local_files_only=True).to("cuda").eval()
    import numpy as np
    inputs = processor(images=np.zeros((518, 518, 3), dtype=np.uint8), return_tensors="pt",
                       do_resize=False, do_center_crop=False).to("cuda")
    with torch.no_grad():
        output = model(**inputs).last_hidden_state
    if tuple(output.shape) != (1, 1370, 384):
        raise ValueError(f"Unexpected DINO output shape: {tuple(output.shape)}")
    print(f"[OK] DINO CUDA inference and PCA lifecycle checked; OpenCV {cv2.__version__}.")
    versions = {name: metadata.version(name) for name in
                ("isaacsim", "isaaclab", "torch", "torchvision", "numpy", "transformers")}
    print(json.dumps(versions, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--check-bundle", action="store_true")
    parser.add_argument('--experiment', help='default: primary experiment in the package manifest')
    parser.add_argument('--print-experiment', action='store_true', help='print only the selected experiment name')
    args = parser.parse_args()
    experiment = selected_experiment(PROJECT, args.experiment)
    if args.print_experiment:
        print(experiment)
        return
    verify_files(PROJECT, experiment=experiment)
    if args.check_bundle:
        verify_bundle(PROJECT)
    if args.runtime:
        verify_runtime(experiment)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        sys.exit(f"[ERROR] {error}")

"""Inspect recorded data, labels and local assets. Never starts Isaac or DINO."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import cv2
import numpy as np
from common import LABELS, local_path, read_config, verify_local_assets
from projection import project_patches


def load_frames(dataset: Path) -> list[dict]:
    records = [json.loads(line) for line in (dataset / "frames.jsonl").read_text().splitlines() if line.strip()]
    if not records or len({r["frame_id"] for r in records}) != len(records):
        raise ValueError("Empty dataset or duplicate frame IDs")
    return records


def frame_path(dataset: Path, name: str) -> Path:
    path = (dataset / name).resolve()
    if not path.is_relative_to(dataset.resolve()):
        raise ValueError("Frame path escapes dataset")
    return path


def inspect_frame(dataset: Path, record: dict, config: dict):
    if record["label"] not in LABELS:
        raise ValueError(f"Unknown label: {record['label']}")
    rgb = cv2.imread(str(frame_path(dataset, record["rgb"])))
    depth = np.load(frame_path(dataset, record["depth"]), allow_pickle=False)
    expected = (config["camera"]["height"], config["camera"]["width"])
    if rgb is None or rgb.shape != (*expected, 3) or depth.shape != expected:
        raise ValueError(f"Wrong image/depth shape: {record['frame_id']}")
    k = np.asarray(record["intrinsics"])
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    points, center, pure = project_patches(depth, record, config["projection"], config["dino"]["input_size"]//14)
    if not center.any() or not pure.any():
        raise ValueError(f"No valid terrain patches: {record['frame_id']}; inspect its RGB/depth")
    return rgb, depth, points, center, pure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="local raw dataset directory")
    args = parser.parse_args()
    verify_local_assets()
    print("Bundled assets and model hashes: OK")
    if not args.dataset:
        return
    dataset = local_path(args.dataset)
    if not (dataset / "complete.json").is_file():
        raise ValueError("Acquisition is incomplete; complete.json absent")
    config = read_config(dataset / "config.json")
    frames = load_frames(dataset)
    counts, patches = Counter(), Counter()
    for frame in frames:
        *_, pure = inspect_frame(dataset, frame, config)
        counts[frame["label"]] += 1
        patches[frame["label"]] += int(pure.sum())
    if set(counts) != set(LABELS):
        raise ValueError("All three terrains are required")
    print("Frames:", dict(counts), "Pure patches:", dict(patches))


if __name__ == "__main__":
    main()

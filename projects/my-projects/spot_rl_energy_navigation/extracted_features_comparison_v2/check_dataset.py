"""Validate recorded Spot RGB-D and independent semantic ground truth; no Isaac."""
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


def frame_path(dataset, name):
    path = (dataset/name).resolve()
    if not path.is_relative_to(dataset.resolve()):
        raise ValueError("File path escapes dataset")
    return path


def load_frames(dataset):
    records = [json.loads(x) for x in (dataset/"frames.jsonl").read_text().splitlines() if x.strip()]
    if not records or len({r["frame_id"] for r in records}) != len(records):
        raise ValueError("Empty dataset or duplicate IDs")
    return records


def inspect_frame(dataset, record, config, manifest):
    rgb = cv2.imread(str(frame_path(dataset, record["rgb"])))
    depth = np.load(frame_path(dataset, record["depth"]), allow_pickle=False)
    labels = np.load(frame_path(dataset, record["semantic_labels"]), allow_pickle=False)
    shape = (config["camera"]["height"], config["camera"]["width"])
    if rgb is None or rgb.shape != (*shape, 3) or depth.shape != shape or labels.shape != shape:
        raise ValueError(f"Invalid frame sizes: {record['frame_id']}")
    if not np.isin(labels, [-1, *range(len(LABELS))]).all():
        raise ValueError("Invalid semantic labels")
    k = np.asarray(record["intrinsics"])
    root = np.asarray(record["robot_root_state_w"])
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or root.shape != (13,) or not np.isfinite(root).all():
        raise ValueError("Invalid camera/robot metadata")
    projected = project_patches(depth, labels, record, config, manifest)
    return rgb, depth, labels, projected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--previews", action="store_true", help="save one RGB/semantic overlay per target terrain")
    args = parser.parse_args()
    manifest = verify_local_assets()
    print("Local robot, controller, mixed board, textures and DINO hashes: OK")
    if not args.dataset:
        return
    dataset = local_path(args.dataset)
    if not (dataset/"complete.json").is_file():
        raise ValueError("Acquisition incomplete")
    config = read_config(dataset/"config.json")
    captured = json.loads((dataset/"assets_manifest.json").read_text())
    if captured["files"] != manifest["files"]:
        raise ValueError("Dataset assets differ from current local payload")
    counts, groups = Counter(), {k: set() for k in LABELS}
    previewed = set()
    palette = np.array([[80, 80, 80], [240, 210, 80], [80, 130, 220], [40, 170, 220],
                        [70, 170, 70], [180, 70, 180], [80, 80, 230]], dtype=np.uint8)
    for record in load_frames(dataset):
        rgb, _, semantic, projected = inspect_frame(dataset, record, config, captured)
        pure = projected[2].copy()
        instances = projected[3]
        allowed = [i for i, station in enumerate(captured["stations"]) if station["variant"] == record["variant"]]
        pure[~np.isin(instances, allowed)] = -1
        for i, label in enumerate(LABELS):
            counts[label] += int((pure == i).sum())
            if (pure == i).any():
                groups[label].add(record["episode_id"])
        if args.previews and record["target_label"] not in previewed:
            previewed.add(record["target_label"])
            preview_dir = dataset/"previews"
            preview_dir.mkdir(exist_ok=True)
            overlay = rgb.copy()
            mask = semantic >= 0
            overlay[mask] = (.55*rgb[mask] + .45*palette[semantic[mask]]).astype(np.uint8)
            legend = np.zeros((125, rgb.shape[1], 3), dtype=np.uint8)
            for i, label in enumerate(LABELS):
                cv2.rectangle(legend, (8, 5+i*17), (23, 18+i*17), tuple(int(v) for v in palette[i]), -1)
                cv2.putText(legend, label, (30, 17+i*17), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
            image = np.vstack([overlay, legend])
            if not cv2.imwrite(str(preview_dir/f"{record['target_label']}_{record['frame_id']}.png"), image):
                raise IOError("Could not save preview")
    print("Pure patches:", dict(counts))
    print("Episodes observing each class:", {k: len(v) for k, v in groups.items()})
    missing = [k for k in LABELS if counts[k] == 0]
    if missing:
        raise ValueError(f"No valid data for {missing}: inspect images/failed episodes; do not interpret a seven-class comparison")


if __name__ == "__main__":
    main()

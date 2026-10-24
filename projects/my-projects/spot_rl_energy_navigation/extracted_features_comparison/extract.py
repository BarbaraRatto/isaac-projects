"""Offline DINO extraction from a COMPLETE saved acquisition; no Isaac imports."""
from __future__ import annotations

import argparse
import json
import sys

sys.dont_write_bytecode = True
import cv2
import numpy as np
from check_dataset import inspect_frame, load_frames
from common import local_path, read_config, verify_local_assets, write_json
from projection import aggregate_cells, front_cell_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", help="new local directory; default data/features/<dataset name>")
    parser.add_argument("--device", default="cpu", help="cpu avoids using the training GPU")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    dataset = local_path(args.dataset)
    if not (dataset / "complete.json").is_file():
        parser.error("Acquisition is incomplete")
    output = local_path(args.output or f"data/features/{dataset.name}")
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    config = read_config(dataset / "config.json")
    frames = load_frames(dataset)
    manifest = verify_local_assets()
    # Validate all records before loading the model or writing output.
    for record in frames:
        inspect_frame(dataset, record, config)
    import torch
    from dino_features import DinoFeatures
    torch.set_num_threads(args.torch_threads)
    model = DinoFeatures(config["dino"], args.device)
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "provenance.json", {"raw_dataset": str(dataset), "bundled_assets": manifest,
                                          "device": args.device, "torch_version": torch.__version__})
    with (output / "frames.jsonl").open("w") as records:
        for record in frames:
            bgr, _, points, center, pure = inspect_frame(dataset, record, config)
            features = model.extract(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if features.shape != (len(points), 384) or not np.isfinite(features).all():
                raise ValueError("Non-finite or unexpected DINO features")
            # RL grid uses centre projection. Strict patch dataset uses the purity mask.
            grid, counts = aggregate_cells(features, points, center, config["projection"])
            cell_ids = np.flatnonzero(counts.ravel())
            front_ids = front_cell_ids(record, counts, config["projection"])
            filename = f"{record['frame_id']}.npz"
            np.savez_compressed(output / filename,
                patches=features[pure], patch_ids=np.flatnonzero(pure), patch_xyz=points[pure].astype(np.float32),
                cells=grid.reshape(-1, 384)[cell_ids], cell_ids=cell_ids,
                cell_counts=counts.ravel()[cell_ids],
                front_cells=grid.reshape(-1, 384)[front_ids], front_cell_ids=front_ids)
            row = {**record, "features": filename, "pure_patches": int(pure.sum()),
                   "valid_cells": len(cell_ids), "valid_unique_front_cells": len(front_ids)}
            records.write(json.dumps(row) + "\n")
            records.flush()
            print(f"[FEATURES] {record['frame_id']} patches={pure.sum()} cells={len(cell_ids)} front={len(front_ids)}", flush=True)
    write_json(output / "complete.json", {"frames": len(frames), "feature_dim": 384})
    print(f"Features: {output}")


if __name__ == "__main__":
    main()

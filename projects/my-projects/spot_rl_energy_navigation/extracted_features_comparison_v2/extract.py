"""Offline raw DINO384 and mixed-scene cell features; no simulator startup."""
import argparse
import json
import sys
sys.dont_write_bytecode = True
import cv2
import numpy as np
from common import LABELS, local_path, read_config, verify_local_assets, write_json
from check_dataset import load_frames, inspect_frame
from projection import aggregate_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    dataset = local_path(args.dataset)
    if not (dataset/"complete.json").is_file():
        parser.error("Acquisition incomplete")
    output = local_path(args.output or f"data/features/{dataset.name}")
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    config = read_config(dataset/"config.json")
    manifest = verify_local_assets()
    captured = json.loads((dataset/"assets_manifest.json").read_text())
    if captured["files"] != manifest["files"]:
        raise ValueError("Recorded/current asset hashes differ")
    frames = load_frames(dataset)
    for frame in frames:
        inspect_frame(dataset, frame, config, manifest)
    import torch
    from dino_features import DinoFeatures
    torch.set_num_threads(args.torch_threads)
    model = DinoFeatures(config["dino"], args.device)
    output.mkdir(parents=True)
    write_json(output/"config.json", config)
    write_json(output/"provenance.json", {"raw_dataset": str(dataset), "manifest": manifest})
    counts, coverage = 0, {name: 0 for name in LABELS}
    with (output/"frames.jsonl").open("w") as stream:
        for record in frames:
            rgb, _, _, (points, labels, pure, instances, valid) = inspect_frame(dataset, record, config, manifest)
            features = model.extract(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            if features.shape != (len(points), 384) or not np.isfinite(features).all():
                raise ValueError("Invalid DINO vectors")
            cells = aggregate_cells(features, points, labels, instances, valid, record, config["projection"])
            # Keep only the same donor partition as the target station. A frame
            # never contributes training and test instances to the linear probe.
            for instance_id, station in enumerate(manifest["stations"]):
                if station["variant"] != record["variant"]:
                    continue
                patch_mask = (pure >= 0) & (instances == instance_id)
                cell_mask = cells["instances"] == instance_id
                front_mask = cell_mask & cells["front"]
                rl_mask = cells["rl_instances"] == instance_id
                if not patch_mask.any() and not cell_mask.any():
                    continue
                filename = f"{record['frame_id']}_{station['name']}.npz"
                np.savez_compressed(output/filename,
                    patches=features[patch_mask], cells=cells["vectors"][cell_mask],
                    front_cells=cells["vectors"][front_mask], front_cells_rl_filter=cells["rl_vectors"][rl_mask],
                    patch_ids=np.flatnonzero(patch_mask), cell_ids=cells["ids"][cell_mask],
                    front_cell_ids=cells["ids"][front_mask], rl_cell_ids=cells["rl_ids"][rl_mask])
                row = {"frame_id": f"{record['frame_id']}_{station['name']}", "source_frame_id": record["frame_id"],
                       "label": station["label"], "terrain_instance": station["name"], "features": filename,
                       "pose_group": record["pose_group"], "episode_id": record["episode_id"], "kind": record["kind"],
                       "acquisition_split": "train" if record["variant"] == 0 else "test"}
                stream.write(json.dumps(row)+"\n")
                coverage[station["label"]] += int(patch_mask.sum())
                counts += 1
            stream.flush()
            print(f"[V2-FEATURES] {record['frame_id']} pure patches={(pure >= 0).sum()} labelled cells={len(cells['ids'])}", flush=True)
    write_json(output/"complete.json", {"frames": len(frames), "feature_records": counts,
                                      "feature_dim": 384, "pure_patch_coverage": coverage})
    print(f"Features: {output}")


if __name__ == "__main__":
    main()

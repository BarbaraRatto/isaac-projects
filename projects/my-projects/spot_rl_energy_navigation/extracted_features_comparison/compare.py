"""Offline all-pairs comparison, grouped linear probe, and cosine distributions."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
import sys

sys.dont_write_bytecode = True
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize

from check_dataset import frame_path, load_frames
from common import LABELS, local_path, read_config, write_json


def stats(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "std": float(values.std()),
            "p05": float(np.quantile(values, .05)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95))}


def cosine_samples(frames_by_label: dict, a: str, b: str, count: int, rng):
    """Sample frame pairs uniformly; always require DIFFERENT pose groups.

    This avoids calling adjacent patches or light variants of the same pose
    independent views. Pair samples are descriptive, not independent trials.
    """
    first, second = frames_by_label[a], frames_by_label[b]
    eligible = [(i, j) for i, f in enumerate(first) for j, g in enumerate(second)
                if f["pose_group"] != g["pose_group"]]
    if not eligible:
        raise ValueError(f"Need at least two distinct views for {a}/{b}")
    result = np.empty(count, dtype=np.float32)
    for k in range(count):
        i, j = eligible[rng.integers(len(eligible))]
        x, y = first[i]["values"], second[j]["values"]
        result[k] = np.clip(x[rng.integers(len(x))] @ y[rng.integers(len(y))], -1, 1)
    return result


def linear_probe(x, y, groups, settings, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=settings["test_fraction"], random_state=seed)
    train, test = next(splitter.split(x, y, groups))
    if len(set(y[train])) != len(LABELS) or len(set(y[test])) != len(LABELS):
        raise ValueError("Train/test split lacks a terrain; collect more matched poses")
    model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(x[train], y[train])
    predicted = model.predict(x[test])
    unique = np.unique(groups[test])
    if len(unique) < 2:
        raise ValueError("Need at least two held-out pose groups")
    # Resample complete pose groups, not individual correlated patch vectors.
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(settings["bootstrap_repeats"]):
        ids = np.concatenate([np.flatnonzero(groups[test] == g) for g in rng.choice(unique, len(unique), replace=True)])
        if len(set(y[test][ids])) == len(LABELS):
            boots.append(balanced_accuracy_score(y[test][ids], predicted[ids]))
    interval = np.quantile(boots, [.025, .975]).tolist() if boots else None
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y[test], predicted)),
        "chance_balanced_accuracy": 1/len(LABELS),
        "group_bootstrap_95_interval": interval,
        "confusion_matrix": confusion_matrix(y[test], predicted, labels=list(LABELS)).tolist(),
        "train_pose_groups": np.unique(groups[train]).tolist(),
        "test_pose_groups": unique.tolist(),
        "train_samples": len(train), "test_samples": len(test),
        "samples_by_label_test": {k: int((y[test] == k).sum()) for k in LABELS},
        "normalization": "L2 per vector; no PCA; C=1 fixed, no test-set tuning",
    }


def compare_representation(dataset, frames, representation, config, output):
    rng = np.random.default_rng(config["seed"])
    by_label = {k: [] for k in LABELS}
    arrays, labels, group_list, norms = [], [], [], {k: [] for k in LABELS}
    skipped = []
    for frame in frames:
        with np.load(frame_path(dataset, frame["features"]), allow_pickle=False) as data:
            values = data[representation].astype(np.float32)
        if values.ndim != 2 or values.shape[1] != 384 or not np.isfinite(values).all():
            raise ValueError(f"Invalid features: {frame['frame_id']}")
        lengths = np.linalg.norm(values, axis=1)
        values, lengths = values[lengths > 1e-8], lengths[lengths > 1e-8]
        if not len(values):
            skipped.append(frame["frame_id"])
            continue
        keep = rng.choice(len(values), min(len(values), config["analysis"]["samples_per_frame"]), replace=False)
        values, lengths = values[keep], lengths[keep]
        normalized = normalize(values).astype(np.float32)
        label = frame["label"]
        by_label[label].append({"values": normalized, "pose_group": frame["pose_group"]})
        arrays.append(normalized)
        labels.extend([label]*len(values))
        group_list.extend([frame["pose_group"]]*len(values))
        norms[label].extend(lengths.tolist())
    if any(not by_label[k] for k in LABELS):
        raise ValueError(f"{representation}: some terrains have no valid vectors")
    if len(set(group_list)) < 6:
        raise ValueError("Collect at least six distinct pose groups")
    distributions = {}
    pairs = config["analysis"]["cosine_pairs"]
    matrix = np.empty((len(LABELS), len(LABELS)))
    with (output / f"{representation}_cosine_samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["terrain_a", "terrain_b", "cosine"])
        for i, a in enumerate(LABELS):
            for j in range(i, len(LABELS)):
                b = LABELS[j]
                values = cosine_samples(by_label, a, b, pairs, rng)
                distributions[(a, b)] = values
                matrix[i, j] = matrix[j, i] = values.mean()
                writer.writerows((a, b, float(v)) for v in values)
    with (output / f"{representation}_similarity.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["terrain", *LABELS])
        writer.writerows([name, *matrix[i].tolist()] for i, name in enumerate(LABELS))
    pair_results = []
    for a, b in combinations(LABELS, 2):
        within = np.concatenate([distributions[(a, a)], distributions[(b, b)]])
        between = distributions[(a, b)]
        # ROC AUC: does a higher cosine identify same-terrain pairs? 0.5 = chance.
        auc = roc_auc_score(np.r_[np.ones(len(within)), np.zeros(len(between))], np.r_[within, between])
        pair_results.append({"a": a, "b": b, "within_cosine": stats(within), "between_cosine": stats(between),
                             "within_minus_between": float(within.mean()-between.mean()), "same_vs_different_auc": float(auc)})
    x, y, groups = np.concatenate(arrays), np.asarray(labels), np.asarray(group_list)
    result = {
        "labels": LABELS, "mean_cosine_matrix": matrix.tolist(),
        "diagonal": "same terrain across distinct pose groups; NOT forced to one",
        "pair_results": pair_results,
        "norms_before_normalization": {k: stats(np.asarray(norms[k])) for k in LABELS},
        "frames_by_label": {k: len(v) for k, v in by_label.items()},
        "sample_counts": {k: int((y == k).sum()) for k in LABELS},
        "skipped_empty_frames": skipped,
        "linear_probe": linear_probe(x, y, groups, config["analysis"], config["seed"]),
    }
    write_json(output / f"{representation}_metrics.json", result)
    # Only visualization uses PCA later. RL and numeric comparisons retain 384-D.
    display_ids = rng.choice(len(x), min(2000, len(x)), replace=False)
    np.savez_compressed(output / f"{representation}_plot_samples.npz", features=x[display_ids], labels=y[display_ids])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", help="new local directory; default results/<feature dataset name>")
    args = parser.parse_args()
    dataset = local_path(args.features)
    if not (dataset / "complete.json").is_file():
        parser.error("Feature extraction is incomplete")
    config = read_config(dataset / "config.json")
    frames = load_frames(dataset)
    output = local_path(args.output or f"results/{dataset.name}")
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    results = {}
    for representation in ("patches", "cells", "front_cells"):
        results[representation] = compare_representation(dataset, frames, representation, config, output)
        print(representation, results[representation]["linear_probe"]["balanced_accuracy"])
    write_json(output / "summary.json", {"feature_dataset": str(dataset), "results": results,
        "limitations": ["Views of existing simulated assets, not independent real terrain instances",
                        "Tokens have global image context; nine-point purity is a geometric approximation",
                        "Bootstrap over poses; pair counts are not independent sample counts",
                        "Static camera, no robot occlusion; grouped holdout tests poses, not new textures",
                        "Front cells reproduce numerical sampling relative to a virtual robot; no locomotion"]})
    print(f"Results: {output}")


if __name__ == "__main__":
    main()

"""Decode renderer ground truth without shadowing Kit's semantics namespace.

Pure NumPy; DINO never supplies labels.
"""
import numpy as np
from common import LABELS


def decode_semantic_labels(ids, info):
    if not isinstance(info, dict) or not isinstance(info.get("idToLabels"), dict):
        raise ValueError("Semantic annotator did not return idToLabels")
    labels = np.full(ids.shape, -1, dtype=np.int8)
    for number, metadata in info["idToLabels"].items():
        value = metadata.get("class", "") if isinstance(metadata, dict) else ""
        names = value if isinstance(value, list) else [str(value)]
        matches = [i for i, name in enumerate(LABELS) if name in names]
        if len(matches) == 1:
            labels[ids == int(number)] = matches[0]
    return labels

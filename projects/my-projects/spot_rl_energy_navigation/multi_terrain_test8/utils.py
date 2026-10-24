"""Local metadata, result decoding and replay memory estimates."""

import hashlib
import json
from pathlib import Path

import numpy as np

from multi_terrain_test8.config import OBSERVATION_SIZE, experiment_signature
from multi_terrain_test8.layouts import training_conditions


def number(info: dict, key: str) -> float:
    value = info[key]
    return float(value.item()) if hasattr(value, "item") else float(value)


def reason(info: dict) -> str:
    if number(info, "is_success"):
        return "goal"
    if info.get("TimeLimit.truncated"):
        return "timeout"
    if info.get("out_of_bounds", False):
        return "out_of_bounds"
    if info.get("fallen", False):
        return "fallen"
    return "unsafe"


def terrain_hashes() -> dict:
    return {item.layout.name: hashlib.sha256(item.layout.usd.read_bytes()).hexdigest()
            for item in training_conditions()}


def replay_bytes(buffer_size: int, num_envs: int) -> int:
    # SB3 rounds to full vector steps; observations and next_observations float32.
    transitions = max(buffer_size // num_envs, 1) * num_envs
    return transitions * (2 * OBSERVATION_SIZE * 4 + 2 * 4 + 3 * 4)


def validate_checkpoint_config(path: Path) -> dict:
    saved = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
    if saved.get("signature") != experiment_signature():
        raise RuntimeError("Checkpoint has incompatible test8 policy/environment settings")
    if saved.get("terrain_sha256") != terrain_hashes():
        raise RuntimeError("The terrain USDs differ from the training assets")
    return saved


def stop_actions(num_envs: int) -> np.ndarray:
    actions = np.zeros((num_envs, 2), dtype=np.float32)
    actions[:, 0] = -1.0
    return actions

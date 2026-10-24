"""Reproducible artifact identities, checkpointing and append-only CSV logs (no simulator)."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from multi_terrain_test10.config import LOCAL_DINO, DINO_INPUT_SIZE, experiment_signature
from multi_terrain_test10.perception import file_sha256


def source_identity(model_path=None):
    from multi_terrain_test10.layouts import TRAIN_LAYOUTS
    directory = Path(model_path or LOCAL_DINO).resolve()
    # Require concrete local weights to bind PCA to the exact feature extractor.
    names = ('config.json', 'preprocessor_config.json', 'model.safetensors')
    missing = [str(directory/name) for name in names if not (directory/name).is_file()]
    if missing:
        raise FileNotFoundError('Local DINO weights required (use --dino-model): '+', '.join(missing))
    return dict(dino={name: file_sha256(directory/name) for name in names},
                input_size=DINO_INPUT_SIZE, resize='opencv_linear_square_no_crop',
                terrain={item.name: file_sha256(item.usd) for item in TRAIN_LAYOUTS})


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def save_checkpoint(path, learner, pca, identity, best_score=None):
    path = Path(path)
    state = dict(schema=1, signature=experiment_signature(), identity=identity,
                 learner=learner.state_dict(), rng=rng_state(), best_score=best_score,
                 pca=dict(mean=pca.mean, components=pca.components,
                          explained=pca.explained, metadata=pca.metadata))
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def snapshot_sources(directory):
    """Keep the complete local implementation beside each training run."""
    source = Path(__file__).resolve().parent
    target = Path(directory)/'sources'
    target.mkdir()
    for pattern in ('*.py', '*.sh', '*.md'):
        for path in source.glob(pattern):
            (target/path.name).write_bytes(path.read_bytes())


class CSVLog:
    def __init__(self, path, fields):
        self.stream = Path(path).open('w', newline='', buffering=1)
        self.writer = csv.DictWriter(self.stream, fieldnames=fields, extrasaction='raise')
        self.writer.writeheader()

    def write(self, record):
        self.writer.writerow(record)

    def close(self):
        self.stream.close()


EPISODE_FIELDS = ['transitions', 'env_id', 'layout', 'reason', 'success', 'steps', 'energy_j',
                  'path_m', 'final_distance_m', 'nav_return', 'energy_cost']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

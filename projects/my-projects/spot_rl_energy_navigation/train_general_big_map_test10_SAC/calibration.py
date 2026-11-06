"""Simulator-independent sampling and checks for seven-material PCA calibration."""
import hashlib
import json
from pathlib import Path
import numpy as np

from train_general_big_map_test10_SAC.config import PCA_DATASET, TERRAIN_NAMES


def point_labels(xy, metadata, margin=.12):
    """Material at a depth hit; reject boundary patches to reduce mixed labels."""
    grid = np.array([[TERRAIN_NAMES.index(x) for x in row] for row in metadata['grid']])
    size = np.array([metadata['cell_width_m'], metadata['cell_height_m']])
    offset = np.asarray(xy)-np.asarray(metadata['bounds_min'][:2])
    colrow = np.floor(offset/size).astype(int)
    within = offset-colrow*size
    valid = ((colrow >= 0).all(1) & (colrow[:, 0] < grid.shape[1]) &
             (colrow[:, 1] < grid.shape[0]) & (within >= margin).all(1) &
             (within <= size-margin).all(1))
    labels = np.full(len(xy), -1, dtype=int)
    labels[valid] = grid[colrow[valid, 1], colrow[valid, 0]]
    return labels


def view_plan(points, evaluation, metadata, seed=123, train_views=96, validation_views=24):
    """Disjoint camera base positions from TRAIN mission candidates only.

    Ice is viewed from neighbouring non-ice ground. Calibration does not walk or
    change endpoint rules. Hold-out views are diagnostic, not unseen-map tests.
    """
    rng = np.random.default_rng(seed)
    pools = {'train': [], 'validation': []}
    # Split each cell's training candidates, keeping the PCA hold-out separate.
    for row, col in np.unique(points[:, 5:7], axis=0):
        ids = np.flatnonzero(~evaluation & (points[:, 5] == row) & (points[:, 6] == col))
        ids = rng.permutation(ids)
        n = max(1, len(ids)//5)
        pools['validation'].extend(ids[:n]); pools['train'].extend(ids[n:])
    size = np.array([metadata['cell_width_m'], metadata['cell_height_m']])
    origin = np.asarray(metadata['bounds_min'][:2])
    excluded_cells = {name: [(r, c) for r, row in enumerate(metadata['grid'])
                           for c, value in enumerate(row) if value == name]
                      for name in ('t2_slippery', 't7_stairs')}
    result = []
    for split, count in (('train', train_views), ('validation', validation_views)):
        pool = np.array(pools[split], dtype=int)
        for code in range(len(TERRAIN_NAMES)):
            if TERRAIN_NAMES[code] in excluded_cells:
                # Aim at a patch just inside an ice cell, within the local map.
                choices = []
                for r, c in excluded_cells[TERRAIN_NAMES[code]]:
                    lo = origin+np.array([c, r])*size+.3
                    hi = origin+np.array([c+1, r+1])*size-.3
                    targets = np.clip(points[pool, :2], lo, hi)
                    distances = np.linalg.norm(targets-points[pool, :2], axis=1)
                    for j in np.flatnonzero((distances >= .6) & (distances <= 1.9)):
                        choices.append((int(pool[j]), targets[j]))
                if not choices:
                    raise ValueError(f'No safe nearby ice viewpoints for {split}')
                for j in rng.choice(len(choices), count, replace=len(choices) < count):
                    index, target = choices[j]
                    delta = target-points[index, :2]
                    yaw = np.arctan2(delta[1], delta[0])+rng.uniform(-.25, .25)
                    result.append(dict(split=split, target_material=code, point_id=index, yaw=float(yaw)))
            else:
                ids = pool[points[pool, 4] == code]
                if not len(ids):
                    raise ValueError(f'No viewpoints for material {code}, {split}')
                for index in rng.choice(ids, count, replace=len(ids) < count):
                    result.append(dict(split=split, target_material=code, point_id=int(index),
                                       yaw=float(rng.uniform(-np.pi, np.pi))))
    rng.shuffle(result)
    return result


class Reservoir:
    def __init__(self, capacity, seed):
        self.rng = np.random.default_rng(seed)
        self.capacity = capacity
        self.samples = [[] for _ in TERRAIN_NAMES]
        self.seen = np.zeros(len(TERRAIN_NAMES), dtype=int)
        self.frames = np.zeros(len(TERRAIN_NAMES), dtype=int)

    def add(self, features, labels):
        for code in np.unique(labels[labels >= 0]):
            values = features[labels == code]
            self.frames[code] += 1
            bucket = self.samples[code]
            for value in values:
                if not np.isfinite(value).all():
                    raise ValueError('Non-finite raw DINO features')
                self.seen[code] += 1
                if len(bucket) < self.capacity:
                    bucket.append(value.copy())
                else:
                    index = int(self.rng.integers(self.seen[code]))
                    if index < self.capacity:
                        bucket[index] = value.copy()

    def balanced(self, minimum, minimum_frames=8):
        counts = [len(x) for x in self.samples]
        if min(counts) < minimum or int(self.frames.min()) < minimum_frames:
            raise ValueError(f'Insufficient PCA coverage: samples={counts}, views={self.frames.tolist()}. '
                             'Increase --views-per-terrain / --validation-views; training will not start.')
        n = min(counts)
        arrays = [np.asarray(x)[self.rng.choice(len(x), n, replace=False)] for x in self.samples]
        return np.concatenate(arrays), np.repeat(np.arange(len(TERRAIN_NAMES)), n)


def pca_digest(pca):
    h = hashlib.sha256()
    for array in (pca.mean, pca.components, pca.explained):
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def validate_calibration(pca, identity):
    m = pca.metadata
    direct = m.get('identity') == identity
    transferred = False
    manifest = Path(__file__).parent/'artifacts/manifest.json'
    if not direct and manifest.is_file():
        transfer = json.loads(manifest.read_text()).get('pca_transfer', {})
        transferred = (transfer.get('target_identity') == identity and
                       transfer.get('calibration_identity') == m.get('identity') and
                       transfer.get('array_sha256') == pca_digest(pca) and
                       all(m.get('identity', {}).get(k) == identity.get(k)
                           for k in ('dino', 'input_size', 'resize')))
    if m.get('dataset') != PCA_DATASET or not (direct or transferred):
        raise ValueError('PCA is not calibrated on this map/DINO/protocol or approved by the frozen transfer manifest')
    if m.get('materials') != list(TERRAIN_NAMES) or m.get('fitted_per_terrain', 0) < 1000:
        raise ValueError('PCA calibration does not cover all seven materials')
    if min(m.get('validation_counts', [0])) < 250 or len(m.get('validation_counts', [])) != 7:
        raise ValueError('PCA has no complete independent-view validation')


def validation_metrics(pca, features, labels):
    def measure(x):
        centered = x.astype(np.float64)-x.mean(0, dtype=np.float64)
        denominator = np.square(centered).sum()
        if denominator <= 0:
            raise ValueError('Zero variance in PCA validation views')
        return dict(patches=len(x), variance_retained=float(np.square(centered@pca.components.T).sum()/denominator))
    return dict(overall=measure(features),
                by_material={name: measure(features[labels == code]) for code, name in enumerate(TERRAIN_NAMES)})

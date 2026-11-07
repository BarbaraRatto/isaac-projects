"""Simulator-independent sampling and checks for two-material PCA calibration."""
import numpy as np

from train_general_big_map_test11_SAC_Re.config import PCA_DATASET, TERRAIN_NAMES


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

    Both colors are sampled directly. Calibration does not walk or change
    endpoint rules. Hold-out views are diagnostic, not unseen-map tests.
    """
    rng = np.random.default_rng(seed)
    pools = {'train': [], 'validation': []}
    # Split each cell's training candidates, keeping the PCA hold-out separate.
    for row, col in np.unique(points[:, 5:7], axis=0):
        ids = np.flatnonzero(~evaluation & (points[:, 5] == row) & (points[:, 6] == col))
        ids = rng.permutation(ids)
        n = max(1, len(ids)//5)
        pools['validation'].extend(ids[:n]); pools['train'].extend(ids[n:])
    result = []
    for split,count in (('train',train_views),('validation',validation_views)):
        pool = np.array(pools[split], dtype=int)
        for code in range(len(TERRAIN_NAMES)):
            ids = pool[points[pool,4] == code]
            if not len(ids):
                raise ValueError(f'No viewpoints for material {code}, {split}')
            for index in rng.choice(ids,count,replace=len(ids)<count):
                result.append(dict(split=split,target_material=code,point_id=int(index),
                                   yaw=float(rng.uniform(-np.pi,np.pi))))
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


def validate_calibration(pca, identity):
    m = pca.metadata
    if m.get('dataset') != PCA_DATASET or m.get('identity') != identity:
        raise ValueError('PCA is not calibrated on this map/DINO/protocol; use prepare_pca.py with a new --output')
    if m.get('materials') != list(TERRAIN_NAMES) or m.get('fitted_per_terrain', 0) < 1000:
        raise ValueError('PCA calibration does not cover both lab materials')
    if min(m.get('validation_counts', [0])) < 250 or len(m.get('validation_counts', [])) != len(TERRAIN_NAMES):
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

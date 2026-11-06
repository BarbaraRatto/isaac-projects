"""Persistent RGB-D statistics in a fixed episode frame; no terrain/label access.

Each observed 10 cm XY bin stores the latest batch of visual samples and/or
dense geometric moments, independently. Unobserved bins are never filled.
Query groups bin centroids into the current yaw-aligned 50 cm actor cells.
Covariances retain within-bin roughness. No repeated grid rotation/resampling.
"""
import numpy as np

from train_general_big_map_test10_SAC_Re.config import (
    LOCAL_CELLS, CELL_SIZE, MEMORY_RESOLUTION_M, MEMORY_AGE_SECONDS,
    GEOMETRY_MIN_POINTS, GEOMETRY_MIN_XY_EIGENVALUE,
)
from train_general_big_map_test10_SAC_Re.perception import cell_indices, world_to_local


class _BinStore:
    """Keep insertion-ordered bins in reusable arrays instead of stacking records."""
    def __init__(self, shape):
        self.indices = {}
        self.shape = shape
        self.capacity = 0
        self.centers = np.empty((0, 3), dtype=np.float64)
        self.data = np.empty((0, *shape), dtype=np.float64)
        self.weights = np.empty(0, dtype=np.float64)
        self.times = np.empty(0, dtype=np.float64)

    def __len__(self):
        return len(self.indices)

    def values(self):
        for i in range(len(self)):
            yield self.centers[i], self.data[i], self.weights[i], self.times[i]

    def update(self, keys, centers, data, weights, time_s):
        old_size = len(self)
        rows = np.empty(len(keys), dtype=np.intp)
        for i, key in enumerate(keys):
            rows[i] = self.indices.setdefault(tuple(key), len(self.indices))
        if len(self) > self.capacity:
            self.capacity = max(len(self), 2*self.capacity, 256)
            for name, shape in (('centers', (3,)), ('data', self.shape),
                                ('weights', ()), ('times', ())):
                old = getattr(self, name)
                new = np.empty((self.capacity, *shape), dtype=np.float64)
                new[:old_size] = old[:old_size]
                setattr(self, name, new)
        self.centers[rows] = centers
        self.data[rows] = data
        self.weights[rows] = weights
        self.times[rows] = time_s


class EpisodeMap:
    def __init__(self):
        self.anchor = None
        self.visual = _BinStore((32,))
        self.geometry = _BinStore((3, 3))

    def update(self, patch_world, features, dense_world, base_world, time_s):
        if self.anchor is None:
            self.anchor = np.asarray(base_world, dtype=np.float64).copy()
        if features.shape != (len(patch_world), 32):
            raise ValueError('Expected one PCA32 vector per patch point')
        for points, values, cache in ((patch_world, features, self.visual),
                                       (dense_world, None, self.geometry)):
            p = np.asarray(points, dtype=np.float64) - self.anchor
            if not len(p):
                continue
            if not np.isfinite(p).all() or (values is not None and not np.isfinite(values).all()):
                raise ValueError('Non-finite map input')
            keys, inv, count = np.unique(np.floor(p[:, :2]/MEMORY_RESOLUTION_M).astype(np.int64),
                                         axis=0, return_inverse=True, return_counts=True)
            n = len(keys)
            def means(data):
                return np.column_stack([np.bincount(inv, weights=v, minlength=n)/count
                                        for v in data.T])
            centers = means(p)
            if values is not None:
                data = means(values)
            else:
                # Center BEFORE squaring, avoiding cancellation at distant env origins.
                delta = p-centers[inv]
                data = means((delta[:, :, None]*delta[:, None, :]).reshape(-1, 9)).reshape(-1, 3, 3)
            cache.update(keys, centers, data, count, float(time_s))

    def query(self, base_world, yaw, time_s):
        out = np.zeros((LOCAL_CELLS, CELL_SIZE), dtype=np.float32)
        if self.anchor is None:
            return out
        base = np.asarray(base_world, dtype=np.float64)-self.anchor
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
        for visual, cache in ((True, self.visual), (False, self.geometry)):
            if not cache:
                continue
            centers = cache.centers[:len(cache)]
            local = world_to_local(centers, base, yaw)
            ids = cell_indices(local)
            # Old bins remain in memory, but only current actor cells need data.
            valid = ids >= 0
            local, ids = local[valid], ids[valid]
            data = cache.data[:len(cache)][valid]
            weight = cache.weights[:len(cache)][valid]
            ages = np.clip((time_s-cache.times[:len(cache)][valid])/MEMORY_AGE_SECONDS, 0., 1.)
            for cell in np.unique(ids):
                selected = ids == cell
                w = weight[selected]
                total = w.sum()
                age = np.average(ages[selected], weights=w)
                if visual:
                    out[cell, :32] = np.average(data[selected], axis=0, weights=w)
                    out[cell, 36] = 1
                    out[cell, 38] = age
                    continue
                if total < GEOMETRY_MIN_POINTS:
                    continue
                pts = local[selected]
                center = np.average(pts, axis=0, weights=w)
                delta = pts-center
                covs = rotation @ data[selected] @ rotation.T
                covariance = np.average(covs+delta[:, :, None]*delta[:, None, :], axis=0, weights=w)
                xy = covariance[:2, :2]
                if np.linalg.eigvalsh(xy)[0] < GEOMETRY_MIN_XY_EIGENVALUE:
                    continue
                slope = np.linalg.solve(xy, covariance[:2, 2])
                normal = np.array([-slope[0], -slope[1], 1.])
                roughness = np.sqrt(max(0., normal @ covariance @ normal)/(normal @ normal))
                out[cell, 32:36] = (center[2], np.arctan(slope[0]), np.arctan(slope[1]), roughness)
                out[cell, 37] = 1
                out[cell, 39] = age
        if not np.isfinite(out).all():
            raise ValueError('Non-finite persistent map')
        return out


class MapBank:
    """Own one independent map per env; partial resets cannot leak old episodes."""
    def __init__(self):
        self.maps = {}

    def reset(self, env_ids):
        for i in env_ids:
            self.maps.pop(int(i), None)

    def observe(self, env_id, patches, features, dense, base, yaw, time_s):
        memory = self.maps.setdefault(int(env_id), EpisodeMap())
        memory.update(patches, features, dense, base, time_s)
        return memory.query(base, yaw, time_s)

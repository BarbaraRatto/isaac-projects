"""Persistent RGB-D statistics in a fixed episode frame; no terrain/label access.

Each observed 10 cm XY bin stores the latest batch of visual samples and/or
dense geometric moments, independently. Unobserved bins are never filled.
Query groups bin centroids into the current yaw-aligned 50 cm actor cells.
Covariances retain within-bin roughness. No repeated grid rotation/resampling.
"""
import numpy as np

from multi_terrain_test11.config import (
    LOCAL_CELLS, CELL_SIZE, MEMORY_RESOLUTION_M, MEMORY_AGE_SECONDS,
    GEOMETRY_MIN_POINTS, GEOMETRY_MIN_XY_EIGENVALUE,
)
from multi_terrain_test11.perception import cell_indices, world_to_local


class EpisodeMap:
    def __init__(self):
        self.anchor = None
        self.visual = {}
        self.geometry = {}

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
            for key, center, datum, weight in zip(keys, centers, data, count):
                cache[tuple(key)] = (center, datum, float(weight), float(time_s))

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
            records = list(cache.values())
            centers = np.stack([r[0] for r in records])
            data = np.stack([r[1] for r in records])
            weight = np.array([r[2] for r in records])
            ages = np.clip((time_s-np.array([r[3] for r in records]))/MEMORY_AGE_SECONDS, 0., 1.)
            local = world_to_local(centers, base, yaw)
            ids = cell_indices(local)
            for cell in np.unique(ids[ids >= 0]):
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

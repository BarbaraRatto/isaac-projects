"""Simulator-independent PCA, backprojection and gravity-aligned local cells."""
import hashlib
import json
from pathlib import Path

import numpy as np

from multi_terrain_test13.config import (
    FEATURE_DIM, RAW_FEATURE_DIM, CELL_SIZE, LOCAL_CELLS, LOOK_X_M, LOOK_Y_M,
    RESOLUTION_M, GEOMETRY_MIN_POINTS, GEOMETRY_MIN_XY_EIGENVALUE,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class FrozenPCA:
    def __init__(self, mean, components, explained, metadata):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.components = np.asarray(components, dtype=np.float32)
        self.explained = np.asarray(explained, dtype=np.float64)
        self.metadata = metadata
        if self.mean.shape != (RAW_FEATURE_DIM,) or self.components.shape != (FEATURE_DIM, RAW_FEATURE_DIM):
            raise ValueError('PCA must transform 384 features to 32 components')
        if not all(np.isfinite(x).all() for x in (self.mean, self.components, self.explained)):
            raise ValueError('Non-finite PCA artifact')
        if not np.allclose(self.components @ self.components.T, np.eye(FEATURE_DIM), atol=1e-4):
            raise ValueError('PCA components are not orthonormal')

    def transform(self, features):
        return (features - self.mean) @ self.components.T

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream:
            np.savez_compressed(stream, mean=self.mean, components=self.components,
                                explained=self.explained, metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data['mean'], data['components'], data['explained'],
                       json.loads(str(data['metadata'])))

    @classmethod
    def fit(cls, features, metadata):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != RAW_FEATURE_DIM or len(features) < 1000:
            raise ValueError('Collect at least 1000 valid 384-D terrain patches')
        if not np.isfinite(features).all():
            raise ValueError('Non-finite calibration features')
        mean = features.mean(0)
        centered = features - mean
        eigenvalues, vectors = np.linalg.eigh(centered.T @ centered / (len(features) - 1))
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0)
        if eigenvalues.sum() <= 0:
            raise ValueError('Calibration features have zero variance')
        components = vectors[:, order[:FEATURE_DIM]].T
        # Deterministic sign convention, also useful when comparing saved artifacts.
        components *= np.sign(components[np.arange(FEATURE_DIM), np.abs(components).argmax(1)])[:, None]
        return cls(mean, components, eigenvalues[:FEATURE_DIM] / eigenvalues.sum(), metadata)


def quaternion_matrix(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def backproject(depth, intrinsic, camera_pos, camera_quat_ros, stride=4, patches=False):
    """Depth is distance_to_image_plane (optical z), never radial distance."""
    height, width = depth.shape
    if patches:
        u, v = np.meshgrid((np.arange(37)+0.5)*width/37, (np.arange(37)+0.5)*height/37)
    else:
        u, v = np.meshgrid(np.arange(stride//2, width, stride), np.arange(stride//2, height, stride))
    ui = np.clip(np.rint(u).astype(int), 0, width-1)
    vi = np.clip(np.rint(v).astype(int), 0, height-1)
    z = depth[vi, ui].ravel()
    valid = np.isfinite(z) & (z > 0.2) & (z < 15.0)
    k = intrinsic
    # Use the sampled pixel coordinates, keeping RGB patch/depth association explicit.
    optical = np.column_stack(((ui.ravel()[valid]-k[0, 2])*z[valid]/k[0, 0],
                               (vi.ravel()[valid]-k[1, 2])*z[valid]/k[1, 1], z[valid]))
    return optical @ quaternion_matrix(camera_quat_ros).T + camera_pos, np.flatnonzero(valid)


def world_to_local(points, base_pos, yaw):
    delta = np.asarray(points) - np.asarray(base_pos)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack((c*delta[:, 0]+s*delta[:, 1],
                            -s*delta[:, 0]+c*delta[:, 1], delta[:, 2]))


def cell_indices(points):
    ix = np.floor((points[:, 0] - (LOOK_X_M[0]-RESOLUTION_M/2))/RESOLUTION_M).astype(int)
    iy = np.floor((points[:, 1] - (LOOK_Y_M[0]-RESOLUTION_M/2))/RESOLUTION_M).astype(int)
    inside = (ix >= 0) & (ix < len(LOOK_X_M)) & (iy >= 0) & (iy < len(LOOK_Y_M))
    return np.where(inside, ix*len(LOOK_Y_M)+iy, -1)


def local_cells(patch_points, features, dense_points):
    """All points already expressed in the yaw/gravity-aligned base frame."""
    cells = np.zeros((LOCAL_CELLS, CELL_SIZE), dtype=np.float32)
    if features.shape != (len(patch_points), FEATURE_DIM):
        raise ValueError('Projected patch/features mismatch')
    patch_ids, dense_ids = cell_indices(patch_points), cell_indices(dense_points)
    for i in range(LOCAL_CELLS):
        f = features[patch_ids == i]
        if len(f):
            cells[i, :32] = f.mean(0)
            cells[i, 36] = 1
        p = dense_points[dense_ids == i]
        if len(p) < GEOMETRY_MIN_POINTS:
            continue
        centered = p - p.mean(0)
        covariance_xy = centered[:, :2].T @ centered[:, :2] / len(p)
        if np.linalg.eigvalsh(covariance_xy)[0] < GEOMETRY_MIN_XY_EIGENVALUE:
            continue
        slope = np.linalg.solve(covariance_xy, centered[:, :2].T @ centered[:, 2] / len(p))
        residual = (centered[:, 2] - centered[:, :2] @ slope) / np.sqrt(1 + slope @ slope)
        cells[i, 32:36] = (p[:, 2].mean(), np.arctan(slope[0]), np.arctan(slope[1]),
                           np.sqrt(np.mean(residual**2)))
        cells[i, 37] = 1
    if not np.isfinite(cells).all():
        raise ValueError('Non-finite local map')
    return cells

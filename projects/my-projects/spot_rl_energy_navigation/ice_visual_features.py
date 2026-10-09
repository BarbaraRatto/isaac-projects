"""DINO patch projection and shared visual observation for the ice choice task.

The atlas contains no terrain labels or energy costs. Both training modes use
the same 32-component PCA basis and the same body-relative sampling grid.
"""

from pathlib import Path

import numpy as np
import torch
from isaaclab.utils.math import quat_apply

from rl_config import TERRAIN_USD


RESOLUTION_M = 0.5
X0_M, Y0_M = 12.0, -1.5
ROWS, COLS = 36, 30  # X: 12..30 m, Y: -1.5..13.5 m
FEATURE_DIM = 384
PCA_DIM = 32
LOOK_X_M = (1.5, 3.0, 4.5, 6.0)
LOOK_Y_M = (-2.5, -1.25, 0.0, 1.25, 2.5)
LOCAL_CELLS = len(LOOK_X_M) * len(LOOK_Y_M)
VISUAL_SIZE = LOCAL_CELLS * (PCA_DIM + 1)



def camera_ground_points(camera) -> tuple[np.ndarray, np.ndarray]:
    """Project all finite-depth DINO patch centres onto the visible terrain.

    The old short-course helper discarded the top half of the image. That
    would hide ice and ramps several metres ahead in this longer task.
    """
    data = camera.data
    depth = data.output["distance_to_image_plane"][0].cpu().numpy().squeeze()
    height, width = depth.shape
    patch_centres = (np.arange(37) + 0.5) / 37.0
    uu, vv = np.meshgrid(patch_centres * width, patch_centres * height)
    ui = np.clip(np.rint(uu).astype(int), 0, width - 1)
    vi = np.clip(np.rint(vv).astype(int), 0, height - 1)
    distance = depth[vi, ui].reshape(-1)
    intrinsic = data.intrinsic_matrices[0].cpu().numpy()
    fx, fy, cx, cy = (intrinsic[0, 0], intrinsic[1, 1],
                      intrinsic[0, 2], intrinsic[1, 2])
    depth_valid = np.isfinite(distance) & (distance > 0.2) & (distance < 15.0)
    safe_depth = np.where(depth_valid, distance, 0.0)
    optical = np.column_stack(((uu.reshape(-1) - cx) * safe_depth / fx,
                               (vv.reshape(-1) - cy) * safe_depth / fy,
                               safe_depth))
    optical_valid = torch.from_numpy(optical[depth_valid].astype(np.float32)).to(data.pos_w.device)
    world = quat_apply(data.quat_w_ros[0].expand(len(optical_valid), -1), optical_valid)
    world += data.pos_w[0]
    points = world.cpu().numpy()
    terrain = (points[:, 2] > -0.3) & (points[:, 2] < 0.65)
    accepted = np.zeros(len(depth_valid), dtype=bool)
    accepted[np.flatnonzero(depth_valid)[terrain]] = True
    return points[terrain], accepted


def camera_patches(camera, dino) -> tuple[np.ndarray, np.ndarray]:
    rgb = camera.data.output["rgb"][0].cpu().numpy().copy()
    features = dino.extract(rgb).reshape(-1, FEATURE_DIM)
    points, accepted = camera_ground_points(camera)
    return points, features[accepted]


def grid_indices(points: np.ndarray):
    rows = np.floor((points[:, 0] - X0_M) / RESOLUTION_M).astype(int)
    cols = np.floor((points[:, 1] - Y0_M) / RESOLUTION_M).astype(int)
    inside = (rows >= 0) & (rows < ROWS) & (cols >= 0) & (cols < COLS)
    return rows[inside], cols[inside], inside


def mask_from_points(points: np.ndarray) -> np.ndarray:
    mask = np.zeros((ROWS, COLS), dtype=bool)
    rows, cols, _ = grid_indices(points)
    mask[rows, cols] = True
    return mask


class FeatureField:
    def __init__(self):
        self.sums = np.zeros((ROWS, COLS, FEATURE_DIM), dtype=np.float64)
        self.count = np.zeros((ROWS, COLS), dtype=np.int32)

    def insert(self, points: np.ndarray, features: np.ndarray) -> None:
        if len(points) != len(features) or features.shape[1:] != (FEATURE_DIM,):
            raise ValueError("DINO features and projected terrain points do not match")
        rows, cols, inside = grid_indices(points)
        np.add.at(self.sums, (rows, cols), features[inside])
        np.add.at(self.count, (rows, cols), 1)

    def finish(self):
        valid = self.count > 0
        features = np.zeros((ROWS, COLS, FEATURE_DIM), dtype=np.float32)
        features[valid] = (self.sums[valid] / self.count[valid, None]).astype(np.float32)
        return features, valid


def save_atlas(path: Path, field: FeatureField, build_seconds: float) -> int:
    features, valid = field.finish()
    samples = features[valid]
    if len(samples) < PCA_DIM + 1:
        raise RuntimeError(f"Only {len(samples)} occupied cells; need at least {PCA_DIM + 1}")
    mean = samples.mean(axis=0)
    centered = samples - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[:PCA_DIM].T
    projected = centered @ axes
    scale = np.maximum(projected.std(axis=0), 1e-4)
    total_variance = float((singular ** 2).sum())
    variance_fraction = (float((singular[:PCA_DIM] ** 2).sum() / total_variance)
                         if total_variance > 0 else 0.0)
    source = TERRAIN_USD.stat()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, features=features, valid=valid, count=field.count,
        mean=mean, axes=axes, scale=scale,
        resolution_m=RESOLUTION_M, x0_m=X0_M, y0_m=Y0_M,
        terrain_size=source.st_size, terrain_mtime_ns=source.st_mtime_ns,
        pca_variance_fraction=variance_fraction, build_seconds=build_seconds,
    )
    print(f"[ATLAS] occupied={int(valid.sum())}/{ROWS * COLS} "
          f"PCA-{PCA_DIM} variance_fraction={variance_fraction:.3f} "
          f"build_seconds={build_seconds:.1f}", flush=True)
    return int(valid.sum())


class VisualObservation:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.atlas = data["features"].astype(np.float32)
            self.valid = data["valid"].astype(bool)
            self.mean = data["mean"].astype(np.float32)
            self.axes = data["axes"].astype(np.float32)
            self.scale = data["scale"].astype(np.float32)
            if (int(data["terrain_size"]) != TERRAIN_USD.stat().st_size
                    or int(data["terrain_mtime_ns"]) != TERRAIN_USD.stat().st_mtime_ns):
                raise RuntimeError("The terrain USD changed after this feature atlas was made")
            if (float(data["resolution_m"]) != RESOLUTION_M
                    or float(data["x0_m"]) != X0_M or float(data["y0_m"]) != Y0_M):
                raise RuntimeError("The feature atlas uses a different terrain grid")
        if (self.atlas.shape != (ROWS, COLS, FEATURE_DIM)
                or self.valid.shape != (ROWS, COLS)
                or self.axes.shape != (FEATURE_DIM, PCA_DIM)):
            raise RuntimeError("The feature atlas has incompatible dimensions")
        self.atlas_small = self.compress(self.atlas)

    def compress(self, features: np.ndarray) -> np.ndarray:
        return np.clip(((features - self.mean) @ self.axes) / self.scale,
                       -5.0, 5.0).astype(np.float32)

    def local(self, robot_xy: np.ndarray, yaw: float,
              field: np.ndarray, valid: np.ndarray) -> np.ndarray:
        result = np.zeros((LOCAL_CELLS, PCA_DIM + 1), dtype=np.float32)
        cy, sy = np.cos(yaw), np.sin(yaw)
        for i, (dx, dy) in enumerate((x, y) for x in LOOK_X_M for y in LOOK_Y_M):
            # The ZED X horizontal FOV is about 105 degrees.
            if abs(np.arctan2(dy, dx)) > np.deg2rad(50.0):
                continue
            x = robot_xy[0] + cy * dx - sy * dy
            y = robot_xy[1] + sy * dx + cy * dy
            row = int(np.floor((x - X0_M) / RESOLUTION_M))
            col = int(np.floor((y - Y0_M) / RESOLUTION_M))
            if 0 <= row < ROWS and 0 <= col < COLS and valid[row, col]:
                result[i, :PCA_DIM] = field[row, col]
                result[i, -1] = 1.0
        return result.reshape(-1)

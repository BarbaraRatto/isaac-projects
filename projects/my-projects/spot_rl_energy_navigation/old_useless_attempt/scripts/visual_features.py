"""DINO feature field and identical local observation for cached/online modes."""

from pathlib import Path

import numpy as np
import torch
from isaaclab.utils.math import quat_apply


GRID_RESOLUTION = 0.5
GRID_X0 = 0.0
GRID_Y0 = -1.5
GRID_ROWS = 36  # 18 m
GRID_COLS = 12  # 6 m
FEATURE_DIM = 384
REDUCED_DIM = 12
LOOK_X = (1.5, 3.0, 4.5)
LOOK_Y = (-1.2, 0.0, 1.4, 2.8)
LOCAL_CELLS = len(LOOK_X) * len(LOOK_Y)
LOCAL_SIZE = LOCAL_CELLS * (REDUCED_DIM + 1)


def camera_ground_points(camera) -> tuple[np.ndarray, np.ndarray]:
    """Project visible lower-image patch centres onto the world terrain."""
    data = camera.data
    depth = data.output["distance_to_image_plane"][0].cpu().numpy().squeeze()
    h, w = depth.shape
    patch_centres = (np.arange(37) + 0.5) / 37.0
    uu, vv = np.meshgrid(patch_centres * w, patch_centres * h)
    ui = np.clip(np.rint(uu).astype(int), 0, w - 1)
    vi = np.clip(np.rint(vv).astype(int), 0, h - 1)
    z = depth[vi, ui].reshape(-1)
    intr = data.intrinsic_matrices[0].cpu().numpy()
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    valid = np.isfinite(z) & (z > 0.2) & (z < 9.0)
    valid[:18 * 37] = False
    safe_z = np.where(valid, z, 0.0)
    xyz_optical = np.column_stack(((uu.reshape(-1) - cx) * safe_z / fx,
                                   (vv.reshape(-1) - cy) * safe_z / fy, safe_z))
    xyz = torch.from_numpy(xyz_optical[valid].astype(np.float32)).to(data.pos_w.device)
    world = quat_apply(data.quat_w_ros[0].expand(len(xyz), -1), xyz)
    world += data.pos_w[0]
    points = world.cpu().numpy()
    ground = (points[:, 2] > -0.3) & (points[:, 2] < 0.65)
    accepted = np.zeros(len(valid), dtype=bool)
    accepted[np.flatnonzero(valid)[ground]] = True
    return points[ground], accepted


def camera_patches(camera, dino) -> tuple[np.ndarray, np.ndarray]:
    """Return visible world points and the matching live DINO features."""
    rgb = camera.data.output["rgb"][0].cpu().numpy().copy()
    features = dino.extract(rgb).reshape(-1, FEATURE_DIM)
    points, accepted = camera_ground_points(camera)
    return points, features[accepted]


def visible_mask(points: np.ndarray) -> np.ndarray:
    mask = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    rows, cols, _ = indices(points)
    mask[rows, cols] = True
    return mask


def indices(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.floor((points[:, 0] - GRID_X0) / GRID_RESOLUTION).astype(int)
    cols = np.floor((points[:, 1] - GRID_Y0) / GRID_RESOLUTION).astype(int)
    valid = (rows >= 0) & (rows < GRID_ROWS) & (cols >= 0) & (cols < GRID_COLS)
    return rows[valid], cols[valid], valid


class FeatureField:
    def __init__(self):
        self.sums = np.zeros((GRID_ROWS, GRID_COLS, FEATURE_DIM), dtype=np.float64)
        self.count = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)

    def insert(self, points: np.ndarray, features: np.ndarray):
        rows, cols, valid = indices(points)
        np.add.at(self.sums, (rows, cols), features[valid])
        np.add.at(self.count, (rows, cols), 1)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        features = np.zeros_like(self.sums, dtype=np.float32)
        valid = self.count > 0
        features[valid] = (self.sums[valid] / self.count[valid, None]).astype(np.float32)
        return features, valid


class VisualObservation:
    def __init__(self, atlas_file: Path):
        data = np.load(atlas_file)
        self.atlas = data["features"].astype(np.float32)
        self.valid = data["valid"].astype(bool)
        self.mean = data["mean"].astype(np.float32)
        self.axes = data["axes"].astype(np.float32)
        self.scale = data["scale"].astype(np.float32)
        if self.atlas.shape != (GRID_ROWS, GRID_COLS, FEATURE_DIM):
            raise RuntimeError(f"Invalid atlas shape: {self.atlas.shape}")
        self.atlas_small = self.compress(self.atlas)

    def compress(self, features: np.ndarray) -> np.ndarray:
        value = ((features - self.mean) @ self.axes) / self.scale
        return np.clip(value, -5.0, 5.0).astype(np.float32)

    def local(self, robot_xy: np.ndarray, yaw: float,
              field: np.ndarray | None = None,
              valid: np.ndarray | None = None) -> np.ndarray:
        if field is None:
            field, valid = self.atlas_small, self.valid
        out = np.zeros((LOCAL_CELLS, REDUCED_DIM + 1), dtype=np.float32)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        for i, (dx, dy) in enumerate((x, y) for x in LOOK_X for y in LOOK_Y):
            point = np.array([robot_xy[0] + cos_yaw * dx - sin_yaw * dy,
                              robot_xy[1] + sin_yaw * dx + cos_yaw * dy])
            row = int(np.floor((point[0] - GRID_X0) / GRID_RESOLUTION))
            col = int(np.floor((point[1] - GRID_Y0) / GRID_RESOLUTION))
            if abs(np.arctan2(dy, dx)) > np.deg2rad(50.0):
                continue
            candidates = [(r, c) for r in range(max(0, row - 1), min(GRID_ROWS, row + 2))
                          for c in range(max(0, col - 1), min(GRID_COLS, col + 2))
                          if valid[r, c]]
            if candidates:
                nearest_row, nearest_col = min(
                    candidates, key=lambda rc: (rc[0] - row) ** 2 + (rc[1] - col) ** 2
                )
                out[i, :REDUCED_DIM] = field[nearest_row, nearest_col]
                out[i, -1] = 1.0
        return out.reshape(-1)


def save_atlas(path: Path, field: FeatureField):
    features, valid = field.finish()
    samples = features[valid]
    if len(samples) < 30:
        raise RuntimeError(f"Only {len(samples)} observed grid cells")
    mean = samples.mean(axis=0)
    centred = samples - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axes = vt[:REDUCED_DIM].T
    scale = np.maximum(np.std(centred @ axes, axis=0), 1e-4)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, valid=valid, mean=mean,
                        axes=axes, scale=scale, count=field.count)
    return int(valid.sum())

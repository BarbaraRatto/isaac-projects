"""DINO feature positions adapted to a 6x6 m board with 2 m terrain cells."""

import numpy as np

from full_grid_energy_choice.visual_features import (
    FEATURE_DIM, PCA_DIM, VisualObservation,
)

RESOLUTION_M = 0.5
X0_M = Y0_M = -1.0
ROWS = COLS = 12
LOOK_X_M = (0.5, 1.0, 1.5, 2.0)
LOOK_Y_M = (-1.0, -0.5, 0.0, 0.5, 1.0)
LOCAL_CELLS = len(LOOK_X_M) * len(LOOK_Y_M)
VISUAL_SIZE = LOCAL_CELLS * (PCA_DIM + 1)


class BinaryFeatureField:
    def __init__(self):
        self.sums = np.zeros((ROWS, COLS, FEATURE_DIM), dtype=np.float64)
        self.counts = np.zeros((ROWS, COLS), dtype=np.int32)

    def insert(self, points: np.ndarray, features: np.ndarray) -> None:
        if len(points) != len(features) or features.shape[1:] != (FEATURE_DIM,):
            raise ValueError("DINO patches and projected terrain points do not match")
        row = np.floor((points[:, 0] - X0_M) / RESOLUTION_M).astype(int)
        col = np.floor((points[:, 1] - Y0_M) / RESOLUTION_M).astype(int)
        inside = (row >= 0) & (row < ROWS) & (col >= 0) & (col < COLS)
        np.add.at(self.sums, (row[inside], col[inside]), features[inside])
        np.add.at(self.counts, (row[inside], col[inside]), 1)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        valid = self.counts > 0
        features = np.zeros((ROWS, COLS, FEATURE_DIM), dtype=np.float32)
        features[valid] = (self.sums[valid] / self.counts[valid, None]).astype(np.float32)
        return features, valid


class BinaryVisualObservation(VisualObservation):
    """Reuse the old atlas's PCA axes, but no stored features or positions."""

    def local(self, robot_xy: np.ndarray, yaw: float,
              field: np.ndarray, valid: np.ndarray) -> np.ndarray:
        result = np.zeros((LOCAL_CELLS, PCA_DIM + 1), dtype=np.float32)
        cy, sy = np.cos(yaw), np.sin(yaw)
        for i, (dx, dy) in enumerate((x, y) for x in LOOK_X_M for y in LOOK_Y_M):
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

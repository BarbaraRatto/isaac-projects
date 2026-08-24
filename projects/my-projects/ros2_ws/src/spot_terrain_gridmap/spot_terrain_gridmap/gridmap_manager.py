"""
Local feature grid‑map manager.

Maintains a 2‑D numpy grid of DINOv2 feature vectors centred on the robot
in the ``odom`` frame.  Supports:
  • shifting the grid when the robot moves (preserving data in view);
  • inserting features with overwrite semantics;
  • exporting raw data for ROS message construction.

Pure numpy — no ROS dependencies.
"""

from __future__ import annotations

import numpy as np


class GridmapManager:
    """Fixed‑size, robot‑centred feature grid‑map."""

    def __init__(
        self,
        size_x: float = 5.0,
        size_y: float = 5.0,
        resolution: float = 0.1,
        feature_dim: int = 384,
    ) -> None:
        """
        Parameters
        ----------
        size_x, size_y : physical extent of the grid [m].
        resolution : cell side length [m].
        feature_dim : dimensionality of each feature vector (384 for DINOv2‑S).
        """
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.feature_dim = feature_dim

        self.n_rows = int(round(size_x / resolution))  # cells in X  (50)
        self.n_cols = int(round(size_y / resolution))  # cells in Y  (50)

        # Feature storage – NaN means "no data".
        self.feature_grid = np.full(
            (self.n_rows, self.n_cols, feature_dim), np.nan, dtype=np.float32
        )
        self.valid_mask = np.zeros(
            (self.n_rows, self.n_cols), dtype=bool
        )

        # Grid centre in the odom frame (initialised to origin).
        self.center_x: float = 0.0
        self.center_y: float = 0.0
        self._centre_initialised = False

    # ------------------------------------------------------------------
    # Centre management & grid shifting
    # ------------------------------------------------------------------
    def update_center(self, robot_x: float, robot_y: float) -> None:
        """Move the grid centre to track the robot, shifting stored data.

        Cells that remain in view after the shift keep their feature data.
        Newly exposed cells are set to NaN / invalid.

        Parameters
        ----------
        robot_x, robot_y : robot position in the ``odom`` frame [m].
        """
        if not self._centre_initialised:
            # First call – just snap the centre, no shift needed.
            self.center_x = robot_x
            self.center_y = robot_y
            self._centre_initialised = True
            return

        dx_cells = int(round((robot_x - self.center_x) / self.resolution))
        dy_cells = int(round((robot_y - self.center_y) / self.resolution))

        if dx_cells == 0 and dy_cells == 0:
            return  # No movement larger than half a cell.

        # If the shift is larger than the grid, just clear everything.
        if abs(dx_cells) >= self.n_rows or abs(dy_cells) >= self.n_cols:
            self.feature_grid[:] = np.nan
            self.valid_mask[:] = False
        else:
            self._shift_axis(dx_cells, axis=0)
            self._shift_axis(dy_cells, axis=1)

        # Snap the centre to the grid (avoids accumulated float drift).
        self.center_x += dx_cells * self.resolution
        self.center_y += dy_cells * self.resolution

    def _shift_axis(self, shift_cells: int, axis: int) -> None:
        """Roll data along *axis* and clear the newly exposed cells."""
        if shift_cells == 0:
            return

        self.feature_grid = np.roll(self.feature_grid, -shift_cells, axis=axis)
        self.valid_mask = np.roll(self.valid_mask, -shift_cells, axis=axis)

        # Build a slice object that selects the newly exposed region.
        n = self.feature_grid.shape[axis]
        if shift_cells > 0:
            # Robot moved forward → new cells at the "end" of the axis.
            sel = [slice(None)] * 3  # for feature_grid (3‑D)
            sel[axis] = slice(n - shift_cells, n)
            self.feature_grid[tuple(sel)] = np.nan

            sel_2d = [slice(None)] * 2  # for valid_mask (2‑D)
            sel_2d[axis] = slice(n - shift_cells, n)
            self.valid_mask[tuple(sel_2d)] = False
        else:
            abs_shift = -shift_cells  # positive
            sel = [slice(None)] * 3
            sel[axis] = slice(0, abs_shift)
            self.feature_grid[tuple(sel)] = np.nan

            sel_2d = [slice(None)] * 2
            sel_2d[axis] = slice(0, abs_shift)
            self.valid_mask[tuple(sel_2d)] = False

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------
    def world_to_grid(
        self, x: float, y: float
    ) -> tuple[int, int] | None:
        """Convert an odom‑frame (x, y) position to grid indices (row, col).

        Returns ``None`` if the point falls outside the grid.
        """
        row = int(round((x - self.center_x) / self.resolution)) + self.n_rows // 2
        col = int(round((y - self.center_y) / self.resolution)) + self.n_cols // 2

        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return (row, col)
        return None

    # ------------------------------------------------------------------
    # Feature insertion (overwrite)
    # ------------------------------------------------------------------
    def insert_features(
        self,
        points_odom: np.ndarray,
        features: np.ndarray,
        valid_mask: np.ndarray,
    ) -> int:
        """Insert feature vectors into the grid at projected 3‑D positions.

        Uses **overwrite** semantics: the last feature written to a cell wins.

        Parameters
        ----------
        points_odom : (N, 3) – 3‑D points in the odom frame.
        features : (N, feature_dim) – corresponding feature vectors.
        valid_mask : (N,) bool – True for points with valid depth.

        Returns
        -------
        n_inserted : number of features actually written (within grid bounds).
        """
        n_inserted = 0
        for idx in np.where(valid_mask)[0]:
            x, y, _z = points_odom[idx]
            cell = self.world_to_grid(x, y)
            if cell is None:
                continue
            row, col = cell
            self.feature_grid[row, col, :] = features[idx]
            self.valid_mask[row, col] = True
            n_inserted += 1
        return n_inserted

    # ------------------------------------------------------------------
    # Data export (consumed by the ROS node to build messages)
    # ------------------------------------------------------------------
    def get_feature_layer(self, k: int) -> np.ndarray:
        """Return the k‑th feature dimension as a (n_rows, n_cols) array.

        NaN in cells with no data.
        """
        return self.feature_grid[:, :, k]

    def get_occupancy_data(self) -> np.ndarray:
        """Return an int8 array for ``nav_msgs/OccupancyGrid``.

        ``100`` where a valid feature exists, ``0`` elsewhere.
        """
        occ = np.zeros((self.n_rows, self.n_cols), dtype=np.int8)
        occ[self.valid_mask] = 100
        return occ

    def reset(self) -> None:
        """Clear all stored features (full reset)."""
        self.feature_grid[:] = np.nan
        self.valid_mask[:] = False

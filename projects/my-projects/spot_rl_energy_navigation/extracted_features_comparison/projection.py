"""NumPy-only depth projection, pure patch mask, grid means and front samples."""
from __future__ import annotations

import math
import numpy as np


def rotation_matrix(quaternion_wxyz) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Invalid camera quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def project_patches(depth: np.ndarray, record: dict, config: dict, patches: int = 37):
    """Training-compatible patch CENTRES; strict purity checks 9 depth positions.

    Purity is a geometric filter, not semantic segmentation: DINO tokens still
    have image-wide context. Only one labelled terrain is visible per frame.
    """
    height, width = depth.shape
    k = np.asarray(record["intrinsics"])
    rotation = rotation_matrix(record["camera_quaternion_ros_wxyz"])
    translation = np.asarray(record["camera_position_w"])
    rows, cols = np.meshgrid(np.arange(patches), np.arange(patches), indexing="ij")
    bounds = record["terrain_bounds_xy"]

    def sample(offset_x, offset_y):
        u = (cols.ravel() + offset_x) / patches * width
        v = (rows.ravel() + offset_y) / patches * height
        d = depth[np.clip(np.rint(v).astype(int), 0, height-1),
                  np.clip(np.rint(u).astype(int), 0, width-1)]
        valid = np.isfinite(d) & (d > config["depth_min_m"]) & (d < config["depth_max_m"])
        safe = np.where(valid, d, 0)
        optical = np.column_stack(((u-k[0, 2])*safe/k[0, 0], (v-k[1, 2])*safe/k[1, 1], safe))
        world = optical @ rotation.T + translation
        margin = config["boundary_margin_m"]
        valid &= ((world[:, 0] > bounds[0]+margin) & (world[:, 0] < bounds[1]-margin)
                  & (world[:, 1] > bounds[2]+margin) & (world[:, 1] < bounds[3]-margin)
                  & (world[:, 2] > config["z_min_m"]) & (world[:, 2] < config["z_max_m"]))
        return world, valid

    world, center_valid = sample(0.5, 0.5)
    pure = center_valid.copy()
    for dx in (0.05, 0.5, 0.95):
        for dy in (0.05, 0.5, 0.95):
            pure &= sample(dx, dy)[1]
    return world, center_valid, pure


def aggregate_cells(features: np.ndarray, points: np.ndarray, accepted: np.ndarray, config: dict):
    shape = tuple(config["grid_shape"])
    origin = np.asarray(config["grid_origin_m"])
    ids = np.floor((points[:, :2]-origin)/config["cell_resolution_m"]).astype(int)
    col, row = ids[:, 0], ids[:, 1]
    keep = accepted & (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    sums = np.zeros((*shape, features.shape[1]), dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int32)
    np.add.at(sums, (row[keep], col[keep]), features[keep])
    np.add.at(counts, (row[keep], col[keep]), 1)
    valid = counts > 0
    means = np.zeros_like(sums, dtype=np.float32)
    means[valid] = (sums[valid]/counts[valid, None]).astype(np.float32)
    return means, counts


def front_cell_ids(record: dict, counts: np.ndarray, config: dict) -> np.ndarray:
    """Same 20 locations/FOV mask as RL, relative to a virtual robot origin.

    No robot is simulated: its yaw and 0.42 m camera offset define the origin.
    Return UNIQUE grid IDs so duplicate front samples are not independent data.
    """
    yaw = math.radians(record["yaw_deg"])
    rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    robot_xy = np.asarray(record["camera_position_w"][:2]) - rot @ np.array([config["virtual_camera_forward_offset_m"], 0])
    ids = []
    for x in config["look_x_m"]:
        for y in config["look_y_m"]:
            if abs(math.degrees(math.atan2(y, x))) > config["max_front_angle_deg"]:
                continue
            world = robot_xy + rot @ np.array([x, y])
            col, row = np.floor((world-np.asarray(config["grid_origin_m"]))/config["cell_resolution_m"]).astype(int)
            if 0 <= row < counts.shape[0] and 0 <= col < counts.shape[1] and counts[row, col] > 0:
                ids.append(int(row*counts.shape[1]+col))
    return np.asarray(sorted(set(ids)), dtype=np.int32)

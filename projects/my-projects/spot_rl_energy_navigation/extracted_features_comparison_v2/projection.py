"""Mixed-scene depth/semantic projection and labelled 0.5 m cell averages."""
from __future__ import annotations
import math
import numpy as np
from common import LABELS


def rotation_matrix(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Invalid quaternion")
    w, x, y, z = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def project_patches(depth, semantic, record, config, manifest):
    size = config["dino"]["input_size"]//14
    cfg = config["projection"]
    height, width = depth.shape
    row, col = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    k = np.asarray(record["intrinsics"])
    rotation = rotation_matrix(record["camera_quaternion_ros_wxyz"])
    translation = np.asarray(record["camera_position_w"])

    def sample(dx, dy):
        u, v = (col.ravel()+dx)/size*width, (row.ravel()+dy)/size*height
        xi = np.clip(np.rint(u).astype(int), 0, width-1)
        yi = np.clip(np.rint(v).astype(int), 0, height-1)
        d = depth[yi, xi]
        valid = np.isfinite(d) & (d > cfg["depth_min_m"]) & (d < cfg["depth_max_m"])
        safe = np.where(valid, d, 0)
        optical = np.column_stack(((u-k[0, 2])*safe/k[0, 0], (v-k[1, 2])*safe/k[1, 1], safe))
        points = optical @ rotation.T + translation
        labels = semantic[yi, xi].astype(np.int16).copy()
        instances = np.full(len(d), -1, dtype=np.int16)
        # Renderer labels exclude Spot/background; XY bounds identify the donor
        # instance and remove edge skirts. No upper absolute-Z cutoff here.
        margin = cfg["boundary_margin_m"]
        for i, station in enumerate(manifest["stations"]):
            xmin, xmax, ymin, ymax = station["bounds_xy"]
            inside = ((points[:, 0] > xmin+margin) & (points[:, 0] < xmax-margin)
                      & (points[:, 1] > ymin+margin) & (points[:, 1] < ymax-margin)
                      & (labels == LABELS.index(station["label"])))
            instances[inside] = i
        valid &= (instances >= 0) & (points[:, 2] > cfg["surface_z_min_m"])
        labels[~valid], instances[~valid] = -1, -1
        return points, labels, instances, valid

    points, center_labels, instances, valid = sample(.5, .5)
    pure_labels = center_labels.copy()
    for dx in (.05, .5, .95):
        for dy in (.05, .5, .95):
            _, labels, instance, _ = sample(dx, dy)
            pure_labels[(labels != center_labels) | (instance != instances)] = -1
    return points, center_labels, pure_labels, instances, valid


def front_ids(record, cfg):
    state = record["robot_root_state_w"]
    w, x, y, z = state[3:7]
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    rows, cols = cfg["grid_shape"]
    result = set()
    for x in cfg["look_x_m"]:
        for y in cfg["look_y_m"]:
            if abs(math.degrees(math.atan2(y, x))) > cfg["max_front_angle_deg"]:
                continue
            xy = np.asarray(state[:2]) + rotation @ np.array([x, y])
            col, row = np.floor((xy-np.asarray(cfg["grid_origin_m"]))/cfg["cell_resolution_m"]).astype(int)
            if 0 <= row < rows and 0 <= col < cols:
                result.add(int(row*cols+col))
    return result


def aggregate_cells(features, points, labels, instances, valid, record, cfg):
    """Average raw vectors, reject mixed/unknown cells, retain known donor IDs.

    Unknown centres count against purity. Also calculate the legacy absolute
    height-filter branch separately so high surfaces are not silently lost.
    """
    origin = np.asarray(cfg["grid_origin_m"])
    col, row = np.floor((points[:, :2]-origin)/cfg["cell_resolution_m"]).astype(int).T
    rows, cols = cfg["grid_shape"]
    inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    ids = row*cols+col
    selected_front = front_ids(record, cfg)
    result = {name: [] for name in ("vectors", "labels", "instances", "ids", "front", "rl_vectors", "rl_labels", "rl_instances", "rl_ids")}
    for cell_id in np.unique(ids[inside]):
        all_ids = np.flatnonzero(inside & (ids == cell_id))
        known = all_ids[valid[all_ids] & (labels[all_ids] >= 0)]
        if not len(known):
            continue
        # Require near-pure terrain class AND donor instance within this cell.
        unique, count = np.unique(instances[known], return_counts=True)
        donor = int(unique[np.argmax(count)])
        picked = known[instances[known] == donor]
        if len(picked)/len(all_ids) < cfg["cell_label_purity"]:
            continue
        result["vectors"].append(features[known].mean(axis=0))
        result["labels"].append(int(labels[picked[0]]))
        result["instances"].append(donor)
        result["ids"].append(int(cell_id))
        result["front"].append(cell_id in selected_front)
        legacy = known[(points[known, 2] > cfg["rl_z_min_m"]) & (points[known, 2] < cfg["rl_z_max_m"])]
        if len(legacy) and cell_id in selected_front:
            result["rl_vectors"].append(features[legacy].mean(axis=0))
            result["rl_labels"].append(int(labels[picked[0]]))
            result["rl_instances"].append(donor)
            result["rl_ids"].append(int(cell_id))
    for key in result:
        result[key] = np.asarray(result[key], dtype=np.float32 if key in ("vectors", "rl_vectors") else (bool if key == "front" else np.int32))
    for key in ("vectors", "rl_vectors"):
        result[key] = result[key].reshape(-1, features.shape[1])
    return result

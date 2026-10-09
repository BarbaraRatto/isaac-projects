"""One frozen DINOv2 for all ZED X views in the full-grid task."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from isaaclab.utils.math import quat_apply

from full_grid_energy_choice.visual_features import (COLS, FEATURE_DIM, PCA_DIM, ROWS,
                                                       FeatureField, VisualObservation)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "spot_terrain_gridmap"))
from spot_terrain_gridmap.dino_feature_extractor import DinoFeatureExtractor  # noqa: E402


class BatchedDino:
    def __init__(self, atlas_file: Path, device: str, batch_size: int = 4):
        if batch_size < 1:
            raise ValueError("DINO batch_size must be positive")
        self.batch_size = batch_size
        started = time.perf_counter()
        self.extractor = DinoFeatureExtractor(
            model_name="facebook/dinov2-small", device=device, input_size=518
        )
        self.visual = VisualObservation(atlas_file)
        self.load_seconds = time.perf_counter() - started
        self.last_dino_ms = 0.0
        self.last_visual_ms = 0.0

    def extract(self, camera, origins: torch.Tensor, robot_xy: torch.Tensor,
                robot_yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        started = time.perf_counter()
        rgb = camera.data.output["rgb"].cpu().numpy()[..., :3].copy()
        model = self.extractor
        features = np.empty((len(rgb), model.n_patches ** 2, FEATURE_DIM), dtype=np.float32)
        dino_ms = 0.0
        # Keep DINO's peak GPU memory near the proven four-camera case even
        # when Isaac Lab renders sixteen views at once.
        for start in range(0, len(rgb), self.batch_size):
            chunk = rgb[start:start + self.batch_size]
            resized = [cv2.resize(frame, (518, 518), interpolation=cv2.INTER_LINEAR)
                       for frame in chunk]
            inputs = model.processor(
                images=resized, return_tensors="pt", do_resize=False, do_center_crop=False
            )
            dino_started = time.perf_counter()
            with torch.inference_mode():
                output = model.model(pixel_values=inputs["pixel_values"].to(model.device))
                patch_features = output.last_hidden_state[:, 1:, :].reshape(
                    len(chunk), -1, FEATURE_DIM
                )
            features[start:start + len(chunk)] = patch_features.float().cpu().numpy()
            dino_ms += (time.perf_counter() - dino_started) * 1000.0
        self.last_dino_ms = dino_ms

        origin_np = origins.detach().cpu().numpy()
        xy_np = robot_xy.detach().cpu().numpy()
        yaw_np = robot_yaw.detach().cpu().numpy()
        results = []
        valid_counts = []
        for i in range(len(rgb)):
            points, accepted = ground_points(camera, i)
            points[:, :2] -= origin_np[i, :2]
            field = FeatureField()
            field.insert(points, features[i, accepted])
            raw, valid = field.finish()
            # Only project occupied grid cells; the original full-grid dense
            # multiplication would waste time on thousands of empty cells.
            compressed = np.zeros((ROWS, COLS, PCA_DIM), dtype=np.float32)
            compressed[valid] = self.visual.compress(raw[valid])
            local = self.visual.local(
                xy_np[i] - origin_np[i, :2], float(yaw_np[i]),
                compressed, valid,
            )
            results.append(local)
            valid_counts.append(int(local.reshape(-1, 33)[:, -1].sum()))
        self.last_visual_ms = (time.perf_counter() - started) * 1000.0
        return (torch.from_numpy(np.stack(results)).to(robot_xy.device),
                torch.tensor(valid_counts, device=robot_xy.device))


def ground_points(camera, env_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Same patch-centre/depth projection and terrain filter as ice_visual_features."""
    data = camera.data
    depth = data.output["distance_to_image_plane"][env_id].cpu().numpy().squeeze()
    height, width = depth.shape
    centres = (np.arange(37) + 0.5) / 37.0
    uu, vv = np.meshgrid(centres * width, centres * height)
    ui = np.clip(np.rint(uu).astype(int), 0, width - 1)
    vi = np.clip(np.rint(vv).astype(int), 0, height - 1)
    distance = depth[vi, ui].reshape(-1)
    intrinsic = data.intrinsic_matrices[env_id].cpu().numpy()
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    depth_valid = np.isfinite(distance) & (distance > 0.2) & (distance < 15.0)
    safe = np.where(depth_valid, distance, 0.0)
    optical = np.column_stack(((uu.reshape(-1) - cx) * safe / fx,
                               (vv.reshape(-1) - cy) * safe / fy, safe))
    optical_valid = torch.from_numpy(optical[depth_valid].astype(np.float32)).to(data.pos_w.device)
    world = quat_apply(data.quat_w_ros[env_id].expand(len(optical_valid), -1), optical_valid)
    world += data.pos_w[env_id]
    points = world.cpu().numpy()
    terrain = (points[:, 2] > -0.3) & (points[:, 2] < 0.65)
    accepted = np.zeros(len(depth_valid), dtype=bool)
    accepted[np.flatnonzero(depth_valid)[terrain]] = True
    return points[terrain], accepted

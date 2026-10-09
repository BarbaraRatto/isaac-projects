"""One frozen DINO model for all cameras; nearby samples for 2 m cells."""

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from binary_choice.visual_features import (
    BinaryFeatureField, BinaryVisualObservation, COLS, FEATURE_DIM, PCA_DIM, ROWS,
)
from full_grid_energy_choice.vision import BatchedDino, ground_points


class BinaryBatchedDino(BatchedDino):
    def __init__(self, atlas_file: Path, device: str, batch_size: int = 4):
        super().__init__(atlas_file, device, batch_size=batch_size)
        self.visual = BinaryVisualObservation(atlas_file)

    def extract(self, camera, origins: torch.Tensor, robot_xy: torch.Tensor,
                robot_yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        started = time.perf_counter()
        rgb = camera.data.output["rgb"].cpu().numpy()[..., :3].copy()
        model = self.extractor
        features = np.empty((len(rgb), model.n_patches ** 2, FEATURE_DIM), dtype=np.float32)
        dino_ms = 0.0
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
            field = BinaryFeatureField()
            field.insert(points, features[i, accepted])
            raw, valid = field.finish()
            compressed = np.zeros((ROWS, COLS, PCA_DIM), dtype=np.float32)
            compressed[valid] = self.visual.compress(raw[valid])
            local = self.visual.local(xy_np[i] - origin_np[i, :2],
                                      float(yaw_np[i]), compressed, valid)
            results.append(local)
            valid_counts.append(int(local.reshape(-1, PCA_DIM + 1)[:, -1].sum()))
        self.last_visual_ms = (time.perf_counter() - started) * 1000.0
        return (torch.from_numpy(np.stack(results)).to(robot_xy.device),
                torch.tensor(valid_counts, device=robot_xy.device))

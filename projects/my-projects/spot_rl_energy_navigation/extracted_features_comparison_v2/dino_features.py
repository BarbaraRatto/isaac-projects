"""Same RGB preprocessing/patch tokens as test6; local weights, no downloads."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from common import local_path


class DinoFeatures:
    def __init__(self, config: dict, device: str):
        from transformers import AutoImageProcessor, AutoModel

        self.size = config["input_size"]
        self.device = torch.device(device)
        directory = str(local_path(config["directory"]))
        self.processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True)
        self.model = AutoModel.from_pretrained(directory, local_files_only=True).to(self.device)
        self.model.eval().requires_grad_(False)
        if self.model.config.hidden_size != config["feature_dim"] or self.model.config.patch_size != 14:
            raise ValueError("Expected DINOv2-small: 384 components, patch size 14")

    @torch.inference_mode()
    def extract(self, rgb: np.ndarray) -> np.ndarray:
        image = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        inputs = self.processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)
        tokens = self.model(pixel_values=inputs["pixel_values"].to(self.device)).last_hidden_state[:, 1:, :]
        return tokens[0].float().cpu().numpy()

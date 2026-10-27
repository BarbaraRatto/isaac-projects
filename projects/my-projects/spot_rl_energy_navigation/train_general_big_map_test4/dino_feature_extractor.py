"""
Wrapper around DINOv2 (frozen) for patch‑level feature extraction.

Handles image preprocessing and frozen patch extraction; no PCA.
Pure Python + PyTorch — no ROS dependencies.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


class DinoFeatureExtractor:
    """Extract patch features from images using a frozen DINOv2 model."""

    PATCH_SIZE = 14  # ViT patch side in pixels (fixed for DINOv2).

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device: str = "cuda",
        input_size: int = 518,
    ) -> None:
        """
        Parameters
        ----------
        model_name : HuggingFace model identifier.
        device : ``"cuda"`` or ``"cpu"``.
        input_size : Square side to resize images to (must be a multiple of 14).
        """
        if input_size % self.PATCH_SIZE != 0:
            raise ValueError(
                f"input_size ({input_size}) must be a multiple of "
                f"PATCH_SIZE ({self.PATCH_SIZE})."
            )

        self.device = torch.device(device)
        self.input_size = input_size
        self.n_patches = input_size // self.PATCH_SIZE  # 37 for 518

        # ---- Load model (frozen) ----------------------------------------
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_dim: int = self.model.config.hidden_size  # 384 for dinov2‑small

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract(self, rgb_image: np.ndarray) -> np.ndarray:
        """Run DINOv2 and return spatial patch features.

        Parameters
        ----------
        rgb_image : (H, W, 3) uint8, RGB colour order.

        Returns
        -------
        features : numpy (n_patches, n_patches, feature_dim)
                   e.g. (37, 37, 384) for input_size=518.
        """
        # Resize to the square input expected by the model.
        resized = cv2.resize(
            rgb_image,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # HuggingFace processor: normalises + converts to tensor.
        inputs = self.processor(
            images=resized, return_tensors="pt",
            do_resize=False,        # already resized
            do_center_crop=False,   # keep full 518×518
        )
        pixel_values = inputs["pixel_values"].to(self.device)

        # Forward pass.
        outputs = self.model(pixel_values=pixel_values)

        # Remove CLS token → (1, n_patches², feature_dim).
        patch_tokens = outputs.last_hidden_state[:, 1:, :]

        # Reshape to spatial grid → (n_patches, n_patches, feature_dim).
        features = (
            patch_tokens[0]
            .cpu()
            .numpy()
            .reshape(self.n_patches, self.n_patches, self.feature_dim)
        )
        return features


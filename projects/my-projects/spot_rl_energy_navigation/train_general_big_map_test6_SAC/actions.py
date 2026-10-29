"""Normalized forward/yaw actions mapped to the frozen locomotion controller."""

import torch

from train_general_big_map_test6_SAC.config import NAV_CFG


def navigation_commands(actions: torch.Tensor) -> torch.Tensor:
    """Normalized navigation actions: (-1, 0) stops; (0, 0) advances at 0.8 m/s."""
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) actions, got {actions.shape}")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("Non-finite navigation actions")
    return torch.stack((
        (actions[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5 * NAV_CFG.max_forward_m_s,
        torch.zeros_like(actions[:, 0]),
        actions[:, 1].clamp(-1.0, 1.0) * NAV_CFG.max_yaw_rad_s,
    ), dim=1)

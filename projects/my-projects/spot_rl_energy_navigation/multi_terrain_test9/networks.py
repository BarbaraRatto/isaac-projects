"""Trainable two-branch visual/state encoders for SAC actor and Q critics."""

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from multi_terrain_test9.config import (
    OBSERVATION_SIZE, STATE_LAYERS, VISUAL_LAYERS, FUSION_LAYERS,
)


def mlp(input_size: int, widths: tuple[int, ...]) -> nn.Sequential:
    layers = []
    for width in widths:
        layers.extend((nn.Linear(input_size, width), nn.ELU()))
        input_size = width
    return nn.Sequential(*layers)


class TwoBranchExtractor(BaseFeaturesExtractor):
    """Observations keep test5 ordering: state first (11), then raw vision (7700)."""

    def __init__(self, observation_space: gym.spaces.Box):
        if observation_space.shape != (OBSERVATION_SIZE,):
            raise ValueError(f"Expected {OBSERVATION_SIZE} input values")
        super().__init__(observation_space, features_dim=VISUAL_LAYERS[-1] + STATE_LAYERS[-1])
        self.visual_branch = mlp(OBSERVATION_SIZE - 11, VISUAL_LAYERS)
        self.state_branch = mlp(11, STATE_LAYERS)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        visual = self.visual_branch(observations[:, 11:])
        state = self.state_branch(observations[:, :11])
        return torch.cat((visual, state), dim=1)


def policy_kwargs() -> dict:
    # SAC uses a separate actor encoder, critic encoder and target critic encoder.
    # The twin Q heads each concatenate the 2 actions after feature extraction.
    return {
        "features_extractor_class": TwoBranchExtractor,
        "net_arch": {"pi": list(FUSION_LAYERS), "qf": list(FUSION_LAYERS)},
        "activation_fn": nn.ELU,
        "share_features_extractor": False,
        "n_critics": 2,
    }

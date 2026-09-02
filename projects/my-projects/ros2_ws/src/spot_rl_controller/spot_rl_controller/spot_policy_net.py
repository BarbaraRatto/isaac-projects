#!/usr/bin/env python3
"""
spot_policy_net.py

Architettura della rete neurale per la policy RL di Spot.

Implementa un features extractor custom per Stable-Baselines3 che combina:
  - Una CNN per elaborare la gridmap spaziale (50×50×N_LAYERS)
  - Un MLP per elaborare i vettori scalari (goal, energia, velocità)

L'output del features extractor viene poi passato alla policy MLP di SB3
per produrre l'azione (vx, vy, omega).

Compatibile con SB3 >= 2.0 e observation space di tipo Dict.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpotFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features extractor custom per la policy RL di Spot.

    Riceve un observation space Dict con:
      - "gridmap":    (N_LAYERS, N_ROWS, N_COLS) float32  — gridmap DINOv2 compressa
      - "goal_vec":   (4,) float32                         — (dx, dy, cos_dtheta, sin_dtheta)
      - "energy_vec": (3,) float32                         — (P_inst, P_avg, E_cum_norm)
      - "robot_vel":  (3,) float32                         — (vx, vy, omega)

    Produce un vettore features 1D di dimensione `features_dim`.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        cnn_channels: list[int] | None = None,
        cnn_kernel_size: int = 3,
        mlp_hidden_sizes: list[int] | None = None,
    ) -> None:
        """
        Parametri
        ---------
        observation_space : spazio di osservazione Dict (da Gymnasium).
        features_dim      : dimensione dell'output del features extractor.
        cnn_channels      : numero di canali per ogni layer Conv2D della CNN.
        cnn_kernel_size   : kernel size Conv2D.
        mlp_hidden_sizes  : hidden layer dell'MLP per i vettori scalari.
        """
        super().__init__(observation_space, features_dim=features_dim)

        # Valori di default per l'architettura
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        if mlp_hidden_sizes is None:
            mlp_hidden_sizes = [512, 256]

        # ── Dimensioni dell'input ──────────────────────────────────────────
        gridmap_shape = observation_space.spaces["gridmap"].shape   # (C, H, W)
        n_channels     = gridmap_shape[0]   # 16 (PCA) + 1 (energy cost) = 17
        n_rows         = gridmap_shape[1]   # 50
        n_cols         = gridmap_shape[2]   # 50

        # Dimensione vettori scalari concatenati
        scalar_dim = (
            observation_space.spaces["goal_vec"].shape[0]    # 4
            + observation_space.spaces["energy_vec"].shape[0]  # 3
            + observation_space.spaces["robot_vel"].shape[0]   # 3
        )  # totale: 10

        # ── CNN per la gridmap ─────────────────────────────────────────────
        # Input: (batch, n_channels, n_rows, n_cols) = (B, 17, 50, 50)
        # Usa Global Average Pooling dopo le conv per rendere l'uscita
        # indipendente dalla risoluzione della gridmap (utile se si cambia).
        cnn_layers: list[nn.Module] = []
        in_ch = n_channels
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=cnn_kernel_size, padding=1),
                nn.ReLU(),
            ]
            in_ch = out_ch
        # Global Average Pooling → (B, last_ch)
        cnn_layers.append(nn.AdaptiveAvgPool2d(1))
        cnn_layers.append(nn.Flatten())  # (B, last_ch)
        self.cnn = nn.Sequential(*cnn_layers)
        cnn_out_dim = cnn_channels[-1]

        # ── MLP per i vettori scalari ──────────────────────────────────────
        scalar_layers: list[nn.Module] = []
        in_dim = scalar_dim
        for hidden in mlp_hidden_sizes:
            scalar_layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        self.scalar_mlp = nn.Sequential(*scalar_layers)
        scalar_out_dim = mlp_hidden_sizes[-1]

        # ── Layer di fusione (CNN + MLP) → features_dim ───────────────────
        fusion_in_dim = cnn_out_dim + scalar_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass del features extractor.

        Parametri
        ---------
        observations : dict con chiavi "gridmap", "goal_vec", "energy_vec", "robot_vel".

        Ritorna
        -------
        Tensor (B, features_dim) — vettore features per la policy.
        """
        # Elaborazione CNN della gridmap
        gridmap_feat = self.cnn(observations["gridmap"])  # (B, cnn_out_dim)

        # Concatenazione vettori scalari
        scalars = torch.cat([
            observations["goal_vec"],
            observations["energy_vec"],
            observations["robot_vel"],
        ], dim=-1)  # (B, 10)

        # Elaborazione MLP degli scalari
        scalar_feat = self.scalar_mlp(scalars)  # (B, scalar_out_dim)

        # Fusione e output
        fused = torch.cat([gridmap_feat, scalar_feat], dim=-1)
        return self.fusion(fused)  # (B, features_dim)

"""Test5 settings; frozen shared settings are copied, never mutated."""

from dataclasses import asdict, dataclass, replace

from binary_choice.config import NAV_CFG_BINARY, TERRAIN_LABELS


NAV_CFG = replace(NAV_CFG_BINARY, max_lateral_m_s=0.0)
FEATURE_DIM = 384
RESOLUTION_M = 0.5
X0_M = Y0_M = -1.0
ROWS = COLS = 12
LOOK_X_M = (0.5, 1.0, 1.5, 2.0)
LOOK_Y_M = (-1.0, -0.5, 0.0, 0.5, 1.0)
LOCAL_CELLS = len(LOOK_X_M) * len(LOOK_Y_M)
VISUAL_SIZE = LOCAL_CELLS * (FEATURE_DIM + 1)
OBSERVATION_SIZE = 11 + VISUAL_SIZE
ACTION_LOW = (0.0, -1.0)
ACTION_HIGH = (1.0, 1.0)


@dataclass(frozen=True)
class PPOSettings:
    timesteps: int = 409_600
    num_envs: int = 32
    n_steps_per_env: int = 32
    batch_size: int = 256
    n_epochs: int = 5
    learning_rate: float = 3e-4
    gamma: float = 0.999
    net_arch: tuple[int, ...] = (512, 256, 128)


PPO_CFG = PPOSettings()


def experiment_signature() -> dict:
    """JSON-compatible settings checked before evaluating a checkpoint."""
    return {
        "experiment": "multi_terrain_test5",
        "environment": asdict(NAV_CFG),
        "observation_size": OBSERVATION_SIZE,
        "visual_components": FEATURE_DIM,
        "visual_representation": "raw_dino_cell_mean_no_pca",
        "local_visual_cells": LOCAL_CELLS,
        "visual_sampling": {
            "look_x_m": list(LOOK_X_M), "look_y_m": list(LOOK_Y_M),
            "resolution_m": RESOLUTION_M, "x0_m": X0_M, "y0_m": Y0_M,
            "max_angle_deg": 50.0, "persistent": False,
        },
        "action_low": list(ACTION_LOW), "action_high": list(ACTION_HIGH),
        "action_order": ["forward_fraction", "yaw_fraction"],
        "net_arch": list(PPO_CFG.net_arch), "activation": "Tanh",
    }

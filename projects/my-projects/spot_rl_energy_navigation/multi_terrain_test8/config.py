"""Local test8 settings: SAC, raw DINO and separate visual/state branches."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass(frozen=True)
class BinaryGeometry:
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    start_yaw_rad: float
    start_jitter_m: float = 0.05
    world_x_bounds: tuple[float, float] = (-0.8, 4.8)
    world_y_bounds: tuple[float, float] = (-0.8, 4.8)


@dataclass(frozen=True)
class BinaryCase:
    name: str
    terrain_usd: Path
    grid: tuple[tuple[str, ...], ...]
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    cell_size_m: float = 2.0
    start_jitter_m: float = 0.05

    @property
    def geometry(self) -> BinaryGeometry:
        width = len(self.grid[0]) * self.cell_size_m
        height = len(self.grid) * self.cell_size_m
        return BinaryGeometry(
            start_xy=self.start_xy,
            goal_xy=self.goal_xy,
            start_yaw_rad=math.atan2(self.goal_xy[1] - self.start_xy[1],
                                     self.goal_xy[0] - self.start_xy[0]),
            start_jitter_m=self.start_jitter_m,
            world_x_bounds=(-self.cell_size_m / 2 + 0.2,
                            width - self.cell_size_m / 2 - 0.2),
            world_y_bounds=(-self.cell_size_m / 2 + 0.2,
                            height - self.cell_size_m / 2 - 0.2),
        )


@dataclass(frozen=True)
class NavigationConfig:
    # One high-level velocity action lasts 0.2 s; the frozen joint policy still
    # receives observations at its own 20 Hz rate.
    physics_dt: float = 1.0 / 200.0
    physics_steps_per_action: int = 40
    settle_seconds: float = 2.0
    max_episode_seconds: float = 40.0
    spawn_height_m: float = 0.75
    goal_radius_m: float = 0.55
    minimum_body_height_m: float = 0.25

    max_forward_m_s: float = 1.6
    max_lateral_m_s: float = 0.0
    max_yaw_rad_s: float = 1.5

    # Reward per high-level step. Energy is mechanical work from measured
    # PhysX joint efforts; test8 changes only the energy coefficient to 0.04.
    progress_reward_per_m: float = 3.0
    energy_penalty_per_j: float = 0.04
    time_penalty_per_step: float = 0.01
    success_bonus: float = 60.0
    failure_penalty: float = 25.0

    @property
    def max_episode_steps(self) -> int:
        return round(self.max_episode_seconds / (self.physics_dt * self.physics_steps_per_action))


NAV_CFG = NavigationConfig()

TERRAIN_NAMES = (
    "t1_asphalt", "t2_slippery", "t3_ramp", "t4_fine_gravel",
    "t5_large_rocks", "t6_obstacles", "t7_stairs",
)
TERRAIN_LABELS = ("asphalt", "ice", "ramp", "gravel", "rocks", "obstacles", "stairs")
TERRAIN_CODES = dict(zip(TERRAIN_NAMES, range(len(TERRAIN_NAMES))))


FEATURE_DIM = 384
RESOLUTION_M = 0.5
X0_M = Y0_M = -1.0
ROWS = COLS = 12
LOOK_X_M = (0.5, 1.0, 1.5, 2.0)
LOOK_Y_M = (-1.0, -0.5, 0.0, 0.5, 1.0)
LOCAL_CELLS = len(LOOK_X_M) * len(LOOK_Y_M)
VISUAL_SIZE = LOCAL_CELLS * (FEATURE_DIM + 1)
OBSERVATION_SIZE = 11 + VISUAL_SIZE
ACTION_LOW = (-1.0, -1.0)
ACTION_HIGH = (1.0, 1.0)


@dataclass(frozen=True)
class SACSettings:
    timesteps: int = 100_000
    num_envs: int = 32
    buffer_size: int = 100_000
    learning_starts: int = 4_096
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.999
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    target_entropy: str = "auto"
    checkpoint_every: int = 51_200
    eval_every: int = 51_200


SAC_CFG = SACSettings()
VISUAL_LAYERS = (2048, 1024)
STATE_LAYERS = (128, 128)
FUSION_LAYERS = (512, 256)


def experiment_signature() -> dict:
    return {
        "experiment": "multi_terrain_test8", "algorithm": "SAC",
        "environment": asdict(NAV_CFG), "observation_size": OBSERVATION_SIZE,
        "visual_components": FEATURE_DIM, "local_visual_cells": LOCAL_CELLS,
        "visual_representation": "raw_dino_cell_mean_no_pca",
        "visual_sampling": {
            "look_x_m": list(LOOK_X_M), "look_y_m": list(LOOK_Y_M),
            "resolution_m": RESOLUTION_M, "x0_m": X0_M, "y0_m": Y0_M,
            "max_angle_deg": 50.0, "persistent": False,
        },
        "action_low": list(ACTION_LOW), "action_high": list(ACTION_HIGH),
        "action_order": ["forward_normalized", "yaw_normalized"],
        "action_mapping": "vx=(a0+1)/2*1.6; vy=0; wz=a1*1.5",
        "architecture": {
            "visual": list(VISUAL_LAYERS), "state": list(STATE_LAYERS),
            "fusion": list(FUSION_LAYERS), "activation": "ELU",
            "share_actor_critic_encoder": False, "n_critics": 2,
        },
    }


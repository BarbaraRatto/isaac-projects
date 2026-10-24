"""Frozen big-map experiment: independent random missions and asymmetric PPO."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass(frozen=True)
class NavigationConfig:
    physics_dt: float = 0.005
    physics_steps_per_action: int = 40
    settle_seconds: float = 2.0
    max_episode_seconds: float = 90.0
    spawn_height_m: float = 0.75
    goal_radius_m: float = 0.55
    minimum_body_height_m: float = 0.25
    max_forward_m_s: float = 1.6
    max_yaw_rad_s: float = 1.5
    progress_reward_per_m: float = 3.0
    time_penalty_per_step: float = 0.02
    success_bonus: float = 150.0
    failure_penalty: float = 300.0
    energy_reference_j: float = 100.0
    energy_weight: float = 1.0
    power_filter_seconds: float = 1.0

    @property
    def action_dt(self):
        return self.physics_dt * self.physics_steps_per_action

    @property
    def max_episode_steps(self):
        return round(self.max_episode_seconds / self.action_dt)


@dataclass(frozen=True)
class PPOSettings:
    timesteps: int = 600_000
    num_envs: int = 32
    rollout_steps: int = 128
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5  # for EACH of the two MSE losses
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    checkpoint_every: int = 50_000
    eval_every: int = 50_000
    eval_episodes_per_env: int = 3


@dataclass(frozen=True)
class MissionSettings:
    distance_bands: tuple = ((3.0, 6.0), (6.0, 12.0), (12.0, 20.0))
    easier_probability: float = 0.25
    promotion_success: float = 0.80
    promotion_evaluations: int = 2
    spawn_excluded: tuple = ("t2_slippery",)
    goal_excluded: tuple = ("t2_slippery", "t7_stairs")
    cell_margin_m: float = 0.65
    footprint_radius_m: float = 0.55
    max_spawn_relief_m: float = 0.16
    deadline_min_s: float = 30.0
    deadline_max_s: float = 90.0
    deadline_base_s: float = 15.0
    deadline_seconds_per_m: float = 4.0
    evaluation_seed: int = 710031
    reachability_filter: bool = False
    distance_metric: str = "euclidean_xy"


MISSION_CFG = MissionSettings()
NAV_CFG = NavigationConfig()
PPO_CFG = PPOSettings()
RAW_FEATURE_DIM = 384
FEATURE_DIM = 32
STATE_SIZE = 15
CELL_SIZE = 40
LOCAL_CELLS = 81
OBSERVATION_SIZE = STATE_SIZE + LOCAL_CELLS * CELL_SIZE
PRIVILEGED_SIZE = 24
CRITIC_SIZE = OBSERVATION_SIZE + PRIVILEGED_SIZE
LAYERS = (512, 256, 128, 64)
RESOLUTION_M = 0.5
LOOK_X_M = tuple(i * 0.5 for i in range(-4, 5))
LOOK_Y_M = LOOK_X_M
MEMORY_RESOLUTION_M = 0.10
MEMORY_AGE_SECONDS = 90.0
DEPTH_STRIDE = 4
GEOMETRY_MIN_POINTS = 10
GEOMETRY_MIN_XY_EIGENVALUE = 1e-4  # square metres; rejects narrow/collinear support
FOOT_NAMES = ('fl_foot', 'fr_foot', 'hl_foot', 'hr_foot')
CONTACT_THRESHOLD_N = 1.0
ACTION_LOW = (-1.0, -1.0)
ACTION_HIGH = (1.0, 1.0)
TERRAIN_NAMES = ('t1_asphalt', 't2_slippery', 't3_ramp', 't4_fine_gravel',
                 't5_large_rocks', 't6_obstacles', 't7_stairs')
TERRAIN_CODES = dict(zip(TERRAIN_NAMES, range(len(TERRAIN_NAMES))))
DINO_MODEL = 'facebook/dinov2-small'
DINO_INPUT_SIZE = 518
HERE = Path(__file__).resolve().parent
DEFAULT_PCA = HERE / 'artifacts' / 'pca32.npz'
# Prefer already downloaded immutable local weights, avoiding network on the cluster.
LOCAL_DINO = HERE.parent / 'extracted_features_comparison_v2' / 'models' / 'dinov2'


def experiment_signature():
    return dict(schema=1, experiment='train_general_big_map_test1',
                algorithm='asymmetric_multi_head_PPO', navigation=asdict(NAV_CFG),
                ppo=asdict(PPO_CFG), actor_size=OBSERVATION_SIZE, critic_size=CRITIC_SIZE,
                layers=list(LAYERS), feature_dim=FEATURE_DIM,
                state_order=['goal_xy', 'linear_velocity_xyz', 'angular_velocity_xyz',
                             'projected_gravity_xyz', 'previous_action_xy', 'power_ema', 'time_left'],
                cell_order=['pca32', 'relative_height', 'slope_x', 'slope_y', 'roughness',
                            'visual_valid', 'geometry_valid', 'visual_age', 'geometry_age'],
                privileged_order=['mu_static_dynamic_4x2', 'contact4', 'normal_force4', 'slip4x2'],
                feet=list(FOOT_NAMES), look_x=list(LOOK_X_M), look_y=list(LOOK_Y_M),
                cell_resolution_m=RESOLUTION_M, map_persistent=True, memory_resolution_m=MEMORY_RESOLUTION_M,
                memory_age_seconds=MEMORY_AGE_SECONDS, memory_fusion="latest_per_world_xy_bin",
                memory_age_aggregation="sample_weighted_mean", localization="simulator_pose",
                depth_stride=DEPTH_STRIDE, geometry_min_points=GEOMETRY_MIN_POINTS,
                geometry_min_xy_eigenvalue=GEOMETRY_MIN_XY_EIGENVALUE,
                terminal_timeout=True, missions=asdict(MISSION_CFG),
                env_spacing_m=80.0, camera_far_clip_m=20.0, reset_pose="startup_settled_stance_above_local_footprint",
                startup_clearance_m=0.02)



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
    minimum_body_height_m: float = 0.25  # diagnostics only; never a fall trigger
    fall_tilt_deg: float = 90.0
    fall_persistence_seconds: float = 1.0
    body_contact_threshold_n: float = 1.0
    max_forward_m_s: float = 1.6
    max_yaw_rad_s: float = 1.5
    progress_reward_per_m: float = 30.0
    time_penalty_per_step: float = 0.02
    success_bonus: float = 1000.0
    failure_penalty: float = 1000.0
    energy_reference_j: float = 100.0
    energy_weight: float = 1.0
    power_filter_seconds: float = 1.0
    heading_penalty_start_deg: float = 110.0
    heading_penalty_per_second: float = 2.0

    @property
    def action_dt(self):
        return self.physics_dt * self.physics_steps_per_action

    @property
    def max_episode_steps(self):
        return round(self.max_episode_seconds / self.action_dt)


@dataclass(frozen=True)
class PPOSettings:
    timesteps: int = 200_000
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
    checkpoint_every: int = 10_000
    eval_every: int = 10_000
    eval_episodes_per_env: int = 1


@dataclass(frozen=True)
class MissionSettings:
    distance_bands: tuple = ((1.5, 3.), (3., 6.), (3., 6.), (6., 12.), (12., 20.))
    heading_limits_deg: tuple = (180.,)*5  # physical alignment handles the initial yaw
    same_asphalt_cell_stages: int = 1
    stage_names: tuple = ('asfalto 1.5–3 m', 'terreni misti 3–6 m senza scale/ostacoli',
                         'tutti i terreni ammessi 3–6 m', 'tutti i terreni ammessi 6–12 m',
                         'tutti i terreni ammessi 12–20 m')
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


@dataclass(frozen=True)
class AlignmentSettings:
    tolerance_deg: float = 10.0
    stable_steps: int = 3
    gain: float = 1.5
    max_yaw_rad_s: float = 0.8
    max_seconds: float = 12.0
    max_steps_without_navigation: int = 300

ALIGN_CFG = AlignmentSettings()
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
STATE_LAYERS = (64, 32)
MAP_LAYERS = (1024, 512, 256, 128)
LAYERS = (128, 64)  # fusion trunk
RESOLUTION_M = 0.5
LOOK_X_M = tuple(i * 0.5 for i in range(-4, 5))
LOOK_Y_M = LOOK_X_M
MEMORY_RESOLUTION_M = 0.10
MEMORY_AGE_SECONDS = 90.0
DEPTH_STRIDE = 4
GEOMETRY_MIN_POINTS = 10
GEOMETRY_MIN_XY_EIGENVALUE = 1e-4  # square metres; rejects narrow/collinear support
BODY_CONTACT_NAME = 'body'  # sole non-leg rigid body with collision geometry in Spot USD
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
DEFAULT_PCA = HERE / 'artifacts' / 'pca32_big_map.npz'
PCA_DATASET = 'real_terrains_v2_seven_material_views_v1'
# Prefer already downloaded immutable local weights, avoiding network on the cluster.
LOCAL_DINO = HERE.parent / 'extracted_features_comparison_v2' / 'models' / 'dinov2'


def experiment_signature():
    return dict(schema=2, experiment='train_general_big_map_test5',
                pca_calibration=PCA_DATASET,
                algorithm='asymmetric_multi_head_PPO', navigation=asdict(NAV_CFG),
                ppo=asdict(PPO_CFG), actor_size=OBSERVATION_SIZE, critic_size=CRITIC_SIZE,
                layers=list(LAYERS), state_layers=list(STATE_LAYERS), map_layers=list(MAP_LAYERS),
                alignment=asdict(ALIGN_CFG), separate_actor_critic_optimizers=True,
                rollout_storage="navigation_only_packed_by_env", critic_state_size=39, feature_dim=FEATURE_DIM,
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
                fall_detection="physics_step_persistence_v1", body_contact_name=BODY_CONTACT_NAME,
                env_spacing_m=80.0, camera_far_clip_m=20.0, reset_pose="startup_settled_stance_above_local_footprint",
                startup_clearance_m=0.02)



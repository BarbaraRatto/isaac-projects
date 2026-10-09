"""Parameters for the first single-Spot navigation RL experiment."""

from dataclasses import dataclass

from rl_config import PHYSICS_DT


@dataclass(frozen=True)
class NavigationConfig:
    # One high-level velocity action lasts 0.2 s; the frozen joint policy still
    # receives observations at its own 20 Hz rate.
    physics_dt: float = PHYSICS_DT
    physics_steps_per_action: int = 40
    settle_seconds: float = 2.0
    max_episode_seconds: float = 12.0

    # The first task stays entirely on the existing asphalt cell. It tests the
    # training pipeline; terrain selection needs a visual observation later.
    terrain_name: str = "t1_asphalt"
    spawn_height_m: float = 0.75
    start_inset_m: float = 1.5
    goal_inset_m: float = 1.0
    goal_radius_m: float = 0.4
    cell_margin_m: float = 0.35
    minimum_body_height_m: float = 0.25

    max_forward_m_s: float = 0.9
    max_lateral_m_s: float = 0.45
    max_yaw_rad_s: float = 0.8

    # Reward per high-level step. Energy is mechanical work from measured
    # PhysX joint efforts, using the same absolute-power definition as step 2.
    progress_reward_per_m: float = 3.0
    energy_penalty_per_j: float = 0.003
    time_penalty_per_step: float = 0.02
    success_bonus: float = 8.0
    failure_penalty: float = 5.0

    @property
    def max_episode_steps(self) -> int:
        return round(self.max_episode_seconds / (self.physics_dt * self.physics_steps_per_action))


NAV_CFG = NavigationConfig()

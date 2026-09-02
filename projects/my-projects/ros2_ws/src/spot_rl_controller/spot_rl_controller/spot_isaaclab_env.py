#!/usr/bin/env python3
"""
spot_isaaclab_env.py

Ambiente RL per Spot in Isaac Lab (DirectRLEnv).

Questo è il file PRINCIPALE per il training: sostituisce spot_env.py
(approccio ROS2) con un ambiente Isaac Lab che gira tutto dentro Isaac Sim,
senza ROS2, su GPU, con N_ENVS robot in parallelo.

Vantaggi rispetto all'approccio ROS2:
  - 500-1000× più veloce (nessuna latenza ROS2, tutto su GPU)
  - N_ENVS robot in parallelo (default: 1024)
  - Reset automatico degli episodi
  - Calcolo energia da joint torques × velocità (stessa fisica, su GPU)

Come funziona la gridmap durante il training:
  Non usa DINOv2 (troppo costoso per 1024 robot in parallelo).
  Usa invece una MAPPA DI COSTO ENERGETICO SINTETICA, costruita
  da:
    1. Tipo di terreno (tile) sotto ogni robot
    2. Coefficiente di costo associato a quel tipo (da letteratura/fisica)
    → mappa 50×50 di valori in [0, 1]

  All'inference sul robot reale, la stessa mappa viene fornita da
  energy_costmap_node.py (che la deriva da DINOv2 + consumi reali).
  Il modello vede lo stesso formato in entrambi i casi.

Observation space (flat vector 2510-d):
  [0:2500]    gridmap energetica 50×50 flatten  — in [0,1]
  [2500:2504] goal_vec (dx, dy, cos_dtheta, sin_dtheta)
  [2504:2507] energy_vec (P_inst_norm, P_avg_norm, E_cum_norm)
  [2507:2510] robot_vel (vx_norm, vy_norm, omega_norm)

Action space:
  (vx_norm, vy_norm, omega_norm) ∈ [-1, 1]³

Uso:
  # Avvio con Isaac Lab (dentro il Python di Isaac Sim):
  ./isaaclab.sh -p spot_rl_controller/train_rl_isaaclab.py --num_envs 1024

  # Smoke test con pochi env:
  ./isaaclab.sh -p spot_rl_controller/train_rl_isaaclab.py --num_envs 4 --headless
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import torch
import numpy as np

# ─── Isaac Lab imports ────────────────────────────────────────────────────────
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen


# ═════════════════════════════════════════════════════════════════════════════
# Costanti: coefficienti di costo energetico per tipo di terreno (tile)
#
# Derivati da:
#   - friction: bassa friction → più scivolamento → più energia
#   - slope: inclinazione → componente gravitazionale
#   - irregolarità: impatti → dissipazione di energia
#
# Ordine: corrisponde all'indice di riga nel TerrainGeneratorCfg
#         (0=flat_asphalt, 1=slippery_flat, ..., 7=wave_hills)
# ═════════════════════════════════════════════════════════════════════════════
TERRAIN_ENERGY_COST = torch.tensor([
    0.10,   # 0 — flat_asphalt:          baseline (attrito alto, pianeggiante)
    0.40,   # 1 — slippery_flat:         friction 0.2→0.3, molta energia per controllo
    0.45,   # 2 — smooth_ramp:           7.5° slope, componente gravitazionale
    0.70,   # 3 — stairs:                impatti ripetuti, step 0-2 cm
    0.45,   # 4 — fine_gravel:           irregolarità fine (0-2.5 mm)
    0.60,   # 5 — large_stones:          irregolarità grande (0-5 mm)
    0.80,   # 6 — discrete_obstacles:    ostacoli 1-2.5 cm, alta dissipazione
    0.55,   # 7 — wave_hills:            onde 0-5 cm, sforzo variabile
], dtype=torch.float32)

# Dimensioni terreno (da create_spot_terrains.py)
TILE_SIZE_X = 2.0   # [m] dimensione di ogni tile in X
TILE_SIZE_Y = 2.0   # [m] dimensione di ogni tile in Y
N_TERRAIN_ROWS = 8  # tipi di terreno (righe della scacchiera)
N_TERRAIN_COLS = 8  # numero di istanze per tipo (colonne)

# Parametri energetici del modello
P_BASE = 50.0       # [W] potenza base (solo per stare in piedi)
P_SCALE = 300.0     # [W] potenza massima stimata durante il movimento
P_MAX_NORM = 200.0  # [W] valore di normalizzazione per la reward


# ═════════════════════════════════════════════════════════════════════════════
# Configurazione del terreno (identica a create_spot_terrains.py)
# ═════════════════════════════════════════════════════════════════════════════
material_normal = RigidBodyMaterialCfg(
    static_friction=1.5,
    dynamic_friction=1.2,
    restitution=0.0,
)
material_slippery = RigidBodyMaterialCfg(
    static_friction=0.3,
    dynamic_friction=0.2,
    restitution=0.0,
)

_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(TILE_SIZE_X, TILE_SIZE_Y),
    border_width=3.0,
    num_rows=N_TERRAIN_ROWS,
    num_cols=N_TERRAIN_COLS,
    horizontal_scale=0.02,
    vertical_scale=0.001,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "1_flat_asphalt":       terrain_gen.MeshPlaneTerrainCfg(proportion=1.0/8.0),
        "2_slippery_flat":      terrain_gen.MeshPlaneTerrainCfg(proportion=1.0/8.0),
        "3_smooth_ramp":        terrain_gen.HfPyramidSlopedTerrainCfg(
                                    proportion=1.0/8.0,
                                    slope_range=(0.0, math.tan(math.radians(7.5))),
                                    platform_width=1.0, border_width=0.25),
        "4_stairs":             terrain_gen.MeshPyramidStairsTerrainCfg(
                                    proportion=1.0/8.0, step_height_range=(0.0, 0.02),
                                    step_width=0.15, platform_width=1.0, border_width=0.25),
        "5_fine_gravel":        terrain_gen.HfRandomUniformTerrainCfg(
                                    proportion=1.0/8.0, noise_range=(0.0, 0.0025),
                                    noise_step=0.01, border_width=0.25),
        "6_large_stones":       terrain_gen.HfRandomUniformTerrainCfg(
                                    proportion=1.0/8.0, noise_range=(0.0, 0.005),
                                    noise_step=0.02, border_width=0.25),
        "7_discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                                    proportion=1.0/8.0, obstacle_height_mode="fixed",
                                    obstacle_height_range=(0.01, 0.025),
                                    obstacle_width_range=(0.1, 0.4), num_obstacles=40,
                                    platform_width=0.2, border_width=0.25),
        "8_wave_hills":         terrain_gen.HfWaveTerrainCfg(
                                    proportion=1.0/8.0, amplitude_range=(0.0, 0.05),
                                    num_waves=4, border_width=0.25),
    },
)


# ═════════════════════════════════════════════════════════════════════════════
# Configurazione robot Spot
# Usa il file USD del robot già nella simulazione
# ═════════════════════════════════════════════════════════════════════════════
def _make_spot_cfg(usd_path: str) -> ArticulationCfg:
    """
    Costruisce la configurazione di Spot partendo dall'USD esistente.

    Usa ImplicitActuatorCfg: lascia la fisica di Isaac Sim gestire i giunti,
    senza dover specificare manualmente PD gains per ogni giunto.
    Adeguato per un controllore a livello di velocità del corpo base.
    """
    from isaaclab.actuators import ImplicitActuatorCfg
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=20.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.65),   # altezza iniziale tipica di Spot
            joint_pos={
                ".*_hx": 0.0,
                ".*_hy": 0.0,
                ".*_kn": -1.5,  # Ginocchia piegate (il limite è -2.793, -0.247)
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Controlla tutti i giunti in modo implicito
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=40.0,
                damping=1.0,
            ),
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# Configurazione Environment (dataclass)
# ═════════════════════════════════════════════════════════════════════════════
@configclass
class SpotEnergyEnvCfg(DirectRLEnvCfg):
    """Configurazione per SpotEnergyEnv."""

    # ── Simulazione ────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,          # 60 Hz fisica
        render_interval=4,      # render ogni 4 step (15 Hz visivo)
    )

    # ── Scena ──────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=20.0,       # distanza tra istanze parallele [m]
        replicate_physics=True,
    )

    # ── Terreno ────────────────────────────────────────────────────────────
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=_TERRAIN_CFG,
        collision_group=-1,
        physics_material=material_normal,
    )

    # ── Robot ──────────────────────────────────────────────────────────────
    # Il percorso USD viene passato a runtime dallo script di training
    robot_usd_path: str = MISSING

    # ── Spazio azioni e osservazioni ───────────────────────────────────────
    # 2510 = 50×50 (gridmap) + 4 (goal) + 3 (energia) + 3 (velocità)
    observation_space: int = 2510
    action_space: int = 3          # (vx_norm, vy_norm, omega_norm)
    state_space: int = 0           # non usiamo asymmetric actor-critic

    # ── Parametri gridmap locale ───────────────────────────────────────────
    gridmap_rows: int   = 50       # celle in X (avanti/dietro)
    gridmap_cols: int   = 50       # celle in Y (sinistra/destra)
    gridmap_res: float  = 0.1      # [m] risoluzione per cella

    # ── Parametri azione ───────────────────────────────────────────────────
    vx_max:    float = 0.5         # [m/s]
    vy_max:    float = 0.3         # [m/s]
    omega_max: float = 1.0         # [rad/s]

    # ── Parametri reward ───────────────────────────────────────────────────
    lambda_energy: float = 1.0   # peso penalità energetica
    lambda_time:   float = 0.005   # penalità per timestep
    goal_bonus:    float = 10.0    # bonus raggiungimento goal
    stuck_penalty: float = 5.0     # penalità stuck

    # ── Parametri episodio ─────────────────────────────────────────────────
    episode_length_s:     float = 60.0   # [s] durata massima episodio
    decimation:           int   = 4      # policy_dt = sim_dt * decimation
    goal_tolerance_m:     float = 0.15   # [m] distanza per goal raggiunto
    stuck_timeout_steps:  int   = 50     # step senza progresso → stuck
    stuck_min_progress_m: float = 0.05   # [m] progresso minimo per non stuck


# ═════════════════════════════════════════════════════════════════════════════
# Environment principale
# ═════════════════════════════════════════════════════════════════════════════
class SpotEnergyEnv(DirectRLEnv):
    """
    Ambiente RL per navigazione energetically-aware di Spot.

    Architettura:
      - N_ENVS robot Spot in parallelo su una griglia di terreni misti
      - Azione: (vx, vy, omega) applicata direttamente al corpo base
      - Reward: progresso verso goal − costo energetico − penalità temporale
      - Osservazione: gridmap energetica 50×50 + goal + energia + velocità
    """

    cfg: SpotEnergyEnvCfg

    def __init__(self, cfg: SpotEnergyEnvCfg, render_mode=None) -> None:
        super().__init__(cfg, render_mode=render_mode)

        # ── Calcolo max_steps dall'episodio in secondi ─────────────────────
        self._max_steps = int(
            cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)
        )
        self._step_count = torch.zeros(self.num_envs, device=self.device)

        # ── Distanza precedente al goal (per calcolare il progresso) ───────
        self._prev_dist  = torch.zeros(self.num_envs, device=self.device)
        self._best_dist  = torch.zeros(self.num_envs, device=self.device)
        self._stuck_ctr  = torch.zeros(self.num_envs, dtype=torch.int32,
                                       device=self.device)

        # ── Goal per ogni env (x, y) nel frame world ──────────────────────
        self._goals = torch.zeros(self.num_envs, 2, device=self.device)

        # ── Energia cumulativa per episodio ────────────────────────────────
        self._ep_energy = torch.zeros(self.num_envs, device=self.device)
        self._ep_power  = torch.zeros(self.num_envs, device=self.device)

        # ── Tabella di costo energetico per tile ───────────────────────────
        self._terrain_cost = TERRAIN_ENERGY_COST.to(self.device)

        # ── Goal candidates (offset rispetto all'origine di ogni env) ──────
        # Posizioni lontane per forzare l'attraversamento di più terreni (strisce di 6m)
        self._goal_offsets = torch.tensor([
            [12.0,  0.0], [12.0,  4.0], [12.0, -4.0],
            [18.0,  0.0], [18.0,  5.0], [18.0, -5.0],
            [22.0,  3.0], [22.0, -3.0], [24.0,  0.0],
            [15.0,  2.0], [15.0, -2.0], [20.0,  2.0],
        ], device=self.device)  # (N_GOALS, 2)

    # ======================================================================
    # Setup scena
    # ======================================================================
    def _setup_scene(self) -> None:
        """Costruisce la scena: terreno + N_ENVS istanze del robot."""
        # ── Terreno ───────────────────────────────────────────────────────
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # ── Robot ─────────────────────────────────────────────────────────
        robot_cfg = _make_spot_cfg(self.cfg.robot_usd_path)
        self._robot = Articulation(robot_cfg)
        
        # ── Clone environments ────────────────────────────────────────────
        self.scene.clone_environments(copy_from_source=False)
        
        # ── Filter collisions for CPU (if needed) ─────────────────────────
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
            
        self.scene.articulations["robot"] = self._robot

        # ── Luce ──────────────────────────────────────────────────────────
        light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ======================================================================
    # Pre-physics step: applica le azioni
    # ======================================================================
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Riceve le azioni (N_ENVS, 3) normalizzate in [-1, 1] e le converte
        in velocità del corpo base.
        """
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """
        Applica la velocità al corpo base di Spot.

        In training usiamo velocity control diretto (più stabile per
        l'esplorazione RL). Sul robot reale, il controllore Spot SDK
        gestisce la locomozione — l'RL produce solo la velocità di riferimento.
        """
        vx    = self._actions[:, 0] * self.cfg.vx_max     # (N_ENVS,)
        vy    = self._actions[:, 1] * self.cfg.vy_max
        omega = self._actions[:, 2] * self.cfg.omega_max

        # Velocità lineare in frame world (ora yaw è sempre 0, quindi coincide)
        vx_world = vx
        vy_world = vy

        # Applica velocità al root dell'articolazione (Z è bloccata a 0, nessuna rotazione)
        root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        root_vel[:, 0] = vx_world
        root_vel[:, 1] = vy_world
        root_vel[:, 2] = 0.0  # Impedisce di cadere
        # root_vel[:, 3:6] è già 0, quindi nessuna velocità angolare (omega ignorato)
        self._robot.write_root_velocity_to_sim(root_vel)
        
        # Effetto "Zombie Perfetto": forziamo altezza a 0.75 e rotazione a 0 assoluto
        current_pos = self._robot.data.root_pos_w.clone()
        current_pos[:, 2] = 0.75
        
        fixed_quat = torch.zeros((self.num_envs, 4), device=self.device)
        fixed_quat[:, 0] = 1.0  # Quaternione identità [1, 0, 0, 0] (no pitch, no roll, no yaw)
        
        self._robot.write_root_pose_to_sim(torch.cat([current_pos, fixed_quat], dim=-1))

    # ======================================================================
    # Osservazioni
    # ======================================================================
    def _get_observations(self) -> dict:
        """
        Costruisce il vettore di osservazione (N_ENVS, 2510).

        Ritorna {"policy": tensor(N_ENVS, 2510)}.
        """
        pos    = self._robot.data.root_pos_w        # (N_ENVS, 3)
        vel    = self._robot.data.root_lin_vel_b    # (N_ENVS, 3) frame base
        ang_vel= self._robot.data.root_ang_vel_b    # (N_ENVS, 3)

        # ── Gridmap energetica 50×50 ───────────────────────────────────────
        gridmap = self._compute_energy_gridmap(pos)  # (N_ENVS, 2500)

        # ── Vettore goal ──────────────────────────────────────────────────
        yaw = self._get_robot_yaw()  # (N_ENVS,)
        dx  = self._goals[:, 0] - pos[:, 0]
        dy  = self._goals[:, 1] - pos[:, 1]
        dtheta = torch.zeros_like(yaw)   # goal senza orientamento preferenziale
        goal_vec = torch.stack([dx, dy, torch.cos(dtheta), torch.sin(dtheta)], dim=-1)
        # (N_ENVS, 4)

        # ── Vettore energetico ────────────────────────────────────────────
        p_norm = (self._ep_power / P_MAX_NORM).clamp(0.0, 1.0)
        energy_vec = torch.stack([p_norm, p_norm, torch.zeros_like(p_norm)], dim=-1)
        # (N_ENVS, 3) — P_inst_norm, P_avg_norm, E_cum_norm (semplificato)

        # ── Velocità robot normalizzata ────────────────────────────────────
        robot_vel = torch.stack([
            (vel[:, 0]     / self.cfg.vx_max).clamp(-1.0, 1.0),
            (vel[:, 1]     / self.cfg.vy_max).clamp(-1.0, 1.0),
            (ang_vel[:, 2] / self.cfg.omega_max).clamp(-1.0, 1.0),
        ], dim=-1)  # (N_ENVS, 3)

        # ── Concatenazione finale ─────────────────────────────────────────
        obs = torch.cat([gridmap, goal_vec, energy_vec, robot_vel], dim=-1)
        # (N_ENVS, 2510)

        return {"policy": obs}

    # ======================================================================
    # Reward
    # ======================================================================
    def _get_rewards(self) -> torch.Tensor:
        """
        r(t) = r_progress - lambda_energy × P_norm - lambda_time

        Tutti i termini su tensori GPU (N_ENVS,).
        """
        pos  = self._robot.data.root_pos_w  # (N_ENVS, 3)
        dist = torch.norm(self._goals - pos[:, :2], dim=-1)  # (N_ENVS,)

        # Progresso verso il goal
        progress   = self._prev_dist - dist
        r_progress = progress / max(self.cfg.goal_tolerance_m, 0.01)

        # Energia stimata: P_base + P_move × terrain_cost × ||vel||
        vel_mag = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=-1)
        terrain_cost = self._get_terrain_cost(pos)
        power = P_BASE + P_SCALE * terrain_cost * vel_mag
        self._ep_power = power
        self._ep_energy += power * self.cfg.sim.dt

        r_energy = self.cfg.lambda_energy * (power / P_MAX_NORM)

        # Penalità temporale fissa
        r_time = self.cfg.lambda_time
        
        # Penalità se stuck
        stuck = self._stuck_ctr >= self.cfg.stuck_timeout_steps
        r_stuck = torch.where(stuck, self.cfg.stuck_penalty, 0.0)
        
        # Bonus traguardo
        goal_reached = dist < self.cfg.goal_tolerance_m
        r_goal = torch.where(goal_reached, 50.0, 0.0)

        reward = r_progress - r_energy - r_time - r_stuck + r_goal
        #reward =  - r_energy - r_stuck + r_goal
        

        # Aggiornamento distanza precedente e stuck detection
        improved = dist < (self._best_dist - self.cfg.stuck_min_progress_m)
        self._best_dist = torch.where(improved, dist, self._best_dist)
        self._stuck_ctr = torch.where(improved,
                                      torch.zeros_like(self._stuck_ctr),
                                      self._stuck_ctr + 1)
        self._prev_dist = dist
        self._step_count += 1

        return reward

    # ======================================================================
    # Done conditions
    # ======================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        terminated: goal raggiunto o stuck irrecuperabile
        truncated:  timeout (max_episode_length_s superato)
        """
        pos  = self._robot.data.root_pos_w
        dist = torch.norm(self._goals - pos[:, :2], dim=-1)

        goal_reached = dist < self.cfg.goal_tolerance_m
        stuck        = self._stuck_ctr >= self.cfg.stuck_timeout_steps
        terminated   = goal_reached | stuck

        # Robot caduto (z molto basso)
        fallen     = pos[:, 2] < 0.2
        terminated = terminated | fallen

        truncated = self._step_count >= self._max_steps

        return terminated, truncated

    # ======================================================================
    # Reset
    # ======================================================================
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        Resetta gli env specificati: posizione robot + nuovo goal.
        Isaac Lab chiama questo metodo automaticamente quando un episodio termina.
        """
        if len(env_ids) == 0:
            return

        # ── Posizione di spawn casuale sul terreno ────────────────────────
        env_origins = self._terrain.env_origins[env_ids]  # (N_reset, 3)
        n = len(env_ids)

        # Spawn casuale entro il tile (evitando i bordi: ±1m)
        spawn_offset = torch.zeros(n, 3, device=self.device)
        spawn_offset[:, 0] = torch.FloatTensor(n).uniform_(-1.5, 1.5).to(self.device)
        spawn_offset[:, 1] = torch.FloatTensor(n).uniform_(-0.5, 0.5).to(self.device)
        spawn_offset[:, 2] = 0.65  # altezza iniziale Spot

        spawn_pos = env_origins + spawn_offset

        # Orientamento casuale (yaw uniforme)
        yaw = torch.FloatTensor(n).uniform_(-math.pi, math.pi).to(self.device)
        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = torch.cos(yaw / 2)   # w
        quat[:, 3] = torch.sin(yaw / 2)   # z

        # Reset stato root
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3]  = spawn_pos
        root_state[:, 3:7] = quat
        root_state[:, 7:]  = 0.0   # azzera velocità
        self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        # Reset giunti
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # ── Scelta nuovo goal ─────────────────────────────────────────────
        # Goal casuale da lista predefinita di offset rispetto all'origine env
        goal_idx = torch.randint(0, len(self._goal_offsets), (n,), device=self.device)
        self._goals[env_ids] = (
            env_origins[:, :2] + self._goal_offsets[goal_idx]
        )

        # ── Reset contatori episodio ──────────────────────────────────────
        self._step_count[env_ids] = 0
        self._ep_energy[env_ids]  = 0.0
        self._ep_power[env_ids]   = 0.0
        self._stuck_ctr[env_ids]  = 0

        # Distanza iniziale al goal
        dist_init = torch.norm(
            self._goals[env_ids] - spawn_pos[:, :2], dim=-1
        )
        self._prev_dist[env_ids] = dist_init
        self._best_dist[env_ids] = dist_init

    # ======================================================================
    # Helper: gridmap energetica sintetica
    # ======================================================================
    def _compute_energy_gridmap(self, robot_pos: torch.Tensor) -> torch.Tensor:
        """
        Costruisce la gridmap 50×50 di costo energetico per ogni robot.

        Per ogni cella della griglia locale attorno al robot:
          1. Calcola la posizione world della cella
          2. Determina a quale tile appartiene (riga = tipo terreno)
          3. Assegna il coefficiente di costo energetico

        Ritorna: tensor (N_ENVS, 2500) in [0, 1].
        """
        n      = self.num_envs
        rows   = self.cfg.gridmap_rows   # 50
        cols   = self.cfg.gridmap_cols   # 50
        res    = self.cfg.gridmap_res    # 0.1 m

        # Offset della griglia rispetto al robot (centrata sul robot)
        half_r = rows * res / 2.0  # 2.5 m
        half_c = cols * res / 2.0  # 2.5 m

        # Crea offset per tutte le celle: (rows, cols, 2)
        r_off = torch.linspace(-half_r + res/2, half_r - res/2, rows,
                               device=self.device)
        c_off = torch.linspace(-half_c + res/2, half_c - res/2, cols,
                               device=self.device)
        # Griglia di offset (rows × cols, 2)
        grid_x, grid_y = torch.meshgrid(r_off, c_off, indexing="ij")
        cell_offsets = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        # (rows*cols, 2)

        # Posizione world di ogni cella per ogni robot:
        # (N_ENVS, 1, 2) + (1, rows*cols, 2) → (N_ENVS, rows*cols, 2)
        cell_world = (robot_pos[:, :2].unsqueeze(1) +
                      cell_offsets.unsqueeze(0))

        # Determina il tipo di terreno per ogni cella usando la logica a scacchiera:
        # tile_idx = (tile_x + tile_y) % N_TERRAIN_ROWS
        tile_x = torch.floor(cell_world[:, :, 0] / TILE_SIZE_X).long()
        tile_y = torch.floor(cell_world[:, :, 1] / TILE_SIZE_Y).long()
        
        # Campo Minato
        is_obstacle = ((tile_x % 2) == 1) & ((tile_y % 2) == 0)
        tile_idx = torch.where(is_obstacle, 6, 0).clamp(0, N_TERRAIN_ROWS - 1)
        # (N_ENVS, rows*cols)

        # Costo energetico per ogni cella
        cost = self._terrain_cost[tile_idx]  # (N_ENVS, rows*cols)

        return cost  # (N_ENVS, 2500) in [0, 1]

    # ======================================================================
    # Helper: costo energetico della posizione corrente del robot
    # ======================================================================
    def _get_terrain_cost(self, pos: torch.Tensor) -> torch.Tensor:
        """Costo del tile sotto il robot corrente. (N_ENVS,)"""
        tile_x = torch.floor(pos[:, 0] / TILE_SIZE_X).long()
        tile_y = torch.floor(pos[:, 1] / TILE_SIZE_Y).long()
        
        # Campo Minato (Ostacoli isolati e autostrade di asfalto)
        # Se x dispari e y pari -> Rocce (indice 6), altrimenti -> Asfalto (indice 0)
        is_obstacle = ((tile_x % 2) == 1) & ((tile_y % 2) == 0)
        tile_idx = torch.where(is_obstacle, 6, 0).clamp(0, N_TERRAIN_ROWS - 1)
        
        return self._terrain_cost[tile_idx]

    # ======================================================================
    # Helper: yaw del robot
    # ======================================================================
    def _get_robot_yaw(self) -> torch.Tensor:
        """Estrae il yaw dal quaternione del robot. (N_ENVS,)"""
        q = self._robot.data.root_quat_w  # (N_ENVS, 4) = (w, x, y, z)
        siny = 2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2])
        cosy = 1.0 - 2.0 * (q[:, 2]*q[:, 2] + q[:, 3]*q[:, 3])
        return torch.atan2(siny, cosy)

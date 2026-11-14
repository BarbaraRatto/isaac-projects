"""Random endpoint sampling and evaluation-driven curriculum; no route planner."""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import math
import torch
from train_general_big_map_test9_SAC.config import MISSION_CFG, NAV_CFG, TERRAIN_NAMES

BANK = Path(__file__).resolve().parent/'artifacts/spawn_bank.npz'


def place_missions(root, origins, clearance, missions):
    """Batch device placement; keep the existing NumPy sampler/RNG and missions.

    Upload coordinates and yaw quaternions in ONE packed transfer, instead of
    constructing many small CUDA tensors/scalar assignments for each robot.
    """
    rows = [[*m['start'][:4], *m['goal'][:4], math.cos(m['yaw']/2), math.sin(m['yaw']/2),
             m['deadline_steps']] for m in missions]
    if not rows:
        return root, origins.clone(), torch.empty(0, dtype=torch.long, device=root.device)
    data = torch.tensor(rows, dtype=root.dtype, device=root.device)
    root[:, :2] = origins[:, :2]+data[:, :2]
    root[:, 2] = origins[:, 2]+data[:, 3]+clearance+.02
    root[:, 3:7] = 0
    root[:, 3], root[:, 6] = data[:, 8], data[:, 9]
    root[:, 7:] = 0
    goals = origins.clone()
    goals[:, :2] += data[:, 4:6]
    goals[:, 2] += data[:, 7]
    return root, goals, data[:, 10].long()

@dataclass
class Curriculum:
    level: int = 0
    streak: int = 0

    def update(self, success_rate):
        if self.level >= len(MISSION_CFG.distance_bands)-1:
            return False
        self.streak = self.streak+1 if success_rate >= MISSION_CFG.promotion_success else 0
        if self.streak >= MISSION_CFG.promotion_evaluations:
            self.level += 1
            self.streak = 0
            return True
        return False

    def state_dict(self):
        return asdict(self)

    def load_state_dict(self, state):
        self.level, self.streak = int(state['level']), int(state['streak'])
        if not 0 <= self.level < len(MISSION_CFG.distance_bands) or not 0 <= self.streak < MISSION_CFG.promotion_evaluations:
            raise ValueError('Invalid curriculum state')


def deadline_steps(distance):
    seconds = np.clip(MISSION_CFG.deadline_base_s + MISSION_CFG.deadline_seconds_per_m*distance,
                      MISSION_CFG.deadline_min_s, MISSION_CFG.deadline_max_s)
    return int(np.ceil(seconds/NAV_CFG.action_dt))

class MissionSampler:
    def __init__(self, path=BANK):
        with np.load(path, allow_pickle=False) as bank:
            self.points = bank['points'].copy()
            self.evaluation = bank['evaluation'].copy()
        self.materials = self.points[:, 4].astype(int)
        self.spawn_allowed = ~np.isin(self.materials, [TERRAIN_NAMES.index(x) for x in MISSION_CFG.spawn_excluded])
        self.goal_allowed = ~np.isin(self.materials, [TERRAIN_NAMES.index(x) for x in MISSION_CFG.goal_excluded])

    def sample(self, band, rng=np.random, evaluation=False):
        lo, hi = MISSION_CFG.distance_bands[band]
        split = self.evaluation == evaluation
        starts = np.flatnonzero(split & self.spawn_allowed)
        goals = np.flatnonzero(split & self.goal_allowed)
        if band < MISSION_CFG.same_asphalt_cell_stages:
            starts = starts[self.materials[starts] == TERRAIN_NAMES.index('t1_asphalt')]
            goals = goals[self.materials[goals] == TERRAIN_NAMES.index('t1_asphalt')]
        elif band == 1:
            excluded = [TERRAIN_NAMES.index(x) for x in ('t6_obstacles', 't7_stairs')]
            starts = starts[~np.isin(self.materials[starts], excluded)]
            goals = goals[~np.isin(self.materials[goals], excluded)]
        types = np.unique(self.materials[starts])
        for _ in range(1000):
            material = rng.choice(types)
            a = int(rng.choice(starts[self.materials[starts]==material]))
            distance = np.linalg.norm(self.points[goals, :2]-self.points[a, :2], axis=1)
            eligible = (distance>=lo)&(distance<=hi)
            if band < MISSION_CFG.same_asphalt_cell_stages:
                eligible &= np.all(self.points[goals, 5:7] == self.points[a, 5:7], axis=1)
            valid = goals[eligible]
            if not len(valid): continue
            b = int(rng.choice(valid))
            d = float(np.linalg.norm(self.points[a, :2]-self.points[b, :2]))
            bearing = np.arctan2(self.points[b, 1]-self.points[a, 1], self.points[b, 0]-self.points[a, 0])
            limit = np.deg2rad(MISSION_CFG.heading_limits_deg[band])
            yaw = (bearing + rng.uniform(-limit, limit) + np.pi) % (2*np.pi) - np.pi
            return dict(start_id=a, goal_id=b, start=self.points[a].tolist(), goal=self.points[b].tolist(),
                        yaw=float(yaw), band=int(band),
                        distance_m=d, deadline_steps=deadline_steps(d))
        raise RuntimeError(f'No endpoint pair in band {band}; no reachability filtering was applied')

    def training(self, level):
        band = level
        if level and np.random.random() < MISSION_CFG.easier_probability:
            band = int(np.random.randint(level))
        return self.sample(band)

    def evaluation_mission(self, env_id, episode_index, seed, stage=0):
        # Independent of other robots' episode lengths and resets.
        band = int(stage)
        rng = np.random.default_rng(np.random.SeedSequence([seed, band, env_id, episode_index]))
        return self.sample(band, rng, evaluation=True)

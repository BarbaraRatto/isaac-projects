"""Medium-distance task using the existing Spot, ZED and DINO simulation."""

from pathlib import Path

import torch

from full_grid_energy_choice.env import FullGridChoiceEnv
from full_grid_energy_choice.config import FullGridGeometry
from full_grid_energy_choice.scene import FullGridChoiceCfg
from medium_distance_choice.task_config import (
    GEOMETRY_MEDIUM, NAV_CFG_MEDIUM, TERRAIN_LABELS, read_terrain_codes,
)


class MediumDistanceEnv(FullGridChoiceEnv):
    def __init__(self, cfg: FullGridChoiceCfg, atlas_file: Path | None,
                 dino_batch_size: int = 4, geometry: FullGridGeometry | None = None,
                 **kwargs):
        super().__init__(cfg, atlas_file, dino_batch_size=dino_batch_size,
                         geometry=geometry or GEOMETRY_MEDIUM, nav_cfg=NAV_CFG_MEDIUM,
                         **kwargs)
        self.terrain_codes = torch.tensor(read_terrain_codes(), dtype=torch.long,
                                          device=self.device)
        self.terrain_entries = torch.zeros((self.num_envs, len(TERRAIN_LABELS)),
                                           dtype=torch.bool, device=self.device)

    def _track_terrain(self) -> None:
        # This map is used only to log where Spot went. It is never an observation
        # or a reward input; the policy sees live camera features instead.
        feet = (self.robot.data.body_pos_w[:, self.foot_ids, :2]
                - self.scene.env_origins[:, None, :2])
        col = torch.floor((feet[..., 0] + 3.0) / 6.0).long()
        row = torch.floor((feet[..., 1] + 1.5) / 3.0).long()
        inside = (row >= 0) & (row < 7) & (col >= 0) & (col < 7)
        codes = self.terrain_codes[row.clamp(0, 6), col.clamp(0, 6)]
        codes = torch.where(inside, codes, -1)
        for code in range(len(TERRAIN_LABELS)):
            self.terrain_entries[:, code] |= (codes == code).any(dim=1)

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        # Isaac Lab resets completed environments before returning from step().
        # Keep the pre-reset body position in info, including the terminal step.
        body_xy = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        self.extras["body_x_m"] = body_xy[:, 0].clone()
        self.extras["body_y_m"] = body_xy[:, 1].clone()
        for code, label in enumerate(TERRAIN_LABELS):
            self.extras[f"{label}_entry"] = self.terrain_entries[:, code].clone()
        return reward

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        self.terrain_entries[env_ids] = False

"""Spot navigation on one selected 3x3 binary-choice board."""

from pathlib import Path

import torch
from isaaclab.assets import Articulation
from isaaclab.sensors import TiledCamera

from binary_choice.config import BinaryCase, NAV_CFG_BINARY, TERRAIN_LABELS
from binary_choice.scene import BinaryChoiceCfg, create_binary_terrain
from binary_choice.terrain import read_terrain_codes
from binary_choice.vision import BinaryBatchedDino
from full_grid_energy_choice.env import FullGridChoiceEnv


class BinaryChoiceEnv(FullGridChoiceEnv):
    def __init__(self, cfg: BinaryChoiceCfg, case: BinaryCase, atlas_file: Path | None,
                 dino_batch_size: int = 4, **kwargs):
        self.case = case
        super().__init__(cfg, atlas_file, dino_batch_size=dino_batch_size,
                         geometry=case.geometry, nav_cfg=NAV_CFG_BINARY,
                         vision_type=BinaryBatchedDino, **kwargs)
        self.terrain_codes = torch.tensor(read_terrain_codes(case), dtype=torch.long,
                                          device=self.device)
        self.terrain_entries = torch.zeros((self.num_envs, len(TERRAIN_LABELS)),
                                           dtype=torch.bool, device=self.device)

    def _setup_scene(self) -> None:
        create_binary_terrain(self.case)
        self.robot = Articulation(self.cfg.robot_cfg)
        self.camera = None if self.measure_only else TiledCamera(self.cfg.camera_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["spot"] = self.robot
        if self.camera is not None:
            self.scene.sensors["zed_x"] = self.camera

    def _track_terrain(self) -> None:
        # Terrain names are recorded only for analysis, never given to PPO.
        feet = (self.robot.data.body_pos_w[:, self.foot_ids, :2]
                - self.scene.env_origins[:, None, :2])
        col = torch.floor((feet[..., 0] + 1.0) / 2.0).long()
        row = torch.floor((feet[..., 1] + 1.0) / 2.0).long()
        inside = (row >= 0) & (row < 3) & (col >= 0) & (col < 3)
        codes = self.terrain_codes[row.clamp(0, 2), col.clamp(0, 2)]
        codes = torch.where(inside, codes, -1)
        for code in range(len(TERRAIN_LABELS)):
            self.terrain_entries[:, code] |= (codes == code).any(dim=1)

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        body_xy = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        self.extras["body_x_m"] = body_xy[:, 0].clone()
        self.extras["body_y_m"] = body_xy[:, 1].clone()
        for code, label in enumerate(TERRAIN_LABELS):
            self.extras[f"{label}_entry"] = self.terrain_entries[:, code].clone()
        return reward

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        self.terrain_entries[env_ids] = False

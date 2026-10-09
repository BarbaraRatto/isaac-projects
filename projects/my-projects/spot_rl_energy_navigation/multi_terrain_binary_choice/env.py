"""One shared Spot policy across distinct terrain layouts and start/goal pairs."""

from pathlib import Path

import torch

from binary_choice.config import TERRAIN_LABELS
from binary_choice.env import BinaryChoiceEnv
from binary_choice.terrain import read_terrain_codes
from multi_terrain_binary_choice.layouts import Condition
from multi_terrain_binary_choice.scene import MultiChoiceCfg, setup_scene


class MultiChoiceEnv(BinaryChoiceEnv):
    cfg: MultiChoiceCfg

    def __init__(self, cfg: MultiChoiceCfg, conditions: tuple[Condition, ...],
                 atlas_file: Path, dino_batch_size: int = 4, **kwargs):
        if len(conditions) != cfg.scene.num_envs:
            raise ValueError("Every environment needs exactly one condition")
        if tuple(item.env_id for item in conditions) != tuple(range(len(conditions))):
            raise ValueError("Conditions must be ordered by env_id")
        self.conditions = conditions
        # All robots first settle on their gravel start cell. After settling,
        # the saved root poses and per-env goals are replaced before the first reset.
        super().__init__(cfg, conditions[0].case, atlas_file,
                         dino_batch_size=dino_batch_size, **kwargs)
        self.terrain_codes = torch.tensor(
            [read_terrain_codes(item.case) for item in conditions],
            dtype=torch.long, device=self.device,
        )
        starts = torch.tensor([item.start_xy for item in conditions], device=self.device)
        goals = torch.tensor([item.goal_xy for item in conditions], device=self.device)
        self.settled_root[:, :2] = self.scene.env_origins[:, :2] + starts
        yaw = torch.atan2(goals[:, 1] - starts[:, 1], goals[:, 0] - starts[:, 0])
        self.settled_root[:, 3] = torch.cos(yaw / 2)
        self.settled_root[:, 4:6] = 0.0
        self.settled_root[:, 6] = torch.sin(yaw / 2)
        self.goal_w[:, :2] = self.scene.env_origins[:, :2] + goals
        self.goal_w[:, 2] = self.scene.env_origins[:, 2]

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _track_terrain(self) -> None:
        feet = (self.robot.data.body_pos_w[:, self.foot_ids, :2]
                - self.scene.env_origins[:, None, :2])
        col = torch.floor((feet[..., 0] + 1.0) / 2.0).long()
        row = torch.floor((feet[..., 1] + 1.0) / 2.0).long()
        inside = (row >= 0) & (row < 3) & (col >= 0) & (col < 3)
        env_id = torch.arange(self.num_envs, device=self.device)[:, None]
        codes = self.terrain_codes[env_id, row.clamp(0, 2), col.clamp(0, 2)]
        codes = torch.where(inside, codes, -1)
        for code in range(len(TERRAIN_LABELS)):
            self.terrain_entries[:, code] |= (codes == code).any(dim=1)

"""Per-robot startup state machine; no simulator import and no learned actions."""
import math
import torch
from train_general_big_map_test5.config import ALIGN_CFG, NAV_CFG


def heading_error(goal_xy, base_xy, yaw):
    delta = goal_xy-base_xy
    angle = torch.atan2(delta[:, 1], delta[:, 0])-yaw
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class InitialAlignment:
    def __init__(self, count, device, settings=ALIGN_CFG):
        self.cfg = settings
        self.active = torch.ones(count, dtype=torch.bool, device=device)
        self.stable = torch.zeros(count, dtype=torch.long, device=device)
        self.steps = torch.zeros(count, dtype=torch.long, device=device)
        self.succeeded = torch.zeros(count, dtype=torch.bool, device=device)
        self.failed = torch.zeros(count, dtype=torch.bool, device=device)
        self.end_error = torch.zeros(count, device=device)

    def reset(self, ids):
        self.active[ids] = True
        for value in (self.stable, self.steps, self.succeeded, self.failed, self.end_error):
            value[ids] = 0

    def actions(self, policy_actions, error):
        result = policy_actions.clamp(-1, 1).clone()
        # (-1, yaw/max_yaw) maps to exactly zero forward speed.
        result[self.active, 0] = -1.
        yaw = (self.cfg.gain*error).clamp(-self.cfg.max_yaw_rad_s, self.cfg.max_yaw_rad_s)
        result[self.active, 1] = yaw[self.active]/NAV_CFG.max_yaw_rad_s
        return result

    def advance(self, error):
        active = self.active.clone()
        self.steps += active.long()
        within = error.abs() <= math.radians(self.cfg.tolerance_deg)
        self.stable[active] = torch.where(within[active], self.stable[active]+1, 0)
        ready = active & (self.stable >= self.cfg.stable_steps)
        # A successful third stable sample at the deadline wins over timeout.
        expired = active & ~ready & (self.steps >= round(self.cfg.max_seconds/NAV_CFG.action_dt))
        self.succeeded |= ready
        self.failed |= expired
        finished = ready | expired
        self.end_error[finished] = error[finished]
        self.active[finished] = False
        return ready

"""Navigation reward terms independent of simulator imports."""
import math

import torch


def orientation_penalty(error_rad, navigation_mask, dt, start_deg=110.0, per_second=2.0):
    """Zero through start_deg, linearly reaching -per_second * dt at 180°.

    The mask describes who controlled THIS step, including the final alignment
    step. Wrapped absolute errors make left/right turns equivalent.
    """
    if not 0.0 <= start_deg < 180.0 or dt <= 0.0 or per_second < 0.0:
        raise ValueError('Invalid orientation penalty parameters')
    error = torch.atan2(torch.sin(error_rad), torch.cos(error_rad)).abs()
    start = math.radians(start_deg)
    fraction = ((error-start)/(math.pi-start)).clamp(0.0, 1.0)
    return -per_second*dt*fraction*navigation_mask.to(error.dtype)

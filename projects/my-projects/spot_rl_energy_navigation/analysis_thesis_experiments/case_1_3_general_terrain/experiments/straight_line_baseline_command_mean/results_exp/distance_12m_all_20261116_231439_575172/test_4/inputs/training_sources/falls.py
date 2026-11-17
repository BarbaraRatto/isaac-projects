"""Persistent fall detection sampled at physics frequency, without simulator imports."""
import math

import torch


class FallTracker:
    def __init__(self, count, device, physics_dt, tilt_deg=90., persistence_seconds=1., contact_threshold_n=1.):
        if physics_dt <= 0 or persistence_seconds <= 0 or not 0 < tilt_deg <= 180 or contact_threshold_n < 0:
            raise ValueError('Invalid fall detection settings')
        self.dt = physics_dt
        self.required_samples = math.ceil(persistence_seconds/physics_dt-1e-9)
        self.tilt_limit = tilt_deg
        self.contact_threshold = contact_threshold_n
        for name in ('tilt_steps', 'body_steps', 'max_tilt_steps', 'max_body_steps'):
            setattr(self, name, torch.zeros(count, device=device, dtype=torch.long))
        for name in ('tilt_deg', 'body_force_n', 'max_tilt_deg', 'max_body_force_n'):
            setattr(self, name, torch.zeros(count, device=device))
        self.tilt_confirmed = torch.zeros(count, device=device, dtype=torch.bool)
        self.body_confirmed = torch.zeros_like(self.tilt_confirmed)

    @property
    def fallen(self):
        return self.tilt_confirmed | self.body_confirmed

    def update(self, gravity_b, body_force_n):
        if not bool(torch.isfinite(gravity_b).all() & torch.isfinite(body_force_n).all()):
            raise RuntimeError('Non-finite physics/contact data; not classified as a fall')
        self.tilt_deg.copy_(torch.rad2deg(torch.acos((-gravity_b[:, 2]).clamp(-1., 1.))))
        self.body_force_n.copy_(body_force_n)
        tilt = self.tilt_deg >= self.tilt_limit
        contact = body_force_n > self.contact_threshold
        self.tilt_steps.copy_(torch.where(tilt, self.tilt_steps+1, 0))
        self.body_steps.copy_(torch.where(contact, self.body_steps+1, 0))
        self.max_tilt_steps.copy_(torch.maximum(self.max_tilt_steps, self.tilt_steps))
        self.max_body_steps.copy_(torch.maximum(self.max_body_steps, self.body_steps))
        self.max_tilt_deg.copy_(torch.maximum(self.max_tilt_deg, self.tilt_deg))
        self.max_body_force_n.copy_(torch.maximum(self.max_body_force_n, body_force_n))
        # Latch a confirmed event until reset, even if it ends before the next
        # navigation boundary. Short events reset their consecutive counters.
        self.tilt_confirmed |= self.tilt_steps >= self.required_samples
        self.body_confirmed |= self.body_steps >= self.required_samples

    def reset(self, ids):
        for name in ('tilt_steps', 'body_steps', 'max_tilt_steps', 'max_body_steps',
                     'tilt_deg', 'body_force_n', 'max_tilt_deg', 'max_body_force_n',
                     'tilt_confirmed', 'body_confirmed'):
            getattr(self, name)[ids] = 0

    def cause(self, index):
        return '+'.join(name for name, flag in (('tilt', self.tilt_confirmed[index]),
                                              ('body_contact', self.body_confirmed[index])) if bool(flag))


def terrain_contact_force(force_matrix):
    """One trunk body, terrain filters only; feet/self contacts are excluded."""
    if force_matrix.ndim != 4 or tuple(force_matrix.shape[1:]) != (1, 49, 3):
        raise RuntimeError(f'Expected trunk contact force matrix N x 1 x 49 x 3, got {force_matrix.shape}')
    # Sum magnitudes per terrain group so opposing contacts cannot cancel out.
    return force_matrix[:, 0].norm(dim=-1).sum(-1)


FALL_FIELDS = ['fall_cause', 'tilt_deg', 'body_contact_force_n', 'tilt_duration_s',
               'body_contact_duration_s', 'max_tilt_deg', 'max_body_contact_force_n',
               'max_tilt_duration_s', 'max_body_contact_duration_s', 'terrain_height_valid',
               'base_clearance_m', 'low_clearance_steps', 'invalid_height_steps', 'legacy_fall_steps']
DIAGNOSTIC_FIELDS = ['env_id', 'episode', 'step', 'time_s', 'x', 'y', 'z', *FALL_FIELDS]

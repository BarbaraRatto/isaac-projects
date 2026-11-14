"""Opt-in synchronized phase timing, without changing the learning configuration."""
from contextlib import contextmanager
import time
import torch


class PhaseTimers:
    def __init__(self, enabled=False, device='cpu'):
        self.enabled, self.device = enabled, torch.device(device)
        self.seconds, self.calls = {}, {}
        self.started = time.perf_counter()

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.seconds[name] = self.seconds.get(name, 0.)+time.perf_counter()-start
            self.calls[name] = self.calls.get(name, 0)+1

    def report(self):
        return dict(enabled=self.enabled, wall_seconds=time.perf_counter()-self.started,
                    phases={key: dict(seconds=value, calls=self.calls[key],
                                      mean_ms=1000*value/self.calls[key])
                            for key, value in self.seconds.items()},
                    note='Cumulative synchronized wall times. Nested vision phases overlap collect/evaluation; do not sum all phases.')

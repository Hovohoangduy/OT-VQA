"""Opt-in synchronized stage timing for pilots; disabled for normal training."""
from contextlib import contextmanager
from time import perf_counter
import torch


class StageProfiler:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.seconds = {}
        self.calls = {}

    @contextmanager
    def measure(self, name, device):
        if not self.enabled:
            yield
            return
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = perf_counter()
        try:
            yield
        finally:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            self.seconds[name] = self.seconds.get(name, 0.) + perf_counter()-started
            self.calls[name] = self.calls.get(name, 0) + 1

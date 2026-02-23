"""Feature replay buffer for catastrophic forgetting prevention.

Stores penultimate-layer activations from earlier phases for efficient
replay. Cheaper than re-running full forward passes on old data.
"""
import random
from collections import defaultdict

import torch


class FeatureReplayBuffer:
    """Store and sample (feature, label) pairs per task."""

    def __init__(self, max_per_task: int = 1000):
        self.max_per_task = max_per_task
        self.buffers: dict[str, list[tuple[torch.Tensor, int]]] = defaultdict(list)

    def save(self, task: str, features: torch.Tensor, label: int):
        """Store a (feature, label) pair. Evicts oldest if at capacity."""
        buf = self.buffers[task]
        buf.append((features.detach().cpu(), label))
        if len(buf) > self.max_per_task:
            buf.pop(0)

    def sample(self, task: str, n: int) -> list[tuple[torch.Tensor, int]]:
        """Sample n items from task buffer. Returns fewer if buffer smaller."""
        buf = self.buffers.get(task, [])
        if not buf:
            return []
        return random.sample(buf, min(n, len(buf)))

    def sample_ratio(self, task: str, ratio: float) -> list[tuple[torch.Tensor, int]]:
        """Sample ratio * buffer_size items."""
        buf = self.buffers.get(task, [])
        n = int(len(buf) * ratio)
        return self.sample(task, max(1, n)) if buf else []

    def tasks(self) -> list[str]:
        """Return list of tasks with stored features."""
        return list(self.buffers.keys())

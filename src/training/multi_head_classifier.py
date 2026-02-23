"""Persistent per-task classifier heads.

Replaces the destructive _rebuild_classifier() pattern that destroyed weights
when switching tasks during curriculum training.
"""
import torch.nn as nn


class MultiHeadClassifier(nn.Module):
    """Dictionary of classifier heads keyed by task name.

    Each head is a simple Linear layer. Heads persist across task switches —
    no weight destruction.
    """

    def __init__(self, input_dim: int, task_classes: dict[str, int]):
        super().__init__()
        self.input_dim = input_dim
        self.heads = nn.ModuleDict({
            task: nn.Linear(input_dim, n_classes)
            for task, n_classes in task_classes.items()
        })

    def forward(self, x, task: str):
        return self.heads[task](x)

    def add_task(self, task: str, n_classes: int):
        """Add a new task head (for tasks discovered at runtime)."""
        self.heads[task] = nn.Linear(self.input_dim, n_classes)

"""Persistent per-task classifier heads (MLP-based).

Each task head is a small MLP: Linear->GELU->Dropout->Linear->GELU->Dropout->Linear.
Provides enough capacity to learn text*structure interactions (vs single Linear).
"""
import torch.nn as nn


def _make_task_head(input_dim: int, n_classes: int,
                    hidden_dim: int = 128, dropout: float = 0.2) -> nn.Sequential:
    """Create an MLP classifier head for one task."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, n_classes),
    )


class MultiHeadClassifier(nn.Module):
    """Dictionary of MLP classifier heads keyed by task name.

    Each head is a 3-layer MLP. Heads persist across task switches —
    no weight destruction.
    """

    def __init__(self, input_dim: int, task_classes: dict[str, int],
                 hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.heads = nn.ModuleDict({
            task: _make_task_head(input_dim, n_classes, hidden_dim, dropout)
            for task, n_classes in task_classes.items()
        })

    def forward(self, x, task: str):
        return self.heads[task](x)

    def add_task(self, task: str, n_classes: int):
        """Add a new task head (for tasks discovered at runtime)."""
        self.heads[task] = _make_task_head(
            self.input_dim, n_classes, self.hidden_dim, self.dropout,
        )

from __future__ import annotations

import torch
import torch.nn as nn


class TopologicalConfidence(nn.Module):
    """Produces calibrated confidence scores from logits + topological features.

    Input topological features: [gradient_ratio, curl_ratio, harmonic_ratio, spectral_gap]
    Output: scalar confidence in [0, 1] per sample.

    Unlike softmax entropy, this measures how cleanly the model reasoned,
    not how peaked the output distribution is.
    """

    def __init__(self, num_classes: int, topo_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes + topo_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        logits: torch.Tensor,
        topo_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) raw model output
            topo_features: (B, topo_dim) topological diagnostics

        Returns:
            (B,) confidence scores in [0, 1]
        """
        combined = torch.cat([logits, topo_features], dim=-1)
        return self.net(combined).squeeze(-1)

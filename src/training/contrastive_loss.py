"""Auxiliary contrastive loss for semantic features.

Gives the semantic model (DSM or LLM adapter) a direct gradient signal:
nodes that are connected (or structurally similar) should have similar
semantic features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss on node semantic features.

    Positive pairs: nodes connected by an edge (adjacency[i,j] > 0).
    Negative pairs: all other node pairs.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            features: (N, D) semantic features per node
            adjacency: (N, N) binary adjacency matrix (positive pairs)

        Returns:
            Scalar loss (0.0 if no positive pairs).
        """
        N = features.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=features.device)

        features = F.normalize(features, dim=-1)
        sim = features @ features.T / self.temperature

        mask_self = ~torch.eye(N, dtype=torch.bool, device=features.device)
        positive_mask = (adjacency > 0) & mask_self

        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        log_prob = sim - torch.logsumexp(
            sim.masked_fill(~mask_self, float('-inf')), dim=-1, keepdim=True
        )

        loss = -(log_prob * positive_mask.float()).sum() / positive_mask.sum()
        return loss

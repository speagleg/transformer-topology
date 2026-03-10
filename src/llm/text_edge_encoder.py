"""Text-derived edge feature encoder.

Computes semantically-grounded edge features from node text embeddings.
Used as a side-channel into GNN message-passing (Phase 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextEdgeEncoder(nn.Module):
    """Compute per-edge text features from endpoint node text embeddings.

    For each edge (u, v), concatenates [text_u, text_v, text_u - text_v,
    text_u * text_v] and projects through an MLP. Output is gated by a
    learned scalar initialized near zero.

    Args:
        text_dim: Dimension of input node text features.
        edge_dim: Dimension of output edge features.
    """

    def __init__(self, text_dim: int = 32, edge_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * text_dim, 2 * edge_dim),
            nn.GELU(),
            nn.Linear(2 * edge_dim, edge_dim),
            nn.LayerNorm(edge_dim),
        )
        self.gate = nn.Parameter(torch.tensor(-3.0))

    def forward(
        self,
        text_features: torch.Tensor | None,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
    ) -> torch.Tensor | None:
        if text_features is None:
            return None
        text_src = text_features[src_idx]
        text_tgt = text_features[tgt_idx]
        edge_input = torch.cat(
            [text_src, text_tgt, text_src - text_tgt, text_src * text_tgt],
            dim=-1,
        )
        return torch.sigmoid(self.gate) * self.mlp(edge_input)

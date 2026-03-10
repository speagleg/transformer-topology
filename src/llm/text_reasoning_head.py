"""Graph-aware text reasoning transformer.

Processes per-node text embeddings with adjacency-masked self-attention,
producing semantically-enriched per-node representations for classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextReasoningHead(nn.Module):
    """Lightweight transformer over node text embeddings with graph-structure attention masking.

    Each transformer layer uses adjacency-masked multi-head self-attention so that
    nodes only attend to their graph neighbors (plus self-loops). This lets text
    features participate in graph-aware reasoning rather than being concatenated
    shallowly at the classifier.

    Args:
        input_dim: Dimension of input text features.
        hidden_dim: Internal hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            TextReasoningLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        text_features: torch.Tensor | None,
        adjacency: torch.Tensor,
    ) -> torch.Tensor | None:
        """Forward pass with adjacency-masked attention.

        Args:
            text_features: Per-node text embeddings of shape (N, input_dim),
                or None if no text is available.
            adjacency: Adjacency matrix of shape (N, N). Non-zero entries
                indicate edges; self-loops are added automatically.

        Returns:
            Enriched text representations of shape (N, hidden_dim), or None
            if text_features is None.
        """
        if text_features is None:
            return None

        x = self.input_proj(text_features)

        # Build additive attention mask: 0 for allowed, -inf for blocked.
        # Add self-loops so every node can attend to itself.
        adj_with_self = adjacency + torch.eye(
            adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype
        )
        blocked = adj_with_self == 0
        attn_mask = torch.zeros_like(adjacency)
        attn_mask.masked_fill_(blocked, float("-inf"))

        for layer in self.layers:
            x = layer(x, attn_mask)

        return x


class TextReasoningLayer(nn.Module):
    """Single transformer layer with adjacency-masked attention and FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Pre-norm transformer block with masked attention.

        Args:
            x: Node representations of shape (N, hidden_dim).
            attn_mask: Additive attention mask of shape (N, N).

        Returns:
            Updated representations of shape (N, hidden_dim).
        """
        # MHA expects batch dimension; add and remove it.
        x_batch = x.unsqueeze(0)
        attn_out, _ = self.attn(x_batch, x_batch, x_batch, attn_mask=attn_mask)
        x = self.norm1(x + attn_out.squeeze(0))
        x = self.norm2(x + self.ffn(x))
        return x

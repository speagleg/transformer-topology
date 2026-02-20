"""Distilled Semantic Model (DSM): 250M trainable transformer with cross-attention port.

Architecture:
    - 16 self-attention layers (configurable)
    - Cross-attention port at layer 4 (configurable) for topo_memory injection
    - No vocabulary/tokenizer — operates on continuous embeddings from TopoBridge
    - Fully trainable (no frozen weights)
"""

import torch
import torch.nn as nn


class DSMBlock(nn.Module):
    """Standard transformer block: self-attention + FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=False,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention + FFN with pre-norm residuals.

        Args:
            x: (seq_len, hidden_dim)

        Returns:
            (seq_len, hidden_dim)
        """
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class DSMCrossAttentionBlock(nn.Module):
    """Transformer block with self-attention + cross-attention + FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=False,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=False,
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Self-attention + cross-attention over topo_memory + FFN.

        Args:
            x: (seq_len, hidden_dim) token sequence.
            memory: (N, hidden_dim) topo_memory from TopoBridge encoder.

        Returns:
            (seq_len, hidden_dim)
        """
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out
        normed = self.norm_cross(x)
        cross_out, _ = self.cross_attn(normed, memory, memory)
        x = x + cross_out
        x = x + self.ff(self.norm2(x))
        return x


class DistilledSemanticModel(nn.Module):
    """250M distilled semantic model with cross-attention port.

    Layers 0 to cross_attn_layer-1: plain self-attention (DSMBlock)
    Layer cross_attn_layer: self-attn + cross-attn (DSMCrossAttentionBlock)
    Layers cross_attn_layer+1 to end: plain self-attention (DSMBlock)

    Args:
        hidden_dim: Model dimension (1024 for full, 64 for test).
        num_heads: Attention heads (16 for full, 4 for test).
        ff_dim: FFN hidden dimension (4096 for full, 256 for test).
        num_layers: Total layers (16 for full, 4 for test).
        cross_attn_layer: Which layer gets cross-attention (4 for full).
        dropout: Dropout probability.
    """

    def __init__(self, hidden_dim: int = 1024, num_heads: int = 16,
                 ff_dim: int = 4096, num_layers: int = 16,
                 cross_attn_layer: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_attn_layer = cross_attn_layer

        layers = []
        for i in range(num_layers):
            if i == cross_attn_layer:
                layers.append(DSMCrossAttentionBlock(
                    hidden_dim, num_heads, ff_dim, dropout,
                ))
            else:
                layers.append(DSMBlock(hidden_dim, num_heads, ff_dim, dropout))
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, prefix: torch.Tensor, topo_memory: torch.Tensor,
                task_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """DSM forward pass.

        Args:
            prefix: (num_prefix, hidden_dim) from TopoBridge PrefixGenerator.
            topo_memory: (N, hidden_dim) projected node embeddings.
            task_tokens: (seq, hidden_dim) optional embedded task text.

        Returns:
            semantic_hidden: (num_prefix + seq, hidden_dim) hidden states.
        """
        if task_tokens is not None:
            x = torch.cat([prefix, task_tokens], dim=0)
        else:
            x = prefix
        for i, layer in enumerate(self.layers):
            if i == self.cross_attn_layer:
                x = layer(x, topo_memory)
            else:
                x = layer(x)
        return self.final_norm(x)

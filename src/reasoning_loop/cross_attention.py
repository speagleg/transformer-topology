"""Cross-attention block for structure→text fusion in iteration 2."""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention: structural embeddings attend to text embeddings.

    h_fused = LayerNorm(h_struct + MultiHeadAttn(Q=h_struct, K=h_text, V=h_text))
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h_struct: torch.Tensor,   # (N, embed_dim) structural embeddings
        h_text: torch.Tensor,     # (M, embed_dim) text embeddings
    ) -> torch.Tensor:            # (N, embed_dim) fused embeddings
        """Cross-attend structural queries to text keys/values."""
        q = h_struct.unsqueeze(0)  # (1, N, d)
        k = h_text.unsqueeze(0)    # (1, M, d)
        v = h_text.unsqueeze(0)    # (1, M, d)

        attn_out, _ = self.cross_attn(q, k, v)  # (1, N, d)
        attn_out = attn_out.squeeze(0)  # (N, d)

        return self.norm(h_struct + attn_out)


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention between structural and text embeddings.

    Direction 1 (s→t): structural queries attend to text keys/values.
    Direction 2 (t→s): text queries attend to structural keys/values.

    Returns both enriched representations.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.s2t = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.t2s = CrossAttentionBlock(embed_dim, num_heads, dropout)

    def forward(
        self,
        h_struct: torch.Tensor,
        h_text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_struct_enriched, h_text_enriched)."""
        h_struct_enriched = self.s2t(h_struct, h_text)
        h_text_enriched = self.t2s(h_text, h_struct)
        return h_struct_enriched, h_text_enriched

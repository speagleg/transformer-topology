import torch
import torch.nn as nn
import math


class TopologicalSpatialAttention(nn.Module):
    """Multi-head attention biased by cell complex adjacency.

    Standard scaled dot-product attention where the attention scores are
    masked according to a topological adjacency matrix. Entries where
    adjacency == 0 are set to -inf before softmax, so only topologically
    connected cells can attend to each other.

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads (must evenly divide embed_dim).
        dropout: Dropout probability applied to attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.edge_proj = nn.Linear(1, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor,
                spatial_focus: torch.Tensor | None = None,
                edge_weights: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with adjacency-masked attention.

        Args:
            x: Node features of shape ``(N, embed_dim)``.
            adjacency: Binary adjacency matrix of shape ``(N, N)``.
                Non-zero entries indicate topologically connected cells.
            spatial_focus: Optional tensor of shape ``(N,)`` with values in
                [0, 1] used as an additive bias on attention scores to steer
                focus toward specific nodes.  When ``None``, no bias is added.
            edge_weights: Optional tensor of shape ``(N, N)`` with scalar edge
                weights.  When provided, projected per-head biases are added
                to the attention scores before softmax.

        Returns:
            Output features of shape ``(N, embed_dim)`` after residual
            connection and layer normalization.
        """
        n = x.shape[0]
        residual = x

        # Project to Q, K, V and reshape for multi-head attention
        # Shape: (num_heads, N, head_dim)
        q = self.q_proj(x).view(n, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(x).view(n, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(n, self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention scores: (num_heads, N, N)
        scores = (q @ k.transpose(-2, -1)) / self.scale

        # Apply adjacency mask: block attention to non-adjacent cells
        mask = adjacency.unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Apply edge weight bias
        if edge_weights is not None:
            edge_bias = self.edge_proj(edge_weights.unsqueeze(-1))  # (N, N, num_heads)
            scores = scores + edge_bias.permute(2, 0, 1)

        # Apply spatial focus bias from GNN executive control signal
        if spatial_focus is not None:
            # spatial_focus: (N,) → broadcast to (num_heads, N, N)
            # Bias the key dimension so attended-to nodes with higher focus
            # receive proportionally higher attention scores.
            scores = scores + spatial_focus.unsqueeze(0).unsqueeze(0)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Replace NaN from all-masked rows (softmax of all -inf) with zeros
        attn_weights = attn_weights.nan_to_num(0.0)

        # Weighted sum of values: (num_heads, N, head_dim) -> (N, embed_dim)
        out = attn_weights @ v
        out = out.transpose(0, 1).contiguous().view(n, self.embed_dim)
        out = self.out_proj(out)

        return self.norm(out + residual)

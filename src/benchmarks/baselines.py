"""Baseline models for multi-hop graph traversal benchmark.

Three baselines to contextualize the TopologyAwareTransformer + GNN Executive
+ ReasoningLoop results:

1. VanillaTransformerBaseline - Standard transformer, no topology awareness
2. GATBaseline - Graph Attention Network (PyG GATConv)
3. GCNBaseline - Graph Convolutional Network (PyG GCNConv)

All models share the same classifier head and forward signature as the main
MultiHopReasoningModel: forward(cc, query_node, target_node) -> logits.
"""

import math
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
from src.cell_complex.cell_complex import CellComplex


class _ClassifierHead(nn.Module):
    """Shared classifier head: (query_emb, target_emb, diff) -> logits."""

    def __init__(self, embedding_dim: int, max_hops: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3 * embedding_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, max_hops + 1),
        )

    def forward(self, node_embeddings: torch.Tensor,
                query_node: int, target_node: int) -> torch.Tensor:
        query_emb = node_embeddings[query_node]
        target_emb = node_embeddings[target_node]
        diff_emb = query_emb - target_emb
        combined = torch.cat([query_emb, target_emb, diff_emb])
        return self.classifier(combined)


# ---------------------------------------------------------------------------
# 1. Vanilla Transformer Baseline
# ---------------------------------------------------------------------------

class VanillaTransformerBaseline(nn.Module):
    """Standard transformer encoder over node embeddings — no topology awareness.

    Treats node embeddings as a flat sequence with learned positional encoding.
    Uses standard multi-head self-attention (no adjacency masking, no spectral
    features). Roughly matches the main model's parameter count (~2.9M) by
    using multiple transformer encoder layers with appropriate hidden dims.
    """

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128,
                 num_layers: int = 4, num_heads: int = 4, max_hops: int = 10,
                 max_nodes: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)

        # Learned positional encoding
        self.pos_embedding = nn.Embedding(max_nodes, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        self.head = _ClassifierHead(embedding_dim, max_hops)

    def forward(self, cc: CellComplex, query_node: int, target_node: int) -> torch.Tensor:
        x = cc.get_embeddings(0)  # (N, embedding_dim)
        N = x.size(0)

        x = self.input_proj(x)  # (N, hidden_dim)
        positions = torch.arange(N, device=x.device)
        x = x + self.pos_embedding(positions)

        # Transformer expects (batch, seq, features) with batch_first=True
        x = x.unsqueeze(0)  # (1, N, hidden_dim)
        x = self.encoder(x)
        x = x.squeeze(0)    # (N, hidden_dim)

        x = self.output_proj(x)  # (N, embedding_dim)
        return self.head(x, query_node, target_node)


# ---------------------------------------------------------------------------
# 2. GAT Baseline
# ---------------------------------------------------------------------------

class GATBaseline(nn.Module):
    """Graph Attention Network baseline using PyG's GATConv.

    Multiple GATConv layers with residual connections. Uses the cell complex's
    edge_index for message passing. Roughly matches the main model's parameter
    count by using multiple layers and heads.
    """

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128,
                 num_layers: int = 4, num_heads: int = 4, max_hops: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)

        assert hidden_dim % num_heads == 0
        head_dim = hidden_dim // num_heads

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GATConv(
                in_channels=hidden_dim,
                out_channels=head_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout,
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        self.head = _ClassifierHead(embedding_dim, max_hops)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cc: CellComplex, query_node: int, target_node: int) -> torch.Tensor:
        x = cc.get_embeddings(0)  # (N, embedding_dim)
        edge_index = cc.edge_index()  # (2, E)

        x = self.input_proj(x)  # (N, hidden_dim)

        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x, edge_index)
            x = self.dropout(x)
            x = norm(x + residual)

        x = self.output_proj(x)  # (N, embedding_dim)
        return self.head(x, query_node, target_node)


# ---------------------------------------------------------------------------
# 3. GCN Baseline
# ---------------------------------------------------------------------------

class GCNBaseline(nn.Module):
    """Graph Convolutional Network baseline using PyG's GCNConv.

    Multiple GCNConv layers with residual connections. Uses the cell complex's
    edge_index for message passing. Roughly matches the main model's parameter
    count by using wider hidden dimensions and more layers.
    """

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128,
                 num_layers: int = 4, max_hops: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        self.head = _ClassifierHead(embedding_dim, max_hops)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cc: CellComplex, query_node: int, target_node: int) -> torch.Tensor:
        x = cc.get_embeddings(0)  # (N, embedding_dim)
        edge_index = cc.edge_index()  # (2, E)

        x = self.input_proj(x)  # (N, hidden_dim)

        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = torch.relu(layer(x, edge_index))
            x = self.dropout(x)
            x = norm(x + residual)

        x = self.output_proj(x)  # (N, embedding_dim)
        return self.head(x, query_node, target_node)

"""Text-conditioned GNN for iteration 2: edge messages modulated by text similarity."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class TextModulatedMessagePassing(MessagePassing):
    """Message passing where edge messages are amplified by source/target text similarity.

    Note: stores text_embs as instance state during forward (standard PyG pattern).
    Not safe for DataParallel concurrent calls, but fine for single-GPU sequential training.
    """

    def __init__(self, in_dim: int, out_dim: int, text_dim: int):
        super().__init__(aggr='add')
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ELU(),
        )
        # Text similarity: [text_src, text_dst, text_src * text_dst] -> scalar
        self.text_sim = nn.Sequential(
            nn.Linear(3 * text_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,          # (N, in_dim)
        edge_index: torch.Tensor,  # (2, E)
        text_embs: torch.Tensor | None = None,  # (N, text_dim)
    ) -> torch.Tensor:
        if text_embs is not None:
            self._text_embs = text_embs
            self._use_text = True
        else:
            self._use_text = False
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor,
                edge_index_i: torch.Tensor, edge_index_j: torch.Tensor) -> torch.Tensor:
        msg = self.msg_mlp(x_j)

        if self._use_text:
            t_src = self._text_embs[edge_index_j]  # (E, text_dim)
            t_dst = self._text_embs[edge_index_i]  # (E, text_dim)
            sim_input = torch.cat([t_src, t_dst, t_src * t_dst], dim=-1)
            sim = self.text_sim(sim_input)  # (E, 1)
            msg = msg * (1 + sim)

        return msg


class TextConditionedGNN(nn.Module):
    """Multi-layer GNN with text-modulated message passing.

    Used only in iteration 2 of the dual-track executive loop.
    Uses hidden_dim == embed_dim to enable residual connections on every layer.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        # Force hidden_dim == embed_dim for residual connections
        hidden_dim = embed_dim
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(
                TextModulatedMessagePassing(embed_dim, embed_dim, embed_dim)
            )

        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,           # (N, embed_dim)
        edge_index: torch.Tensor,   # (2, E)
        text_embs: torch.Tensor | None = None,  # (N, embed_dim)
    ) -> torch.Tensor:              # (N, embed_dim)
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, edge_index, text_embs)
            h_new = norm(h_new)
            h_new = h_new + h  # residual (always same shape)
            h = h_new
        return h

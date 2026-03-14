"""Task-conditioned attention readout over all node embeddings."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class AttentionReadout(nn.Module):
    """Learned attention readout conditioned on task + query/target nodes.

    Produces a global context vector by attending over all node embeddings,
    using a query constructed from the task embedding + local node embeddings.
    """

    def __init__(self, embed_dim: int, num_tasks: int = 19, text_dim: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.task_embedding = nn.Embedding(num_tasks, embed_dim)
        # Query projection: [task_emb, h_query, h_target] -> readout_query
        self.query_proj = nn.Linear(3 * embed_dim, embed_dim)
        self._scale = math.sqrt(embed_dim)
        self.text_dim = text_dim
        # Default output: h_query(D) + h_target(D) + context(D) + topo(4) + fusion(1) + 2*text_dim
        self.output_dim = 3 * embed_dim + 4 + 1 + 2 * text_dim

    def forward(
        self,
        h_out: torch.Tensor,     # (N, embed_dim) all node embeddings
        query_idx: int,
        target_idx: int,
        task_id: int | None = None,
    ) -> torch.Tensor:           # (embed_dim,) global context vector
        """Compute attention-pooled context over all nodes."""
        device = h_out.device

        if task_id is not None:
            task_emb = self.task_embedding(
                torch.tensor(task_id, device=device)
            )
        else:
            task_emb = torch.zeros(self.embed_dim, device=device)

        h_query = h_out[query_idx]
        h_target = h_out[target_idx]

        # Build readout query
        q_input = torch.cat([task_emb, h_query, h_target])
        readout_q = self.query_proj(q_input)

        # Attention over all nodes
        scores = (h_out @ readout_q) / self._scale
        weights = torch.softmax(scores, dim=0)
        context = (weights.unsqueeze(-1) * h_out).sum(dim=0)

        return context

    def build_classifier_input(
        self,
        h_out: torch.Tensor,
        query_idx: int,
        target_idx: int,
        task_id: int | None = None,
        topo_features: torch.Tensor | None = None,
        fusion_weight: torch.Tensor | None = None,
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble full classifier input vector.

        Returns: tensor of h_query(D) + h_target(D) + context(D) + topo(4) + fusion(1) + text_q(text_dim) + text_t(text_dim)
        """
        context = self(h_out, query_idx, target_idx, task_id)

        parts = [
            h_out[query_idx],
            h_out[target_idx],
            context,
        ]

        if topo_features is not None:
            parts.append(topo_features)

        if fusion_weight is not None:
            parts.append(fusion_weight.unsqueeze(0) if fusion_weight.dim() == 0 else fusion_weight)

        if self.text_dim > 0 and text_embeddings is not None:
            parts.append(text_embeddings[query_idx])
            parts.append(text_embeddings[target_idx])
        elif self.text_dim > 0:
            # No text available — zero-pad to maintain consistent dim
            parts.append(torch.zeros(self.text_dim, device=h_out.device))
            parts.append(torch.zeros(self.text_dim, device=h_out.device))

        return torch.cat(parts)

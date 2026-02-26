"""Wrap DSM for TransformerTopologyAnalyzer which expects model(input_tensor)."""

import torch
import torch.nn as nn


class DSMAnalysisWrapper(nn.Module):
    """Wrap DSM to accept a single (batch, seq, dim) input for TransformerTopologyAnalyzer.

    The analyzer calls ``model(input_tensor)`` with a single positional arg,
    but DSM expects ``forward(prefix, topo_memory)``.  This wrapper bridges
    the two interfaces and exposes ``.layers`` so the hook manager can attach.
    """

    def __init__(self, dsm: nn.Module):
        super().__init__()
        self.dsm = dsm
        # Expose .layers so TransformerHookManager finds them
        self.layers = dsm.layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim) — squeeze batch dim for DSM
        prefix = x[0]  # (seq_len, hidden_dim)
        topo_memory = torch.zeros(1, prefix.shape[-1], device=x.device)
        return self.dsm(prefix, topo_memory).unsqueeze(0)

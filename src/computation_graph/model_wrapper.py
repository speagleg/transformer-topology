"""Wrap HierarchicalMultiHopModel for ComputationGraphCapture.

``analyze_computation_graph()`` calls ``model(input_tensor)`` with a single
positional arg. Our model expects ``forward(cc, query_node, target_node, ...)``.
This wrapper bridges the two interfaces.
"""

import torch
import torch.nn as nn

from src.computation_graph.diagnostics import analyze_computation_graph


class CellComplexModelWrapper(nn.Module):
    """Wrap a CellComplex-based model for ComputationGraphCapture.

    Captures a fixed (cc, query, target, metadata) so that the wrapper's
    ``forward(dummy)`` drives the real model.
    """

    def __init__(self, model: nn.Module, cc, query: int, target: int,
                 metadata=None):
        super().__init__()
        self.model = model
        self.cc = cc
        self.query = query
        self.target = target
        self.metadata = metadata

    def forward(self, dummy: torch.Tensor) -> torch.Tensor:
        logits = self.model(
            self.cc, self.query, self.target, metadata=self.metadata,
        )
        return logits.unsqueeze(0)  # (num_classes,) → (1, num_classes)


def analyze_computation_graph_cc(model, cc, query, target, answer, criterion,
                                  metadata=None):
    """Convenience: run computation graph analysis on a CellComplex model.

    Clones the CellComplex to avoid mutation side-effects.
    """
    device = next(model.parameters()).device
    wrapper = CellComplexModelWrapper(model, cc.clone().to(device), query, target, metadata)
    dummy = torch.zeros(1, device=device)
    target_tensor = torch.tensor([answer], device=device)
    return analyze_computation_graph(wrapper, dummy, target_tensor, criterion)

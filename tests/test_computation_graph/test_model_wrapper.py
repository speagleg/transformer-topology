"""Tests for CellComplexModelWrapper and analyze_computation_graph_cc."""

import torch
import torch.nn as nn

from src.benchmarks.run_comparison import HierarchicalMultiHopModel
from src.benchmarks.multi_hop import MultiHopDataset
from src.computation_graph.model_wrapper import (
    CellComplexModelWrapper,
    analyze_computation_graph_cc,
)


def _make_model_and_sample():
    model = HierarchicalMultiHopModel(
        embedding_dim=16, gnn_hidden=32,
        gnn_spatial_layers=1, gnn_spectral_layers=1,
        max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2,
        tat_ff_dim=32, max_classes=6,
        max_iterations=2, convergence_threshold=0.1,
        use_wave_dynamics=False, use_higher_order=False,
    )
    ds = MultiHopDataset(
        num_samples=1, min_hops=1, max_hops=5,
        num_distractors=3, embedding_dim=16,
    )
    cc, query, target, answer = ds[0][:4]
    return model, cc, query, target, answer


class TestCellComplexModelWrapper:
    def test_wrapper_forward_returns_logits(self):
        model, cc, query, target, answer = _make_model_and_sample()
        wrapper = CellComplexModelWrapper(model, cc.clone(), query, target)
        dummy = torch.zeros(1)
        logits = wrapper(dummy)
        assert logits.dim() == 2  # (1, num_classes) for criterion compat
        assert logits.shape[1] == 6  # max_classes

    def test_analyze_computation_graph_cc(self):
        model, cc, query, target, answer = _make_model_and_sample()
        criterion = nn.CrossEntropyLoss()
        diag = analyze_computation_graph_cc(
            model, cc, query, target, answer, criterion,
        )
        assert hasattr(diag, 'gradient_energy_ratio')
        assert hasattr(diag, 'spectral_gap')
        assert diag.num_operations >= 0

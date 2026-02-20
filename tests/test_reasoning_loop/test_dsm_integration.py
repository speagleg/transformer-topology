"""Tests for DSM integration in ExecutiveReasoningLoop."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop


def _make_cc(n_nodes=10, embedding_dim=32):
    """Create a simple chain cell complex for testing."""
    cc = CellComplex(embedding_dim=embedding_dim)
    for i in range(n_nodes):
        cc.add_0_cell(torch.randn(embedding_dim), "node")
    for i in range(n_nodes - 1):
        cc.add_1_cell(i, i + 1, torch.randn(embedding_dim), "edge")
    return cc


class TestDSMInExecutiveLoop:
    def _make_loop(self, use_dsm=True):
        return ExecutiveReasoningLoop(
            embedding_dim=32, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=1,
            max_freqs=8, tat_layers=1,
            tat_spatial_heads=4, tat_spectral_heads=4,
            tat_ff_dim=64, max_iterations=2,
            convergence_threshold=0.01,
            use_wave_dynamics=True,
            use_dsm=use_dsm,
            dsm_config={'dsm_dim': 64, 'num_heads': 4, 'ff_dim': 128,
                        'num_layers': 2, 'cross_attn_layer': 0,
                        'num_prefix': 4},
        )

    def test_forward_with_dsm(self):
        loop = self._make_loop(use_dsm=True)
        cc = _make_cc()
        output, num_iters, diagnostics = loop(cc)
        assert output.shape == (10, 32)
        assert num_iters >= 1

    def test_forward_without_dsm(self):
        loop = self._make_loop(use_dsm=False)
        cc = _make_cc()
        output, num_iters, diagnostics = loop(cc)
        assert output.shape == (10, 32)

    def test_dsm_has_semantic_weight_in_diagnostics(self):
        loop = self._make_loop(use_dsm=True)
        cc = _make_cc()
        _, _, diagnostics = loop(cc)
        control_signals = diagnostics['control_signals']
        for cs in control_signals:
            assert cs.semantic_weight is not None
            assert 0.0 <= cs.semantic_weight.item() <= 1.0

    def test_dsm_config_optional(self):
        loop = self._make_loop(use_dsm=False)
        assert not hasattr(loop, 'topo_bridge') or loop.topo_bridge is None

    def test_output_finite(self):
        loop = self._make_loop(use_dsm=True)
        cc = _make_cc()
        output, _, _ = loop(cc)
        assert torch.isfinite(output).all()

    def test_variable_graph_sizes(self):
        loop = self._make_loop(use_dsm=True)
        for n in [5, 10, 20]:
            cc = _make_cc(n_nodes=n)
            output, _, _ = loop(cc)
            assert output.shape == (n, 32)

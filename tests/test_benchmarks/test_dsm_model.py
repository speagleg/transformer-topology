"""Tests for HierarchicalMultiHopModel with DSM backend."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.run_comparison import HierarchicalMultiHopModel


def _make_cc(n=10, dim=32):
    """Create a simple chain cell complex for testing."""
    cc = CellComplex(embedding_dim=dim)
    for i in range(n):
        cc.add_0_cell(torch.randn(dim), "node")
    for i in range(n - 1):
        cc.add_1_cell(i, i + 1, torch.randn(dim), "edge")
    return cc


class TestHierarchicalModelWithDSM:
    def _make_model(self, use_dsm=True):
        dsm_config = {
            'backend': 'dsm',
            'dsm_dim': 64,
            'num_heads': 4,
            'ff_dim': 128,
            'num_layers': 2,
            'cross_attn_layer': 0,
            'num_prefix': 4,
        }
        return HierarchicalMultiHopModel(
            embedding_dim=32, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=1,
            max_freqs=8, tat_layers=1,
            tat_spatial_heads=4, tat_spectral_heads=4,
            tat_ff_dim=64, max_classes=5,
            max_iterations=2, convergence_threshold=0.01,
            use_llm=use_dsm,
            llm_config=dsm_config if use_dsm else None,
        )

    def test_forward_with_dsm(self):
        model = self._make_model(use_dsm=True)
        cc = _make_cc()
        out = model(cc.clone(), 0, 1)
        assert out.shape == (5,)

    def test_forward_without_dsm(self):
        model = self._make_model(use_dsm=False)
        cc = _make_cc()
        out = model(cc.clone(), 0, 1)
        assert out.shape == (5,)

    def test_dsm_no_topo_bridge_at_model_level(self):
        """When backend=dsm, TopoBridge is inside the loop, not at model level."""
        model = self._make_model(use_dsm=True)
        assert model.topo_bridge is None

    def test_dsm_has_topo_bridge_in_loop(self):
        model = self._make_model(use_dsm=True)
        assert model.executive_loop.topo_bridge is not None

    def test_output_finite(self):
        model = self._make_model(use_dsm=True)
        cc = _make_cc()
        out = model(cc.clone(), 0, 1)
        assert torch.isfinite(out).all()

    def test_backward_pass(self):
        model = self._make_model(use_dsm=True)
        cc = _make_cc()
        out = model(cc.clone(), 0, 1)
        loss = torch.nn.functional.cross_entropy(out.unsqueeze(0), torch.tensor([1]))
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_variable_graph_sizes(self):
        model = self._make_model(use_dsm=True)
        for n in [5, 10, 20]:
            cc = _make_cc(n=n)
            out = model(cc.clone(), 0, min(1, n - 1))
            assert out.shape == (5,)

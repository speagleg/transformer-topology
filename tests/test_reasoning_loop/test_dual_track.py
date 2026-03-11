"""Tests for dual-track executive reasoning loop."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop


def _make_cc(n=10, embed_dim=32):
    """Create a simple CellComplex with node_texts for dual-track testing."""
    cc = CellComplex(embedding_dim=embed_dim)
    for i in range(n):
        cc.add_0_cell(torch.randn(embed_dim), "node")
    edges = [(i, i + 1) for i in range(n - 1)]
    for u, v in edges:
        cc.add_1_cell(u, v, torch.randn(embed_dim), "edge")
    cc.node_texts = [f"concept_{i}" for i in range(n)]
    return cc


DIM = 32
DUAL_KWARGS = dict(
    embedding_dim=DIM, gnn_hidden=64, gnn_spatial_layers=2,
    gnn_spectral_layers=1, max_freqs=8, tat_layers=1,
    tat_spatial_heads=4, tat_spectral_heads=4, tat_ff_dim=64,
    max_iterations=2, use_dual_track=True,
    use_wave_dynamics=False,
)


class TestDualTrackLoop:

    def test_forward_with_text_embeddings(self):
        loop = ExecutiveReasoningLoop(**DUAL_KWARGS)
        cc = _make_cc()
        text_embs = torch.randn(10, DIM)
        out, iters, diag = loop(cc, text_embeddings=text_embs)
        assert out.shape == (10, DIM)
        assert iters == 2  # always 2 in dual-track mode

    def test_forward_without_text_falls_back(self):
        loop = ExecutiveReasoningLoop(**DUAL_KWARGS)
        cc = _make_cc()
        out, iters, diag = loop(cc, text_embeddings=None)
        assert out.shape == (10, DIM)

    def test_diagnostics_include_fusion_weight(self):
        loop = ExecutiveReasoningLoop(**DUAL_KWARGS)
        cc = _make_cc()
        text_embs = torch.randn(10, DIM)
        _, _, diag = loop(cc, text_embeddings=text_embs)
        assert 'fusion_weight' in diag

    def test_gradient_flows_through_both_iterations(self):
        loop = ExecutiveReasoningLoop(**DUAL_KWARGS)
        cc = _make_cc()
        text_embs = torch.randn(10, DIM, requires_grad=True)
        out, _, _ = loop(cc, text_embeddings=text_embs)
        out.sum().backward()
        # Text embeddings should receive gradient (through cross-attention)
        assert text_embs.grad is not None
        # GNN params should also have gradient
        gnn_has_grad = any(
            p.grad is not None
            for p in loop.gnn_executive.parameters()
        )
        assert gnn_has_grad

    def test_non_dual_track_unchanged(self):
        """Non-dual-track mode should still work identically."""
        kwargs = {**DUAL_KWARGS, 'use_dual_track': False}
        loop = ExecutiveReasoningLoop(**kwargs)
        cc = _make_cc()
        out, _, _ = loop(cc)
        assert out.shape == (10, DIM)
        # Should not have dual-track modules
        assert not hasattr(loop, 'text_gnn')

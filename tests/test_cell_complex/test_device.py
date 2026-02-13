"""Tests for CellComplex device support (Step 3 prep)."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex


def _make_cc(embedding_dim=8):
    """Build a small CellComplex for testing."""
    cc = CellComplex(embedding_dim=embedding_dim)
    for i in range(4):
        cc.add_0_cell(torch.randn(embedding_dim), f"node_{i}")
    cc.add_1_cell(0, 1, torch.randn(embedding_dim), "edge")
    cc.add_1_cell(1, 2, torch.randn(embedding_dim), "edge")
    cc.add_1_cell(2, 0, torch.randn(embedding_dim), "edge")
    cc.add_1_cell(2, 3, torch.randn(embedding_dim), "edge")
    cc.add_2_cell([0, 1, 2], torch.randn(embedding_dim), "face")
    return cc


class TestCellComplexDevice:
    def test_default_device_is_cpu(self):
        cc = _make_cc()
        assert cc.device == torch.device('cpu')

    def test_empty_cc_device_is_cpu(self):
        cc = CellComplex(embedding_dim=8)
        assert cc.device == torch.device('cpu')

    def test_to_cpu_roundtrip(self):
        cc = _make_cc()
        original_embs = cc.get_embeddings(0).clone()
        cc.to('cpu')
        assert cc.device == torch.device('cpu')
        assert torch.allclose(cc.get_embeddings(0), original_embs)

    def test_to_returns_self(self):
        cc = _make_cc()
        result = cc.to('cpu')
        assert result is cc

    def test_boundary_operator_device(self):
        cc = _make_cc()
        B1 = cc.boundary_operator(1)
        assert B1.device == cc.device
        B2 = cc.boundary_operator(2)
        assert B2.device == cc.device

    def test_adjacency_matrix_device(self):
        cc = _make_cc()
        A = cc.adjacency_matrix(0)
        assert A.device == cc.device

    def test_edge_weight_matrix_device(self):
        cc = _make_cc()
        W = cc.edge_weight_matrix()
        assert W.device == cc.device

    def test_edge_index_device(self):
        cc = _make_cc()
        ei = cc.edge_index()
        assert ei.device == cc.device

    def test_compute_structural_features_device(self):
        cc = _make_cc()
        sf = cc.compute_structural_features()
        assert sf.device == cc.device

    def test_get_embeddings_empty_device(self):
        cc = CellComplex(embedding_dim=8)
        e0 = cc.get_embeddings(0)
        assert e0.device == torch.device('cpu')
        e1 = cc.get_embeddings(1)
        assert e1.device == torch.device('cpu')
        e2 = cc.get_embeddings(2)
        assert e2.device == torch.device('cpu')

    def test_to_preserves_all_embeddings(self):
        cc = _make_cc()
        e0_before = cc.get_embeddings(0).clone()
        e1_before = cc.get_embeddings(1).clone()
        e2_before = cc.get_embeddings(2).clone()
        cc.to('cpu')
        assert torch.allclose(cc.get_embeddings(0), e0_before)
        assert torch.allclose(cc.get_embeddings(1), e1_before)
        assert torch.allclose(cc.get_embeddings(2), e2_before)

    def test_to_preserves_structure(self):
        cc = _make_cc()
        B1_before = cc.boundary_operator(1).clone()
        cc.to('cpu')
        B1_after = cc.boundary_operator(1)
        assert torch.allclose(B1_before, B1_after)

    def test_chain_complex_property_after_to(self):
        cc = _make_cc()
        cc.to('cpu')
        assert cc.verify_chain_complex()

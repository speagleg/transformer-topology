"""Tests for ControlHead extension with 6-feature embedding topology feedback."""

import torch
import pytest
from src.gnn_executive.control_head import ControlHead


class TestControlHeadEmbeddingTopo:
    def test_backward_compat_3_features(self):
        """Original use_topo_feedback=True still works with 3 features."""
        head = ControlHead(embedding_dim=16, num_freqs=8, use_topo_feedback=True)
        node_embs = torch.randn(5, 16)
        topo = torch.randn(3)
        signal = head(node_embs, topo_features=topo)
        assert signal.frequency_gate.shape == (8,)

    def test_6_feature_embedding_topo(self):
        """New use_embedding_topo_feedback=True accepts 6 features."""
        head = ControlHead(
            embedding_dim=16, num_freqs=8,
            use_topo_feedback=True, use_embedding_topo_feedback=True,
        )
        node_embs = torch.randn(5, 16)
        topo = torch.randn(6)
        signal = head(node_embs, topo_features=topo)
        assert signal.frequency_gate.shape == (8,)

    def test_trunk_dim_with_embedding_topo(self):
        """Trunk input should be embedding_dim + 2 + 6."""
        head = ControlHead(
            embedding_dim=16, num_freqs=8,
            use_topo_feedback=True, use_embedding_topo_feedback=True,
        )
        # trunk input: 16 + 2 + 6 = 24
        assert head.trunk[0].in_features == 24

    def test_trunk_dim_without_embedding_topo(self):
        """Without embedding topo, trunk input should be embedding_dim + 2 + 3."""
        head = ControlHead(
            embedding_dim=16, num_freqs=8, use_topo_feedback=True,
        )
        assert head.trunk[0].in_features == 21  # 16 + 2 + 3

    def test_trunk_dim_no_topo(self):
        """No topo feedback at all: embedding_dim + 2."""
        head = ControlHead(embedding_dim=16, num_freqs=8)
        assert head.trunk[0].in_features == 18  # 16 + 2

"""Tests for TopologicalProfile and LayerCellComplex data structures."""

import time
import numpy as np
import torch
import pytest
from src.cell_complex.cell_complex import CellComplex


def _make_cc(n_nodes=5, dim=8):
    cc = CellComplex(embedding_dim=dim)
    for i in range(n_nodes):
        cc.add_0_cell(torch.randn(dim), "node")
    for i in range(n_nodes - 1):
        cc.add_1_cell(i, i + 1, torch.randn(dim), "edge")
    return cc


class TestLayerCellComplex:
    def test_create_from_embedding(self):
        from src.topology_analyzer.profile import LayerCellComplex
        cc = _make_cc()
        lcc = LayerCellComplex(
            cc=cc, layer_idx=0, source="embedding",
            head_idx=None, neighborhood="knn", k_or_epsilon=5.0,
        )
        assert lcc.layer_idx == 0
        assert lcc.source == "embedding"
        assert lcc.head_idx is None

    def test_create_from_attention(self):
        from src.topology_analyzer.profile import LayerCellComplex
        cc = _make_cc()
        lcc = LayerCellComplex(
            cc=cc, layer_idx=2, source="attention",
            head_idx=3, neighborhood="knn", k_or_epsilon=5.0,
        )
        assert lcc.head_idx == 3
        assert lcc.source == "attention"


class TestTopologicalProfile:
    def test_create_empty(self):
        from src.topology_analyzer.profile import TopologicalProfile
        profile = TopologicalProfile(
            per_layer_persistence={},
            per_layer_betti={},
            per_layer_spectral_gap={},
            per_layer_hodge_ratios={},
            cross_layer_sheaf_gap=0.0,
            per_head_hodge_ratios={},
            per_head_classification={},
            per_layer_head_sheaf_gap={},
            cross_layer_attention_wasserstein={},
            per_layer_effective_rank={},
            per_layer_condition_number={},
            per_layer_sv_persistence={},
            cross_layer_weight_sheaf_gap=0.0,
            model_name="test",
            num_layers=4,
            num_heads=4,
            hidden_dim=64,
            seq_len=16,
            timestamp=time.time(),
        )
        assert profile.num_layers == 4
        assert profile.model_name == "test"

    def test_to_dict_and_summary(self):
        from src.topology_analyzer.profile import TopologicalProfile
        profile = TopologicalProfile(
            per_layer_persistence={0: [np.array([[0.0, 0.5]]), np.empty((0, 2))]},
            per_layer_betti={0: (1, 0)},
            per_layer_spectral_gap={0: 0.5},
            per_layer_hodge_ratios={0: (0.6, 0.3, 0.1)},
            cross_layer_sheaf_gap=0.42,
            per_head_hodge_ratios={(0, 0): (0.7, 0.2, 0.1)},
            per_head_classification={(0, 0): "gradient"},
            per_layer_head_sheaf_gap={0: 0.3},
            cross_layer_attention_wasserstein={},
            per_layer_effective_rank={0: 10.5},
            per_layer_condition_number={0: 100.0},
            per_layer_sv_persistence={0: [np.array([[0.1, 0.9]])]},
            cross_layer_weight_sheaf_gap=0.35,
            model_name="test",
            num_layers=1,
            num_heads=1,
            hidden_dim=64,
            seq_len=16,
            timestamp=time.time(),
        )
        summary = profile.summary()
        assert "cross_layer_sheaf_gap" in summary
        assert summary["cross_layer_sheaf_gap"] == 0.42

    def test_active_features_vector(self):
        from src.topology_analyzer.profile import TopologicalProfile
        profile = TopologicalProfile(
            per_layer_persistence={},
            per_layer_betti={},
            per_layer_spectral_gap={},
            per_layer_hodge_ratios={},
            cross_layer_sheaf_gap=0.42,
            per_head_hodge_ratios={(0, 0): (0.3, 0.5, 0.2)},
            per_head_classification={(0, 0): "curl"},
            per_layer_head_sheaf_gap={},
            cross_layer_attention_wasserstein={},
            per_layer_effective_rank={},
            per_layer_condition_number={},
            per_layer_sv_persistence={},
            cross_layer_weight_sheaf_gap=0.35,
            model_name="test",
            num_layers=1,
            num_heads=1,
            hidden_dim=64,
            seq_len=16,
            timestamp=time.time(),
        )
        features = profile.active_features()
        assert features.shape == (6,)
        assert features[3].item() == pytest.approx(0.42)

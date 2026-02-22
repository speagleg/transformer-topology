"""Tests for WeightSpaceAnalyzer."""

import torch
import numpy as np
import pytest


class TestWeightSpaceAnalyzer:
    def test_effective_rank(self):
        from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
        analyzer = WeightSpaceAnalyzer()
        W = torch.randn(16, 16)
        result = analyzer.analyze_weight_matrix(W, layer_idx=0)
        assert result["effective_rank"] > 10

    def test_low_rank_matrix(self):
        from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
        analyzer = WeightSpaceAnalyzer()
        a = torch.randn(16, 2)
        b = torch.randn(2, 16)
        W = a @ b
        result = analyzer.analyze_weight_matrix(W, layer_idx=0)
        assert result["effective_rank"] < 5

    def test_condition_number(self):
        from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
        analyzer = WeightSpaceAnalyzer()
        W = torch.eye(8) * 2.0
        result = analyzer.analyze_weight_matrix(W, layer_idx=0)
        assert result["condition_number"] == pytest.approx(1.0, abs=0.1)

    def test_sv_persistence(self):
        from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
        analyzer = WeightSpaceAnalyzer()
        W = torch.randn(16, 16)
        result = analyzer.analyze_weight_matrix(W, layer_idx=0)
        assert "sv_persistence" in result
        assert isinstance(result["sv_persistence"], list)

    def test_spectral_gap(self):
        from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
        analyzer = WeightSpaceAnalyzer()
        W = torch.randn(8, 8)
        result = analyzer.analyze_weight_matrix(W, layer_idx=0)
        assert "spectral_gap" in result
        assert result["spectral_gap"] >= 0

    def test_training_dynamics_wasserstein(self):
        from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
        from src.topology_analyzer.profile import TopologicalProfile
        import time

        analyzer = WeightSpaceAnalyzer()

        def _make_profile(sv_diag):
            return TopologicalProfile(
                per_layer_persistence={}, per_layer_betti={},
                per_layer_spectral_gap={}, per_layer_hodge_ratios={},
                cross_layer_sheaf_gap=0.0,
                per_head_hodge_ratios={}, per_head_classification={},
                per_layer_head_sheaf_gap={},
                cross_layer_attention_wasserstein={},
                per_layer_effective_rank={0: 5.0},
                per_layer_condition_number={0: 10.0},
                per_layer_sv_persistence={0: [sv_diag]},
                cross_layer_weight_sheaf_gap=0.0,
                model_name="test", num_layers=1, num_heads=1,
                hidden_dim=16, seq_len=8, timestamp=time.time(),
            )

        p1 = _make_profile(np.array([[0.0, 0.5], [0.1, 0.8]]))
        p2 = _make_profile(np.array([[0.0, 0.6], [0.2, 0.9]]))
        result = analyzer.training_dynamics([p1, p2])
        assert "wasserstein_distances" in result
        assert len(result["wasserstein_distances"]) == 1

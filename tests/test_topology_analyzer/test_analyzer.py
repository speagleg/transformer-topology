"""Tests for the full TransformerTopologyAnalyzer integration."""

import torch
import torch.nn as nn
import pytest


def _make_transformer(num_layers=3, hidden_dim=32, num_heads=4):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim, nhead=num_heads, dim_feedforward=64,
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


class TestTransformerTopologyAnalyzer:
    def test_full_profile(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        model = _make_transformer(num_layers=3, hidden_dim=32, num_heads=4)
        analyzer = TransformerTopologyAnalyzer(
            model, model_name="test_transformer",
            num_heads=4, hidden_dim=32,
        )
        x = torch.randn(1, 8, 32)
        profile = analyzer.analyze(x)
        assert profile.model_name == "test_transformer"
        assert profile.num_layers == 3
        assert len(profile.per_layer_spectral_gap) == 3
        assert len(profile.per_layer_betti) == 3
        assert len(profile.per_layer_hodge_ratios) == 3

    def test_profile_with_attention(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        model = _make_transformer(num_layers=2, hidden_dim=32, num_heads=4)
        analyzer = TransformerTopologyAnalyzer(
            model, model_name="test", num_heads=4, hidden_dim=32,
            analyze_attention=True,
        )
        x = torch.randn(1, 8, 32)
        profile = analyzer.analyze(x)
        # Should have per-head analysis
        assert len(profile.per_head_hodge_ratios) > 0 or len(profile.per_head_classification) > 0

    def test_profile_with_weights(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        model = _make_transformer(num_layers=2, hidden_dim=32, num_heads=4)
        analyzer = TransformerTopologyAnalyzer(
            model, model_name="test", num_heads=4, hidden_dim=32,
            analyze_weights=True,
        )
        x = torch.randn(1, 8, 32)
        profile = analyzer.analyze(x)
        assert len(profile.per_layer_effective_rank) > 0
        assert len(profile.per_layer_condition_number) > 0

    def test_active_features_shape(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        model = _make_transformer(num_layers=2, hidden_dim=32, num_heads=4)
        analyzer = TransformerTopologyAnalyzer(
            model, model_name="test", num_heads=4, hidden_dim=32,
        )
        x = torch.randn(1, 8, 32)
        profile = analyzer.analyze(x)
        features = profile.active_features()
        assert features.shape == (6,)

    def test_summary_dict(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        model = _make_transformer(num_layers=2, hidden_dim=32, num_heads=4)
        analyzer = TransformerTopologyAnalyzer(
            model, model_name="test", num_heads=4, hidden_dim=32,
        )
        x = torch.randn(1, 8, 32)
        profile = analyzer.analyze(x)
        summary = profile.summary()
        assert "cross_layer_sheaf_gap" in summary

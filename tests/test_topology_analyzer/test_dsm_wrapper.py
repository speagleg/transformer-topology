"""Tests for DSMAnalysisWrapper."""

import torch
from src.llm.dsm import DistilledSemanticModel
from src.topology_analyzer.dsm_wrapper import DSMAnalysisWrapper
from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer


class TestDSMAnalysisWrapper:
    def test_wrapper_exposes_layers(self):
        dsm = DistilledSemanticModel(
            hidden_dim=32, num_heads=4, ff_dim=64,
            num_layers=3, cross_attn_layer=1,
        )
        wrapper = DSMAnalysisWrapper(dsm)
        assert hasattr(wrapper, 'layers')
        assert len(wrapper.layers) == 3

    def test_wrapper_forward(self):
        dsm = DistilledSemanticModel(
            hidden_dim=32, num_heads=4, ff_dim=64,
            num_layers=2, cross_attn_layer=1,
        )
        wrapper = DSMAnalysisWrapper(dsm)
        x = torch.randn(1, 8, 32)
        out = wrapper(x)
        assert out.shape == (1, 8, 32)

    def test_wrapper_with_analyzer(self):
        dsm = DistilledSemanticModel(
            hidden_dim=32, num_heads=4, ff_dim=64,
            num_layers=2, cross_attn_layer=1,
        )
        wrapper = DSMAnalysisWrapper(dsm)
        analyzer = TransformerTopologyAnalyzer(
            wrapper, model_name="dsm_test",
            num_heads=4, hidden_dim=32,
        )
        x = torch.randn(1, 8, 32)
        profile = analyzer.analyze(x)
        assert profile.model_name == "dsm_test"
        assert profile.num_layers == 2
        features = profile.active_features()
        assert features.shape == (6,)

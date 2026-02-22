"""Integration test: TransformerTopologyAnalyzer on our DSM."""

import torch
import torch.nn as nn
import pytest
from src.llm.dsm import DistilledSemanticModel


class _DSMWrapper(nn.Module):
    """Wrap DSM to accept a single (batch, seq, dim) input for the analyzer."""

    def __init__(self, dsm: DistilledSemanticModel):
        super().__init__()
        self.dsm = dsm
        # Expose .layers so hook manager finds them
        self.layers = dsm.layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim) -> squeeze batch for DSM
        prefix = x[0]  # (seq_len, hidden_dim)
        topo_memory = torch.zeros(1, prefix.shape[-1])  # dummy
        return self.dsm(prefix, topo_memory).unsqueeze(0)


class TestDSMIntegration:
    def test_analyze_dsm(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        dsm = DistilledSemanticModel(
            hidden_dim=32, num_heads=4, ff_dim=64,
            num_layers=3, cross_attn_layer=1,
        )
        wrapper = _DSMWrapper(dsm)
        analyzer = TransformerTopologyAnalyzer(
            wrapper, model_name="test_dsm",
            num_heads=4, hidden_dim=32,
            analyze_weights=True,
        )
        x = torch.randn(1, 8, 32)
        profile = analyzer.analyze(x)
        assert profile.model_name == "test_dsm"
        assert profile.num_layers == 3

    def test_dsm_hidden_states_captured(self):
        """Verify hooks work on DSM's .layers attribute."""
        from src.topology_analyzer.hooks import TransformerHookManager
        dsm = DistilledSemanticModel(
            hidden_dim=32, num_heads=4, ff_dim=64,
            num_layers=3, cross_attn_layer=1,
        )
        prefix = torch.randn(8, 32)
        topo_memory = torch.randn(10, 32)

        with TransformerHookManager(dsm) as manager:
            _ = dsm(prefix, topo_memory)
            hidden = manager.get_hidden_states()

        # DSM has .layers with 3 modules
        assert len(hidden) == 3
        for layer_idx, h in hidden.items():
            assert h.shape[0] == 8  # seq_len matches prefix length
            assert h.shape[1] == 32  # hidden_dim

    def test_active_features_from_dsm_profile(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        dsm = DistilledSemanticModel(
            hidden_dim=32, num_heads=4, ff_dim=64,
            num_layers=2, cross_attn_layer=1,
        )
        wrapper = _DSMWrapper(dsm)
        analyzer = TransformerTopologyAnalyzer(
            wrapper, model_name="dsm", num_heads=4, hidden_dim=32,
        )
        x = torch.randn(1, 8, 32)
        profile = analyzer.analyze(x)
        features = profile.active_features()
        assert features.shape == (6,)

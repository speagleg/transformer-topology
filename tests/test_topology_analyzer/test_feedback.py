"""Tests for TopologyFeedback and EmbeddingTopologyAdvisor."""

import time
import torch
import numpy as np
import pytest
from src.topology_analyzer.profile import TopologicalProfile


def _make_profile(**overrides):
    defaults = dict(
        per_layer_persistence={},
        per_layer_betti={},
        per_layer_spectral_gap={0: 0.5, 1: 0.3},
        per_layer_hodge_ratios={0: (0.6, 0.3, 0.1), 1: (0.4, 0.4, 0.2)},
        cross_layer_sheaf_gap=0.42,
        per_head_hodge_ratios={(0, 0): (0.3, 0.5, 0.2), (0, 1): (0.7, 0.2, 0.1)},
        per_head_classification={(0, 0): "curl", (0, 1): "gradient"},
        per_layer_head_sheaf_gap={0: 0.3},
        cross_layer_attention_wasserstein={},
        per_layer_effective_rank={0: 10.0, 1: 5.0},
        per_layer_condition_number={0: 50.0, 1: 200.0},
        per_layer_sv_persistence={},
        cross_layer_weight_sheaf_gap=0.35,
        model_name="test",
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        seq_len=16,
        timestamp=time.time(),
    )
    defaults.update(overrides)
    return TopologicalProfile(**defaults)


class TestTopologyFeedback:
    def test_profile_to_features_shape(self):
        from src.topology_analyzer.feedback import TopologyFeedback
        fb = TopologyFeedback()
        profile = _make_profile()
        features = fb.profile_to_features(profile)
        assert features.shape == (6,)

    def test_features_match_active_features(self):
        from src.topology_analyzer.feedback import TopologyFeedback
        fb = TopologyFeedback()
        profile = _make_profile()
        features = fb.profile_to_features(profile)
        active = profile.active_features()
        assert torch.allclose(features, active, atol=1e-5)


class TestEmbeddingTopologyAdvisor:
    def test_no_alerts_on_healthy_profile(self):
        from src.topology_analyzer.feedback import EmbeddingTopologyAdvisor
        advisor = EmbeddingTopologyAdvisor()
        profile = _make_profile(
            cross_layer_sheaf_gap=0.5,
            per_layer_effective_rank={0: 15.0, 1: 14.0},
        )
        alerts = advisor.check_alerts(profile)
        assert len(alerts) == 0

    def test_alert_on_sheaf_gap_collapse(self):
        from src.topology_analyzer.feedback import EmbeddingTopologyAdvisor
        advisor = EmbeddingTopologyAdvisor()
        profile = _make_profile(cross_layer_sheaf_gap=0.001)
        alerts = advisor.check_alerts(profile)
        assert any("sheaf" in a.lower() or "coherence" in a.lower() for a in alerts)

    def test_alert_on_rank_collapse(self):
        from src.topology_analyzer.feedback import EmbeddingTopologyAdvisor
        advisor = EmbeddingTopologyAdvisor()
        profile = _make_profile(per_layer_effective_rank={0: 1.5, 1: 1.2})
        alerts = advisor.check_alerts(profile)
        assert any("rank" in a.lower() for a in alerts)

    def test_alert_on_high_attention_curl(self):
        from src.topology_analyzer.feedback import EmbeddingTopologyAdvisor
        advisor = EmbeddingTopologyAdvisor()
        profile = _make_profile(
            per_head_hodge_ratios={(0, 0): (0.1, 0.8, 0.1), (0, 1): (0.1, 0.7, 0.2)},
        )
        alerts = advisor.check_alerts(profile)
        assert any("curl" in a.lower() or "circular" in a.lower() for a in alerts)

    def test_suggest_architecture_from_history(self):
        from src.topology_analyzer.feedback import EmbeddingTopologyAdvisor
        advisor = EmbeddingTopologyAdvisor()
        # Create history with declining sheaf gap
        profiles = []
        for i in range(5):
            p = _make_profile(cross_layer_sheaf_gap=0.5 - i * 0.1)
            profiles.append(p)
        suggestions = advisor.suggest_architecture_changes(profiles)
        assert isinstance(suggestions, list)

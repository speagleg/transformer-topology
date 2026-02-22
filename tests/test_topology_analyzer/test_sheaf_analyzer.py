"""Tests for CrossLayerSheafAnalyzer."""

import torch
import pytest


class TestCrossLayerSheafAnalyzer:
    def test_embedding_coherence_smooth(self):
        from src.topology_analyzer.sheaf_analyzer import CrossLayerSheafAnalyzer
        analyzer = CrossLayerSheafAnalyzer(feature_dim=8, num_layers=4)
        base = torch.randn(8)
        features = {i: base + 0.01 * i * torch.randn(8) for i in range(4)}
        gap = analyzer.compute_coherence(features)
        assert isinstance(gap, float)
        assert gap >= 0

    def test_embedding_coherence_discontinuous(self):
        from src.topology_analyzer.sheaf_analyzer import CrossLayerSheafAnalyzer
        analyzer = CrossLayerSheafAnalyzer(feature_dim=8, num_layers=4)
        features = {i: torch.randn(8) * 10 for i in range(4)}
        gap = analyzer.compute_coherence(features)
        assert isinstance(gap, float)

    def test_head_redundancy(self):
        from src.topology_analyzer.sheaf_analyzer import CrossLayerSheafAnalyzer
        analyzer = CrossLayerSheafAnalyzer(feature_dim=8, num_layers=4)
        head_features = {h: torch.randn(8) for h in range(4)}
        gap = analyzer.compute_coherence(head_features)
        assert isinstance(gap, float)

    def test_two_layers(self):
        from src.topology_analyzer.sheaf_analyzer import CrossLayerSheafAnalyzer
        analyzer = CrossLayerSheafAnalyzer(feature_dim=4, num_layers=2)
        features = {0: torch.randn(4), 1: torch.randn(4)}
        gap = analyzer.compute_coherence(features)
        assert isinstance(gap, float)
        assert gap >= 0

    def test_weight_subspace_coherence(self):
        from src.topology_analyzer.sheaf_analyzer import CrossLayerSheafAnalyzer
        analyzer = CrossLayerSheafAnalyzer(feature_dim=8, num_layers=3)
        features = {i: torch.randn(8) for i in range(3)}
        gap = analyzer.compute_coherence(features)
        assert gap >= 0

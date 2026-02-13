import torch
import pytest

from src.benchmarks.diagnostics import (
    DiagnosticCollector,
    _entropy,
    _gini,
    _mean,
    _std,
    _histogram,
)
from src.benchmarks.run_comparison import HierarchicalMultiHopModel
from src.benchmarks.benchmark_dataset import BenchmarkDataset


class TestHelperFunctions:
    def test_entropy_uniform(self):
        """Entropy is maximal for uniform 0.5 distribution."""
        probs = torch.full((10,), 0.5)
        ent = _entropy(probs)
        assert ent.item() > 0.6  # close to ln(2) ≈ 0.693

    def test_entropy_peaked(self):
        """Entropy is low for peaked distribution."""
        probs = torch.tensor([0.99, 0.99, 0.99])
        ent = _entropy(probs)
        assert ent.item() < 0.1

    def test_gini_uniform(self):
        """Gini is 0 for perfectly equal distribution."""
        values = torch.ones(10)
        gini = _gini(values)
        assert abs(gini.item()) < 0.05

    def test_gini_concentrated(self):
        """Gini is high for concentrated distribution."""
        values = torch.zeros(10)
        values[0] = 100.0
        gini = _gini(values)
        assert gini.item() > 0.5

    def test_mean(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0
        assert _mean([]) == 0.0

    def test_std(self):
        assert _std([1.0]) == 0.0
        assert _std([1.0, 3.0]) > 0.0

    def test_histogram(self):
        h = _histogram([1, 2, 2, 3, 3, 3])
        assert h == {1: 1, 2: 2, 3: 3}


class TestDiagnosticCollector:
    def test_collect_returns_records(self):
        """DiagnosticCollector produces one record per sample."""
        model = HierarchicalMultiHopModel(
            embedding_dim=16, gnn_hidden=32,
            gnn_spatial_layers=1, gnn_spectral_layers=1,
            max_freqs=4, tat_layers=1,
            tat_spatial_heads=1, tat_spectral_heads=1,
            tat_ff_dim=32, max_classes=3,
            max_iterations=2, convergence_threshold=0.01,
            use_wave_dynamics=True, use_higher_order=True,
            use_topological_pe=False, use_structural_features=False,
        )
        ds = BenchmarkDataset(
            num_samples=5, task_type="hodge_class",
            n_nodes=10, embedding_dim=16,
        )
        collector = DiagnosticCollector()
        records = collector.collect(model, ds, torch.device('cpu'))
        assert len(records) == 5

        for r in records:
            assert 'num_iterations' in r
            assert 'harmonic_energies' in r
            assert 'frequency_gate_mean' in r
            assert 'spatial_focus_gini' in r
            assert 'confidence_mean' in r
            assert 'diffusion_time' in r
            assert 'wave_damping' in r

    def test_summarize(self):
        """Summary aggregates across records."""
        model = HierarchicalMultiHopModel(
            embedding_dim=16, gnn_hidden=32,
            gnn_spatial_layers=1, gnn_spectral_layers=1,
            max_freqs=4, tat_layers=1,
            tat_spatial_heads=1, tat_spectral_heads=1,
            tat_ff_dim=32, max_classes=3,
            max_iterations=2, convergence_threshold=0.01,
            use_wave_dynamics=True, use_higher_order=True,
        )
        ds = BenchmarkDataset(
            num_samples=5, task_type="hodge_class",
            n_nodes=10, embedding_dim=16,
        )
        collector = DiagnosticCollector()
        collector.collect(model, ds, torch.device('cpu'))
        summary = collector.summarize()

        assert 'num_iterations' in summary
        assert 'mean' in summary['num_iterations']
        assert 'histogram' in summary['num_iterations']
        assert 'harmonic_energy' in summary
        assert 'frequency_gate' in summary
        assert 'spatial_focus' in summary
        assert 'confidence' in summary
        assert 'diffusion_time' in summary

    def test_empty_collector(self):
        """Summarize on empty collector returns empty dict."""
        collector = DiagnosticCollector()
        assert collector.summarize() == {}

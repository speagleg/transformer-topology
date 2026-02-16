"""Integration tests for multi-filter ensemble pipeline."""

import torch
import pytest

from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop
from src.benchmarks.run_comparison import HierarchicalMultiHopModel
from src.benchmarks.diagnostics import DiagnosticCollector


def make_graph(dim: int = 32, n: int = 8) -> CellComplex:
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "node") for _ in range(n)]
    for i in range(n - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "edge")
    return cc


class TestEnsemblePipeline:
    def test_full_pipeline_ensemble_mode(self):
        """End-to-end: CellComplex -> ExecutiveLoop with ensemble -> output."""
        cc = make_graph(dim=32, n=8)
        loop = ExecutiveReasoningLoop(
            embedding_dim=32, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=2,
            max_freqs=8, tat_layers=2,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=128, max_iterations=3,
            use_wave_dynamics=True,
            use_higher_order=True,
            wave_mode='ensemble',
            wave_filter_kwargs={
                'filter_types': ['chebyshev', 'wave_cosine', 'heat'],
                'include_identity': True,
            },
            wave_strength_gate=True,
            wave_use_neural_ode=False,
        )
        output, num_iters, diagnostics = loop(cc)
        assert output.shape == (8, 32)
        assert num_iters >= 1
        assert 'control_signals' in diagnostics
        # Ensemble mode should produce filter_weights
        cs = diagnostics['control_signals'][0]
        assert cs.filter_weights is not None
        assert cs.filter_weights.shape == (4,)  # 3 filters + identity

    def test_ensemble_backward_compat(self):
        """wave_mode='spectral' still works identically (no filter_weights)."""
        cc = make_graph(dim=32, n=6)
        loop = ExecutiveReasoningLoop(
            embedding_dim=32, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=2,
            max_freqs=8, tat_layers=2,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=128, max_iterations=2,
            use_wave_dynamics=True,
            wave_mode='spectral',
        )
        output, num_iters, diagnostics = loop(cc)
        assert output.shape == (6, 32)
        cs = diagnostics['control_signals'][0]
        assert cs.filter_weights is None

    def test_ensemble_with_hierarchical_model(self):
        """HierarchicalMultiHopModel works with ensemble wave config."""
        cc = make_graph(dim=32, n=8)
        wave_config = {
            'wave_mode': 'ensemble',
            'filter_types': ['chebyshev', 'heat'],
            'include_identity': True,
            'laplacian_dim': 0,
            'wave_strength_gate': True,
            'use_neural_ode': False,
        }
        model = HierarchicalMultiHopModel(
            embedding_dim=32, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=2,
            max_freqs=8, tat_layers=2,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=128, max_classes=11,
            max_iterations=2, convergence_threshold=0.1,
            use_wave_dynamics=True, use_higher_order=True,
            wave_config=wave_config,
        )
        logits = model(cc, 0, 3)
        assert logits.shape == (11,)

    def test_ensemble_diagnostics_capture(self):
        """DiagnosticCollector captures filter_weights stats in ensemble mode."""
        dim = 32
        wave_config = {
            'wave_mode': 'ensemble',
            'filter_types': ['chebyshev', 'heat'],
            'include_identity': True,
            'wave_strength_gate': True,
            'use_neural_ode': False,
        }
        model = HierarchicalMultiHopModel(
            embedding_dim=dim, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=2,
            max_freqs=8, tat_layers=2,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=128, max_classes=11,
            max_iterations=2, convergence_threshold=0.1,
            use_wave_dynamics=True, use_higher_order=True,
            wave_config=wave_config,
        )
        # Create a tiny dataset (list of tuples)
        dataset = []
        for _ in range(3):
            cc = make_graph(dim=dim, n=6)
            dataset.append((cc, 0, 2, 3))

        collector = DiagnosticCollector()
        records = collector.collect(model, dataset, torch.device('cpu'))
        assert len(records) == 3
        assert 'filter_weights' in records[0]
        assert 'filter_weights_entropy' in records[0]
        # 2 filters + identity = 3 weights
        assert len(records[0]['filter_weights']) == 3

        summary = collector.summarize()
        assert 'filter_weights' in summary
        assert 'entropy_mean' in summary['filter_weights']
        assert 'avg_weights' in summary['filter_weights']
        assert len(summary['filter_weights']['avg_weights']) == 3

    def test_ensemble_gradient_flow(self):
        """Gradients flow through ensemble filter weights to GNN executive."""
        cc = make_graph(dim=32, n=6)
        loop = ExecutiveReasoningLoop(
            embedding_dim=32, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=2,
            max_freqs=8, tat_layers=2,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=128, max_iterations=1,
            use_wave_dynamics=True,
            wave_mode='ensemble',
            wave_filter_kwargs={
                'filter_types': ['heat', 'chebyshev'],
                'include_identity': True,
            },
            wave_use_neural_ode=False,
        )
        output, _, _ = loop(cc)
        loss = output.sum()
        loss.backward()
        # Filter weights head should receive gradients
        head = loop.gnn_executive.control_head.filter_weights_head
        assert head.weight.grad is not None

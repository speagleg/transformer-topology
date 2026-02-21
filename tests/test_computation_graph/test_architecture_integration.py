import torch
import torch.nn as nn
import pytest
from src.computation_graph.capture import ComputationGraphCapture
from src.computation_graph.diagnostics import analyze_hodge, TopologicalDiagnostics
from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.graph_generators import random_graph
from src.benchmarks.graph_convert import nx_to_cell_complex


def _make_small_model():
    """Build a minimal HierarchicalMultiHopModel for testing."""
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel
    return HierarchicalMultiHopModel(
        embedding_dim=8,
        gnn_hidden=16,
        gnn_spatial_layers=1,
        gnn_spectral_layers=1,
        max_freqs=4,
        tat_layers=1,
        tat_spatial_heads=1,
        tat_spectral_heads=1,
        tat_ff_dim=32,
        max_classes=3,
        max_iterations=2,
        convergence_threshold=0.1,
        use_wave_dynamics=True,
        use_higher_order=False,
        use_topological_pe=False,
        use_structural_features=False,
    )


def _make_sample():
    """Generate a small graph sample."""
    G = random_graph(10, topology='ba')
    cc, _node_map = nx_to_cell_complex(G, embedding_dim=8)
    # Pick valid query/target nodes within the cell complex
    n = cc.num_cells(0)
    query = 0
    target = min(3, n - 1)
    answer = min(2, target)  # answer must be < max_classes (3)
    return cc, query, target, answer, {}


class TestArchitectureIntegration:
    def test_capture_works_on_hierarchical_model(self):
        """ComputationGraphCapture should work with the full architecture."""
        model = _make_small_model()
        cc_input, query, target, answer, meta = _make_sample()
        cc_clone = cc_input.clone()
        with ComputationGraphCapture(model) as cap:
            logits = model(cc_clone, query, target, meta)
            loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), torch.tensor([answer]))
            loss.backward()
        comp_cc = cap.to_cell_complex()
        # Should have many operations (GNN layers, TAT, wave dynamics, classifier)
        assert comp_cc.num_cells(0) >= 5

    def test_hodge_on_architecture_graph(self):
        """Hodge analysis should produce valid ratios for our architecture."""
        model = _make_small_model()
        cc_input, query, target, answer, meta = _make_sample()
        cc_clone = cc_input.clone()
        with ComputationGraphCapture(model) as cap:
            logits = model(cc_clone, query, target, meta)
            loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), torch.tensor([answer]))
            loss.backward()
        comp_cc = cap.to_cell_complex()
        grad_r, curl_r, harm_r = analyze_hodge(comp_cc)
        total = grad_r + curl_r + harm_r
        assert total == 0.0 or abs(total - 1.0) < 1e-4

    def test_multiple_inputs_work(self):
        """Capture should handle multiple different inputs."""
        model = _make_small_model()
        for _ in range(2):
            cc_input, query, target, answer, meta = _make_sample()
            cc_clone = cc_input.clone()
            with ComputationGraphCapture(model) as cap:
                logits = model(cc_clone, query, target, meta)
                loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), torch.tensor([answer]))
                loss.backward()
            comp_cc = cap.to_cell_complex()
            assert comp_cc.num_cells(0) >= 3

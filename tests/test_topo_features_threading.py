"""Tests that topo_features threads through the full model stack."""

import torch
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.executive import GNNExecutive
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop
from src.benchmarks.run_comparison import HierarchicalMultiHopModel
from src.benchmarks.multi_hop import MultiHopDataset


def _make_cc(n_nodes=8, emb_dim=16):
    """Create a small CellComplex for testing."""
    ds = MultiHopDataset(
        num_samples=1, min_hops=1, max_hops=3,
        num_distractors=2, embedding_dim=emb_dim,
    )
    return ds[0][0]  # cc


class TestGNNExecutiveTopoFeatures:
    def test_forward_with_control_no_topo_features(self):
        cc = _make_cc()
        gnn = GNNExecutive(
            embedding_dim=16, hidden_dim=32,
            num_spatial_layers=1, num_spectral_layers=1,
            max_freqs=8, produce_control_signals=True,
        )
        fused, edge_out, control = gnn.forward_with_control(cc)
        assert fused.shape[1] == 16
        assert control.frequency_gate.shape == (8,)

    def test_forward_with_control_with_topo_features(self):
        cc = _make_cc()
        gnn = GNNExecutive(
            embedding_dim=16, hidden_dim=32,
            num_spatial_layers=1, num_spectral_layers=1,
            max_freqs=8, produce_control_signals=True,
            use_topo_feedback=True,
            use_embedding_topo_feedback=True,
        )
        topo_feat = torch.randn(6)
        fused, edge_out, control = gnn.forward_with_control(
            cc, topo_features=topo_feat,
        )
        assert fused.shape[1] == 16
        assert control.frequency_gate.shape == (8,)


class TestExecutiveLoopTopoFeatures:
    def test_forward_no_topo_features(self):
        cc = _make_cc()
        loop = ExecutiveReasoningLoop(
            embedding_dim=16, gnn_hidden=32,
            gnn_spatial_layers=1, gnn_spectral_layers=1,
            max_freqs=8, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=32, max_iterations=2,
            use_wave_dynamics=False,
        )
        output, n_iters, diag = loop(cc)
        assert output.shape[1] == 16

    def test_forward_with_topo_features(self):
        cc = _make_cc()
        loop = ExecutiveReasoningLoop(
            embedding_dim=16, gnn_hidden=32,
            gnn_spatial_layers=1, gnn_spectral_layers=1,
            max_freqs=8, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=32, max_iterations=2,
            use_wave_dynamics=False,
            use_topo_feedback=True,
            use_embedding_topo_feedback=True,
        )
        topo_feat = torch.randn(6)
        output, n_iters, diag = loop(cc, topo_features=topo_feat)
        assert output.shape[1] == 16


class TestHierarchicalModelTopoFeatures:
    def test_forward_no_topo_features(self):
        ds = MultiHopDataset(
            num_samples=1, min_hops=1, max_hops=3,
            num_distractors=2, embedding_dim=16,
        )
        cc, query, target, answer = ds[0][:4]
        model = HierarchicalMultiHopModel(
            embedding_dim=16, gnn_hidden=32,
            gnn_spatial_layers=1, gnn_spectral_layers=1,
            max_freqs=8, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=32, max_classes=6,
            max_iterations=2, convergence_threshold=0.1,
            use_wave_dynamics=False, use_higher_order=False,
        )
        logits = model(cc, query, target)
        assert logits.shape == (6,)

    def test_forward_with_topo_features(self):
        ds = MultiHopDataset(
            num_samples=1, min_hops=1, max_hops=3,
            num_distractors=2, embedding_dim=16,
        )
        cc, query, target, answer = ds[0][:4]
        model = HierarchicalMultiHopModel(
            embedding_dim=16, gnn_hidden=32,
            gnn_spatial_layers=1, gnn_spectral_layers=1,
            max_freqs=8, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=32, max_classes=6,
            max_iterations=2, convergence_threshold=0.1,
            use_wave_dynamics=False, use_higher_order=False,
            use_topo_feedback=True,
            use_embedding_topo_feedback=True,
        )
        topo_feat = torch.randn(6)
        logits = model(cc, query, target, topo_features=topo_feat)
        assert logits.shape == (6,)

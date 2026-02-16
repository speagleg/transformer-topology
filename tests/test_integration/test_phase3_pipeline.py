"""Integration tests for the Phase 3 temporal/causal reasoning pipeline."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.phase3_model import TemporalReasoningModel
from src.benchmarks.temporal_tasks import (
    TemporalDataset,
    generate_propagation_delay_task,
    generate_interference_task,
)
from src.gnn_executive.control_head import ControlSignal


def make_model(embedding_dim=16, max_classes=10, use_wave=True):
    return TemporalReasoningModel(
        embedding_dim=embedding_dim, gnn_hidden=32,
        gnn_spatial_layers=1, gnn_spectral_layers=1,
        max_freqs=4, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2,
        tat_ff_dim=32, max_classes=max_classes,
        max_iterations=2, convergence_threshold=0.5,
        use_wave_dynamics=use_wave,
    )


def make_chain(dim=16, length=6):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "node") for _ in range(length)]
    for i in range(length - 1):
        emb = torch.randn(dim)
        emb[0] = 1.0  # delay = 1
        cc.add_1_cell(nodes[i], nodes[i + 1], emb, "temporal_edge")
    return cc


class TestEndToEndForward:
    def test_forward_with_wave(self):
        model = make_model(use_wave=True)
        cc = make_chain()
        logits = model(cc, 0, 5)
        assert logits.shape == (10,)

    def test_forward_without_wave(self):
        model = make_model(use_wave=False)
        cc = make_chain()
        logits = model(cc, 0, 5)
        assert logits.shape == (10,)

    def test_interference_model(self):
        model = make_model(max_classes=2, use_wave=True)
        cc = make_chain()
        logits = model(cc, 0, 3)
        assert logits.shape == (2,)


class TestGradientFlow:
    def test_gradient_through_full_pipeline(self):
        model = make_model(use_wave=True)
        cc = make_chain()
        logits = model(cc, 0, 5)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([3]))
        loss.backward()
        grad_count = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "No parameters received gradients"

    def test_gradient_reaches_gnn_tat_wave(self):
        model = make_model(use_wave=True)
        cc = make_chain()
        logits = model(cc, 0, 5)
        loss = logits.sum()
        loss.backward()

        gnn_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.executive_loop.gnn_executive.parameters()
            if p.requires_grad
        )
        tat_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.executive_loop.tat.parameters()
            if p.requires_grad
        )
        assert gnn_grads, "No gradients reached GNN executive"
        assert tat_grads, "No gradients reached TAT"


class TestTemporalDatasetIntegration:
    def test_propagation_delay_with_model(self):
        model = make_model(max_classes=10)
        ds = TemporalDataset(num_samples=3, task_type="propagation_delay",
                             embedding_dim=16, n_nodes=6, max_delay=10)
        for i in range(len(ds)):
            cc, q, t, answer = ds[i]
            logits = model(cc, q, t)
            assert logits.shape == (10,)
            assert 0 <= answer < 10

    def test_blocking_with_model(self):
        model = make_model(max_classes=10)
        ds = TemporalDataset(num_samples=3, task_type="blocking",
                             embedding_dim=16, n_nodes=6, max_delay=10)
        for i in range(len(ds)):
            cc, q, t, answer = ds[i]
            logits = model(cc, q, t)
            assert logits.shape == (10,)

    def test_interference_with_model(self):
        model = make_model(max_classes=2)
        ds = TemporalDataset(num_samples=3, task_type="interference",
                             embedding_dim=16, n_nodes=6)
        for i in range(len(ds)):
            cc, q, t, answer = ds[i]
            logits = model(cc, q, t)
            assert logits.shape == (2,)
            assert answer in (0, 1)


class TestTrainingStep:
    def test_single_training_step(self):
        model = make_model(max_classes=10)
        cc, q, t, answer = generate_propagation_delay_task(6, 16, max_delay=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        logits = model(cc, q, t)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([answer]))
        loss.backward()
        optimizer.step()
        assert loss.item() > 0


class TestCurlPreservationInModel:
    """Verify that hodge features reflect the initial curl signal, not zeroed-out post-loop."""

    def test_hodge_features_preserve_curl(self):
        """When edge embeddings have a curl signal, _compute_hodge_features
        should report nonzero curl (via initial_edge_embs path)."""
        model = make_model(max_classes=3, use_wave=False)
        # Build a graph with a triangle
        dim = 16
        cc = CellComplex(embedding_dim=dim)
        for i in range(4):
            cc.add_0_cell(torch.randn(dim), "source" if i == 0 else "node")
        e0 = cc.add_1_cell(0, 1, torch.randn(dim), "edge")
        e1 = cc.add_1_cell(1, 2, torch.randn(dim), "edge")
        e2 = cc.add_1_cell(0, 2, torch.randn(dim), "edge")
        e3 = cc.add_1_cell(2, 3, torch.randn(dim), "edge")
        cc.add_2_cell([e0, e1, e2], torch.randn(dim), "face")

        # Inject curl signal
        B2 = cc.boundary_operator(2)
        w = torch.randn(cc.num_cells(2))
        curl_signal = B2 @ w
        if curl_signal.norm() < 1e-6:
            pytest.skip("Degenerate B2")
        curl_signal = curl_signal / curl_signal.norm()
        emb = cc.get_embeddings(1).clone()
        emb[:, 0] = curl_signal
        cc.set_embeddings(1, emb)

        # Run forward - hodge features should use initial edge embeddings
        logits = model(cc, 0, 3)
        assert logits.shape == (3,)

        # Verify the initial_edge_embs path: manually call _compute_hodge_features
        initial_embs = emb.clone()
        hodge_feat = model._compute_hodge_features(cc, initial_edge_embs=initial_embs)
        curl_feat = hodge_feat[1].item()
        assert curl_feat > 1e-4, (
            f"Curl feature should be nonzero for curl-dominated signal, got {curl_feat:.2e}"
        )


class TestPhase2BackwardCompat:
    def test_phase2_model_still_works(self):
        """Phase 2 model uses the old ReasoningLoop — should still work."""
        from src.benchmarks.phase2_model import TopologicalReasoningModel
        model = TopologicalReasoningModel(
            embedding_dim=16, gnn_hidden=32,
            gnn_spatial_layers=1, gnn_spectral_layers=1,
            max_freqs=4, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=32, max_classes=6,
            max_iterations=2, convergence_threshold=0.5,
        )
        cc = CellComplex(embedding_dim=16)
        for _ in range(4):
            cc.add_0_cell(torch.randn(16), "node")
        for i in range(3):
            cc.add_1_cell(i, i + 1, torch.randn(16), "edge")
        logits = model(cc, 0, 3)
        assert logits.shape == (6,)

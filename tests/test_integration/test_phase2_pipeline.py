"""Integration tests for the Phase 2 topological reasoning pipeline."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.phase2_model import TopologicalReasoningModel
from src.benchmarks.topological_tasks import (
    TopologicalDataset,
    auto_fill_triangles,
    generate_cycle_detection_task,
)
from src.spectral.decomposition import hodge_decomposition
from src.spectral.laplacian import hodge_laplacian_1


def make_model(embedding_dim=16, max_classes=6):
    return TopologicalReasoningModel(
        embedding_dim=embedding_dim, gnn_hidden=32,
        gnn_spatial_layers=1, gnn_spectral_layers=1,
        max_freqs=4, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2,
        tat_ff_dim=32, max_classes=max_classes,
        max_iterations=2, convergence_threshold=0.5,
    )


def make_complex_with_face(dim=16):
    cc = CellComplex(embedding_dim=dim)
    n0 = cc.add_0_cell(torch.randn(dim), "node")
    n1 = cc.add_0_cell(torch.randn(dim), "node")
    n2 = cc.add_0_cell(torch.randn(dim), "node")
    n3 = cc.add_0_cell(torch.randn(dim), "node")
    e0 = cc.add_1_cell(n0, n1, torch.randn(dim), "edge")
    e1 = cc.add_1_cell(n1, n2, torch.randn(dim), "edge")
    e2 = cc.add_1_cell(n2, n0, torch.randn(dim), "edge")
    cc.add_1_cell(n2, n3, torch.randn(dim), "edge")
    cc.add_2_cell([e0, e1, e2], torch.randn(dim))
    return cc


class TestEndToEndForward:
    def test_forward_with_2_cells(self):
        model = make_model()
        cc = make_complex_with_face()
        logits = model(cc, 0, 3)
        assert logits.shape == (6,)

    def test_forward_without_2_cells(self):
        model = make_model()
        cc = CellComplex(embedding_dim=16)
        for _ in range(4):
            cc.add_0_cell(torch.randn(16), "node")
        for i in range(3):
            cc.add_1_cell(i, i + 1, torch.randn(16), "edge")
        logits = model(cc, 0, 3)
        assert logits.shape == (6,)


class TestGradientFlow:
    def test_gradient_through_hodge_path(self):
        model = make_model()
        cc = make_complex_with_face()
        logits = model(cc, 0, 3)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([1]))
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0


class TestChainComplexPreservation:
    def test_chain_complex_after_training_step(self):
        model = make_model()
        cc, q, t, answer = generate_cycle_detection_task(8, 16, has_cycle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        logits = model(cc, q, t)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([answer]))
        loss.backward()
        optimizer.step()
        # Chain complex property should still hold
        if cc.num_cells(2) > 0:
            assert cc.verify_chain_complex()


class TestTopologicalDatasetIntegration:
    def test_cycle_detection_with_model(self):
        model = make_model(max_classes=2)
        ds = TopologicalDataset(num_samples=3, task_type="cycle_detection", embedding_dim=16)
        for i in range(len(ds)):
            cc, q, t, answer = ds[i]
            logits = model(cc, q, t)
            assert logits.shape == (2,)

    def test_betti_number_with_model(self):
        model = make_model(max_classes=6)
        ds = TopologicalDataset(num_samples=3, task_type="betti_number",
                                embedding_dim=16, max_beta=5)
        for i in range(len(ds)):
            cc, q, t, answer = ds[i]
            logits = model(cc, q, t)
            assert logits.shape == (6,)

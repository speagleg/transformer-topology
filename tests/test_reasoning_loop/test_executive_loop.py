"""Tests for the hierarchical executive reasoning loop."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop
from src.gnn_executive.control_head import ControlSignal


def make_chain(dim=16, length=5):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "node") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "edge")
    return cc


DIM = 16
LOOP_KWARGS = dict(
    embedding_dim=DIM, gnn_hidden=32,
    gnn_spatial_layers=1, gnn_spectral_layers=1,
    max_freqs=4, tat_layers=1,
    tat_spatial_heads=2, tat_spectral_heads=2,
    tat_ff_dim=32, max_iterations=3,
    convergence_threshold=0.5,
)


class TestOutputShape:
    def test_basic_output_shape(self):
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS)
        cc = make_chain(dim=DIM, length=5)
        out, num_iters, diag = loop(cc)
        assert out.shape == (5, DIM)
        assert 1 <= num_iters <= 3

    def test_different_graph_sizes(self):
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS)
        for length in [3, 6, 10]:
            cc = make_chain(dim=DIM, length=length)
            out, num_iters, diag = loop(cc)
            assert out.shape == (length, DIM)


class TestDiagnostics:
    def test_diagnostics_populated(self):
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS)
        cc = make_chain(dim=DIM, length=5)
        _, num_iters, diag = loop(cc)
        # harmonic_energies has initial + one per iteration
        assert len(diag['harmonic_energies']) == num_iters + 1
        assert len(diag['convergence_deltas']) == num_iters
        assert len(diag['control_signals']) == num_iters
        assert all(isinstance(cs, ControlSignal) for cs in diag['control_signals'])

    def test_convergence_deltas_are_finite(self):
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS)
        cc = make_chain(dim=DIM, length=5)
        _, _, diag = loop(cc)
        for d in diag['convergence_deltas']:
            assert isinstance(d, float)
            assert not torch.tensor(d).isnan()
            assert not torch.tensor(d).isinf()


class TestWaveIntegration:
    def test_with_wave_dynamics(self):
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS, use_wave_dynamics=True)
        cc = make_chain(dim=DIM, length=5)
        out, _, _ = loop(cc)
        assert out.shape == (5, DIM)

    def test_without_wave_dynamics(self):
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS, use_wave_dynamics=False)
        cc = make_chain(dim=DIM, length=5)
        out, _, _ = loop(cc)
        assert out.shape == (5, DIM)


class TestGradientFlow:
    def test_gradient_through_full_pipeline(self):
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS, use_wave_dynamics=True)
        cc = make_chain(dim=DIM, length=5)
        out, _, _ = loop(cc)
        loss = out.sum()
        loss.backward()
        grad_count = sum(
            1 for p in loop.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "No parameters received gradients"

    def test_gradient_reaches_gnn_and_tat(self):
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS, use_wave_dynamics=False)
        cc = make_chain(dim=DIM, length=5)
        out, _, _ = loop(cc)
        loss = out.sum()
        loss.backward()

        gnn_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in loop.gnn_executive.parameters()
            if p.requires_grad
        )
        tat_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in loop.tat.parameters()
            if p.requires_grad
        )
        assert gnn_grads, "No gradients reached GNN executive"
        assert tat_grads, "No gradients reached TAT"


class TestControlAffectsOutput:
    def test_output_varies_across_runs(self):
        """Different random CCs should produce different outputs."""
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS)
        cc1 = make_chain(dim=DIM, length=5)
        cc2 = make_chain(dim=DIM, length=5)
        out1, _, _ = loop(cc1)
        out2, _, _ = loop(cc2)
        assert not torch.allclose(out1, out2, atol=1e-4)

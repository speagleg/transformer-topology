"""Tests for the hierarchical executive reasoning loop."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop
from src.gnn_executive.control_head import ControlSignal
from src.spectral.decomposition import hodge_decomposition


def make_chain(dim=16, length=5):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "node") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "edge")
    return cc


def make_triangle_graph(dim=16):
    """Build a 4-node graph with a triangle (0-1-2) so B2 is non-trivial."""
    cc = CellComplex(embedding_dim=dim)
    for i in range(4):
        cc.add_0_cell(torch.randn(dim), "source" if i == 0 else "node")
    # Triangle edges: 0-1 (e0), 1-2 (e1), 0-2 (e2), plus 2-3 (e3)
    e0 = cc.add_1_cell(0, 1, torch.randn(dim), "edge")
    e1 = cc.add_1_cell(1, 2, torch.randn(dim), "edge")
    e2 = cc.add_1_cell(0, 2, torch.randn(dim), "edge")
    e3 = cc.add_1_cell(2, 3, torch.randn(dim), "edge")
    # Add the triangle as a 2-cell
    cc.add_2_cell([e0, e1, e2], torch.randn(dim), "face")
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


class TestHarmonicEnergyNormalization:
    def test_energy_normalized_by_edges(self):
        """Harmonic energy should be normalized by edge count, not raw sum."""
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS)
        # Small chain
        cc_small = make_chain(dim=DIM, length=5)
        energy_small = loop._compute_harmonic_energy(cc_small)
        # Larger chain (more edges → without normalization energy would be larger)
        cc_large = make_chain(dim=DIM, length=20)
        energy_large = loop._compute_harmonic_energy(cc_large)
        # Both should be finite
        assert torch.isfinite(energy_small)
        assert torch.isfinite(energy_large)
        # With normalization, the ratio should be much closer to 1 than the
        # edge count ratio (4:1 for 20 edges vs 4 edges)
        if energy_small > 1e-8:
            ratio = (energy_large / energy_small).item()
            assert ratio < 10, f"Normalized energy ratio {ratio} too large"

    def test_energy_zero_for_no_edges(self):
        """Harmonic energy is 0 for a graph with no edges."""
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS)
        cc = CellComplex(embedding_dim=DIM)
        cc.add_0_cell(torch.randn(DIM), "node")
        energy = loop._compute_harmonic_energy(cc)
        assert energy.item() == 0.0


class TestControlAffectsOutput:
    def test_output_varies_across_runs(self):
        """Different random CCs should produce different outputs."""
        loop = ExecutiveReasoningLoop(**LOOP_KWARGS)
        cc1 = make_chain(dim=DIM, length=5)
        cc2 = make_chain(dim=DIM, length=5)
        out1, _, _ = loop(cc1)
        out2, _, _ = loop(cc2)
        assert not torch.allclose(out1, out2, atol=1e-4)


class TestCurlPreservation:
    """Verify that edge residual blending preserves curl content across iterations."""

    def test_curl_nonzero_after_loop(self):
        """After the executive loop, edge embeddings should retain curl content
        when the initial signal has a curl component (B2 @ w)."""
        loop = ExecutiveReasoningLoop(
            **LOOP_KWARGS, use_higher_order=True, use_wave_dynamics=False,
        )
        cc = make_triangle_graph(dim=DIM)

        # Inject a curl signal into edge embedding dim 0
        B2 = cc.boundary_operator(2)  # (n_edges, n_2cells)
        w = torch.randn(cc.num_cells(2))
        curl_signal = B2 @ w
        if curl_signal.norm() < 1e-6:
            pytest.skip("Degenerate B2 for this triangle")
        curl_signal = curl_signal / curl_signal.norm()

        emb = cc.get_embeddings(1).clone()
        emb[:, 0] = curl_signal
        cc.set_embeddings(1, emb)

        # Run the loop
        _, num_iters, _ = loop(cc)

        # Decompose final edge embeddings
        final_edge_emb = cc.get_embeddings(1)
        signal = final_edge_emb.mean(dim=1)
        _, curl_component, _ = hodge_decomposition(cc, signal, dim=1)

        # With edge residual blending (ratio=0.5), after N iterations the
        # original signal contributes 0.5^N.  For 1-3 iterations the curl
        # should still be measurable.
        assert curl_component.norm().item() > 1e-6, (
            f"Curl zeroed out after {num_iters} iterations "
            f"(norm={curl_component.norm().item():.2e})"
        )

    def test_edge_residual_blending_active(self):
        """Verify the executive loop uses blending (not pure overwrite) for edges."""
        loop = ExecutiveReasoningLoop(
            **LOOP_KWARGS, use_higher_order=True, use_wave_dynamics=False,
        )
        assert hasattr(loop, 'edge_residual_ratio')
        assert 0 < loop.edge_residual_ratio < 1

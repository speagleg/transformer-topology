"""Tests for sheaf diffusion numerical stability on challenging topologies."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.wave.dynamics import SheafWaveDynamics
from src.benchmarks.graph_generators import random_graph
from src.benchmarks.graph_convert import nx_to_cell_complex


DIM = 16


def _make_topo_cc(topology: str, n: int = 20) -> CellComplex:
    """Create a CellComplex from a random graph with given topology."""
    G = random_graph(n_nodes=n, topology=topology)
    cc, _node_map = nx_to_cell_complex(G, embedding_dim=DIM)
    return cc


def _run_sheaf_forward(cc: CellComplex) -> torch.Tensor:
    """Run SheafWaveDynamics forward and return output."""
    swd = SheafWaveDynamics(DIM, use_wave_strength_gate=True)
    signal = cc.get_embeddings(0)
    out = swd(cc, signal,
              diffusion_time=torch.tensor(1.0),
              wave_damping=torch.tensor(0.1))
    return out


class TestSheafStabilityOnTopologies:
    def test_sheaf_stable_on_ba_graph(self):
        """BA graphs have hub nodes with high condition number."""
        cc = _make_topo_cc('ba', n=20)
        out = _run_sheaf_forward(cc)
        assert out.shape == (cc.num_cells(0), DIM)
        assert torch.isfinite(out).all()

    def test_sheaf_stable_on_ws_graph(self):
        """WS graphs have small-world clustering."""
        cc = _make_topo_cc('ws', n=20)
        out = _run_sheaf_forward(cc)
        assert out.shape == (cc.num_cells(0), DIM)
        assert torch.isfinite(out).all()

    def test_sheaf_stable_on_sbm_graph(self):
        """SBM graphs have block structure with near-zero spectral gaps."""
        cc = _make_topo_cc('sbm', n=20)
        out = _run_sheaf_forward(cc)
        assert out.shape == (cc.num_cells(0), DIM)
        assert torch.isfinite(out).all()

    def test_sheaf_mixed_topologies_no_nan(self):
        """Run through multiple topologies, all should produce finite output."""
        for topo in ['ba', 'ws', 'sbm', 'er', 'grid', 'tree', 'ladder']:
            cc = _make_topo_cc(topo, n=15)
            out = _run_sheaf_forward(cc)
            assert torch.isfinite(out).all(), f"NaN/Inf on topology={topo}"


class TestSheafRestrictionMapBounds:
    def test_restriction_map_singular_values_bounded(self):
        """After spectral normalization, max singular value should be ≤ 1."""
        swd = SheafWaveDynamics(DIM, use_wave_strength_gate=False)
        cc = _make_topo_cc('ba', n=15)
        signal = cc.get_embeddings(0)

        # Run forward to populate restriction maps, then check the Laplacian
        # We check indirectly by verifying the output is bounded
        out = swd(cc, signal,
                  diffusion_time=torch.tensor(1.0),
                  wave_damping=torch.tensor(0.1))
        # Output should not be much larger than input
        assert out.norm() < signal.norm() * 10, "Output energy exploded"
        assert torch.isfinite(out).all()

    def test_sheaf_laplacian_positive_semidefinite(self):
        """The sheaf Laplacian should have no negative eigenvalues after PSD enforcement."""
        swd = SheafWaveDynamics(DIM, use_wave_strength_gate=False)
        cc = _make_topo_cc('sbm', n=10)
        signal = cc.get_embeddings(0)

        L = swd._build_sheaf_laplacian(cc, signal)
        # Check eigenvalues
        eigenvalues = torch.linalg.eigvalsh(L)
        # All eigenvalues should be >= -1e-6 (numerical tolerance)
        assert (eigenvalues >= -1e-5).all(), f"Negative eigenvalue: {eigenvalues.min()}"


class TestSheafGradientStability:
    def test_sheaf_gradient_finite(self):
        """Backward pass should produce finite gradients."""
        swd = SheafWaveDynamics(DIM, use_wave_strength_gate=True)
        cc = _make_topo_cc('ba', n=15)
        signal = cc.get_embeddings(0).clone().requires_grad_(True)

        out = swd(cc, signal,
                  diffusion_time=torch.tensor(1.0),
                  wave_damping=torch.tensor(0.1))
        loss = out.sum()
        loss.backward()

        # Check restriction_net gradients are finite
        for name, p in swd.restriction_net.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"

    def test_sheaf_large_diffusion_time_stable(self):
        """Large diffusion_time should not cause NaN."""
        swd = SheafWaveDynamics(DIM, use_wave_strength_gate=True)
        cc = _make_topo_cc('er', n=15)
        signal = cc.get_embeddings(0)

        out = swd(cc, signal,
                  diffusion_time=torch.tensor(10.0),
                  wave_damping=torch.tensor(0.1))
        assert torch.isfinite(out).all()

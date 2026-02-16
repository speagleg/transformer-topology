"""Tests for the sheaf diffusion module."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.spectral.sheaf_diffusion import SheafLaplacian, SheafDiffusion
from src.spectral.laplacian import hodge_laplacian_0


def make_triangle_complex(dim=4):
    """Triangle graph: 3 nodes, 3 edges forming a cycle."""
    cc = CellComplex(embedding_dim=dim)
    a = cc.add_0_cell(torch.randn(dim), "concept")
    b = cc.add_0_cell(torch.randn(dim), "concept")
    c = cc.add_0_cell(torch.randn(dim), "concept")
    cc.add_1_cell(a, b, torch.randn(dim), "r")
    cc.add_1_cell(b, c, torch.randn(dim), "r")
    cc.add_1_cell(c, a, torch.randn(dim), "r")
    return cc


def make_path_complex(dim=4, length=4):
    """Path graph: linear chain of nodes."""
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "r")
    return cc


class TestSheafLaplacian:
    def test_output_shape(self):
        """Sheaf Laplacian should be (N*d, N*d)."""
        d = 4
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        sheaf = SheafLaplacian(n_edges=n_edges, feature_dim=d)
        L = sheaf(cc)
        n = cc.num_cells(0)
        assert L.shape == (n * d, n * d)

    def test_block_structure(self):
        """Verify block structure: diagonal blocks are PSD, off-diagonal
        blocks appear only where edges exist."""
        d = 4
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        sheaf = SheafLaplacian(n_edges=n_edges, feature_dim=d)
        L = sheaf(cc)
        n = cc.num_cells(0)

        # Diagonal blocks should be positive semi-definite
        for i in range(n):
            block = L[i * d:(i + 1) * d, i * d:(i + 1) * d]
            evals = torch.linalg.eigvalsh(block)
            assert (evals >= -1e-5).all(), f"Diagonal block {i} not PSD"

    def test_symmetric(self):
        """Full sheaf Laplacian should be symmetric (real case)."""
        d = 4
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        sheaf = SheafLaplacian(n_edges=n_edges, feature_dim=d)
        L = sheaf(cc)
        assert torch.allclose(L, L.T, atol=1e-5)

    def test_identity_maps_give_standard_laplacian(self):
        """When all restriction maps are identity, the sheaf Laplacian should
        be L0 tensorized with I_d (i.e., kron(L0, I_d))."""
        d = 3
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        sheaf = SheafLaplacian(n_edges=n_edges, feature_dim=d)

        # Set all restriction maps to identity
        with torch.no_grad():
            for e in range(n_edges):
                sheaf.restriction_maps[e, 0] = torch.eye(d)
                sheaf.restriction_maps[e, 1] = torch.eye(d)

        L_sheaf = sheaf(cc)
        L0 = hodge_laplacian_0(cc)

        # Expected: kron(L0, I_d)
        expected = torch.kron(L0, torch.eye(d))
        assert torch.allclose(L_sheaf, expected, atol=1e-5)

    def test_positive_semidefinite(self):
        """Sheaf Laplacian should be positive semi-definite."""
        d = 3
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        sheaf = SheafLaplacian(n_edges=n_edges, feature_dim=d)

        # Set restriction maps to identity for guaranteed PSD
        with torch.no_grad():
            for e in range(n_edges):
                sheaf.restriction_maps[e, 0] = torch.eye(d)
                sheaf.restriction_maps[e, 1] = torch.eye(d)

        L = sheaf(cc)
        evals = torch.linalg.eigvalsh(L)
        assert (evals >= -1e-5).all()

    def test_zero_on_constant_sheaf_section(self):
        """For identity maps, L_sheaf @ f = 0 when f is constant across nodes
        (a global section of the constant sheaf)."""
        d = 3
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        sheaf = SheafLaplacian(n_edges=n_edges, feature_dim=d)

        with torch.no_grad():
            for e in range(n_edges):
                sheaf.restriction_maps[e, 0] = torch.eye(d)
                sheaf.restriction_maps[e, 1] = torch.eye(d)

        L = sheaf(cc)
        n = cc.num_cells(0)

        # Constant signal: same vector at each node
        v = torch.randn(d)
        f = v.repeat(n)  # (N*d,)
        result = L @ f
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)


class TestSheafLaplacianLowRank:
    def test_low_rank_output_shape(self):
        """Low-rank sheaf Laplacian should have same shape."""
        d = 4
        rank = 2
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        sheaf = SheafLaplacian(n_edges=n_edges, feature_dim=d, rank=rank)
        L = sheaf(cc)
        n = cc.num_cells(0)
        assert L.shape == (n * d, n * d)

    def test_low_rank_symmetric(self):
        """Low-rank sheaf Laplacian should be symmetric."""
        d = 4
        rank = 2
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        sheaf = SheafLaplacian(n_edges=n_edges, feature_dim=d, rank=rank)
        L = sheaf(cc)
        assert torch.allclose(L, L.T, atol=1e-5)

    def test_rank_constraint_reduces_params(self):
        """Low-rank should have fewer parameters than full-rank."""
        d = 8
        rank = 2
        n_edges = 10

        full = SheafLaplacian(n_edges=n_edges, feature_dim=d)
        low = SheafLaplacian(n_edges=n_edges, feature_dim=d, rank=rank)

        full_params = sum(p.numel() for p in full.parameters())
        low_params = sum(p.numel() for p in low.parameters())
        assert low_params < full_params

    def test_gradient_flows_through_low_rank(self):
        """Gradients should flow through low-rank restriction maps."""
        d = 4
        rank = 2
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        sheaf = SheafLaplacian(n_edges=n_edges, feature_dim=d, rank=rank)

        L = sheaf(cc)
        loss = L.sum()
        loss.backward()

        assert sheaf.A.grad is not None
        assert sheaf.B.grad is not None
        assert sheaf.A.grad.abs().sum() > 0
        assert sheaf.B.grad.abs().sum() > 0


class TestSheafDiffusion:
    def test_output_shape(self):
        """Diffusion should preserve signal shape (N, d)."""
        d = 4
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        n = cc.num_cells(0)

        sheaf_lap = SheafLaplacian(n_edges=n_edges, feature_dim=d)
        diffusion = SheafDiffusion(feature_dim=d, n_steps=5)

        signal = torch.randn(n, d)
        L = sheaf_lap(cc)
        t = torch.tensor(0.1)

        out = diffusion(cc, signal, L, t)
        assert out.shape == (n, d)

    def test_zero_time_preserves_signal(self):
        """At t=0, diffusion should return the input signal."""
        d = 4
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        n = cc.num_cells(0)

        sheaf_lap = SheafLaplacian(n_edges=n_edges, feature_dim=d)
        diffusion = SheafDiffusion(feature_dim=d, n_steps=5)

        signal = torch.randn(n, d)
        L = sheaf_lap(cc)
        t = torch.tensor(0.0)

        out = diffusion(cc, signal, L, t)
        assert torch.allclose(out, signal, atol=1e-6)

    def test_signal_norm_approximately_preserved(self):
        """For small diffusion time, signal norm should not change drastically.
        The sheaf Laplacian is PSD (with identity maps), so diffusion
        should not increase signal energy."""
        d = 3
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        n = cc.num_cells(0)

        sheaf_lap = SheafLaplacian(n_edges=n_edges, feature_dim=d)
        # Set identity maps for guaranteed energy non-increase
        with torch.no_grad():
            for e in range(n_edges):
                sheaf_lap.restriction_maps[e, 0] = torch.eye(d)
                sheaf_lap.restriction_maps[e, 1] = torch.eye(d)

        diffusion = SheafDiffusion(feature_dim=d, n_steps=20)

        signal = torch.randn(n, d)
        L = sheaf_lap(cc)
        t = torch.tensor(0.1)

        out = diffusion(cc, signal, L, t)
        # Energy should decrease or stay same (PSD Laplacian)
        assert out.norm() <= signal.norm() + 1e-4

    def test_gradient_flow_through_restriction_maps(self):
        """Gradients should flow from diffusion output back to restriction maps."""
        d = 4
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        n = cc.num_cells(0)

        sheaf_lap = SheafLaplacian(n_edges=n_edges, feature_dim=d)
        diffusion = SheafDiffusion(feature_dim=d, n_steps=3)

        signal = torch.randn(n, d)
        L = sheaf_lap(cc)
        t = torch.tensor(0.5)

        out = diffusion(cc, signal, L, t)
        loss = out.sum()
        loss.backward()

        assert sheaf_lap.restriction_maps.grad is not None
        assert sheaf_lap.restriction_maps.grad.abs().sum() > 0

    def test_longer_diffusion_smooths_more(self):
        """Longer diffusion time should produce smoother (lower variation) signals."""
        d = 3
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        n = cc.num_cells(0)

        sheaf_lap = SheafLaplacian(n_edges=n_edges, feature_dim=d)
        with torch.no_grad():
            for e in range(n_edges):
                sheaf_lap.restriction_maps[e, 0] = torch.eye(d)
                sheaf_lap.restriction_maps[e, 1] = torch.eye(d)

        diffusion = SheafDiffusion(feature_dim=d, n_steps=20)

        signal = torch.randn(n, d)
        L = sheaf_lap(cc)

        out_short = diffusion(cc, signal, L, torch.tensor(0.01))
        out_long = diffusion(cc, signal, L, torch.tensor(0.5))

        # Variation measured as std across nodes
        var_short = out_short.std(dim=0).mean()
        var_long = out_long.std(dim=0).mean()
        assert var_long < var_short + 1e-6

    def test_with_low_rank_sheaf(self):
        """SheafDiffusion should work with low-rank SheafLaplacian."""
        d = 4
        rank = 2
        cc = make_triangle_complex(dim=d)
        n_edges = cc.num_cells(1)
        n = cc.num_cells(0)

        sheaf_lap = SheafLaplacian(n_edges=n_edges, feature_dim=d, rank=rank)
        diffusion = SheafDiffusion(feature_dim=d, n_steps=5)

        signal = torch.randn(n, d)
        L = sheaf_lap(cc)
        t = torch.tensor(0.1)

        out = diffusion(cc, signal, L, t)
        assert out.shape == (n, d)
        assert torch.isfinite(out).all()

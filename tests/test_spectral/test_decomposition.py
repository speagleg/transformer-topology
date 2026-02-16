import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import spectral_decomposition
from src.spectral.positional_encoding import laplacian_pe

import pytest


def make_triangle_complex(dim=4):
    cc = CellComplex(embedding_dim=dim)
    a = cc.add_0_cell(torch.randn(dim), "concept")
    b = cc.add_0_cell(torch.randn(dim), "concept")
    c = cc.add_0_cell(torch.randn(dim), "concept")
    cc.add_1_cell(a, b, torch.randn(dim), "r")
    cc.add_1_cell(b, c, torch.randn(dim), "r")
    cc.add_1_cell(c, a, torch.randn(dim), "r")
    return cc


class TestSpectralDecomposition:
    def test_eigenvalues_sorted(self):
        cc = make_triangle_complex()
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0)
        assert (eigenvalues[1:] >= eigenvalues[:-1] - 1e-6).all()

    def test_eigenvectors_orthonormal(self):
        cc = make_triangle_complex()
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0)
        identity = eigenvectors.T @ eigenvectors
        assert torch.allclose(identity, torch.eye(3), atol=1e-5)

    def test_top_k(self):
        cc = make_triangle_complex()
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0, k=2)
        assert eigenvalues.shape == (2,)
        assert eigenvectors.shape == (3, 2)


class TestEigenvalueNormalization:
    def test_normalized_max_is_one(self):
        """With normalize=True, max eigenvalue should be ~1.0."""
        cc = make_triangle_complex()
        eigenvalues, _ = spectral_decomposition(cc, dim=0, normalize=True)
        assert eigenvalues.max().item() == pytest.approx(1.0, abs=1e-6)

    def test_unnormalized_max_is_not_one(self):
        """Without normalize, max eigenvalue is the raw Laplacian value."""
        cc = make_triangle_complex()
        eigenvalues, _ = spectral_decomposition(cc, dim=0, normalize=False)
        # For a triangle, max eigenvalue is 3.0
        assert eigenvalues.max().item() > 1.5

    def test_different_sizes_similar_normalized_range(self):
        """Normalized eigenvalues from different-size graphs both span [0, 1]."""
        # Small graph (triangle)
        cc_small = make_triangle_complex(dim=4)
        eig_small, _ = spectral_decomposition(cc_small, dim=0, normalize=True)

        # Larger graph (5-node path)
        cc_large = CellComplex(embedding_dim=4)
        nodes = [cc_large.add_0_cell(torch.randn(4), "c") for _ in range(5)]
        for i in range(4):
            cc_large.add_1_cell(nodes[i], nodes[i + 1], torch.randn(4), "r")
        eig_large, _ = spectral_decomposition(cc_large, dim=0, normalize=True)

        # Both should have max ~1.0 and min ~0.0
        assert eig_small.max().item() == pytest.approx(1.0, abs=1e-6)
        assert eig_large.max().item() == pytest.approx(1.0, abs=1e-6)
        assert eig_small.min().item() >= -1e-6
        assert eig_large.min().item() >= -1e-6

    def test_normalize_with_k(self):
        """Normalization works correctly when combined with k truncation."""
        cc = make_triangle_complex()
        eigenvalues, eigenvectors = spectral_decomposition(cc, dim=0, k=2, normalize=True)
        assert eigenvalues.shape == (2,)
        # First k eigenvalues should all be <= 1.0
        assert (eigenvalues <= 1.0 + 1e-6).all()


class TestLaplacianPE:
    def test_shape(self):
        cc = make_triangle_complex()
        pe = laplacian_pe(cc, dim=0, k=2)
        assert pe.shape == (3, 2)

    def test_sign_invariant(self):
        cc = make_triangle_complex()
        pe = laplacian_pe(cc, dim=0, k=2)
        norms = pe.norm(dim=1)
        assert (norms > 0).all()

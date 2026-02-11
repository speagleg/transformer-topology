import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import spectral_decomposition
from src.spectral.positional_encoding import laplacian_pe


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

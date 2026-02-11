import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.spectral.laplacian import hodge_laplacian_0, hodge_laplacian_1


def make_triangle_complex(dim=4):
    cc = CellComplex(embedding_dim=dim)
    a = cc.add_0_cell(torch.randn(dim), "concept")
    b = cc.add_0_cell(torch.randn(dim), "concept")
    c = cc.add_0_cell(torch.randn(dim), "concept")
    cc.add_1_cell(a, b, torch.randn(dim), "r")
    cc.add_1_cell(b, c, torch.randn(dim), "r")
    cc.add_1_cell(c, a, torch.randn(dim), "r")
    return cc


def make_chain_complex(dim=4, length=4):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "r")
    return cc


class TestHodgeLaplacian0:
    def test_shape(self):
        cc = make_triangle_complex()
        L0 = hodge_laplacian_0(cc)
        assert L0.shape == (3, 3)

    def test_symmetric(self):
        cc = make_triangle_complex()
        L0 = hodge_laplacian_0(cc)
        assert torch.allclose(L0, L0.T)

    def test_positive_semidefinite(self):
        cc = make_triangle_complex()
        L0 = hodge_laplacian_0(cc)
        eigenvalues = torch.linalg.eigvalsh(L0)
        assert (eigenvalues >= -1e-6).all()

    def test_row_sum_zero(self):
        cc = make_triangle_complex()
        L0 = hodge_laplacian_0(cc)
        assert torch.allclose(L0.sum(dim=1), torch.zeros(3), atol=1e-6)

    def test_chain_spectrum(self):
        cc = make_chain_complex(length=5)
        L0 = hodge_laplacian_0(cc)
        eigenvalues = torch.linalg.eigvalsh(L0)
        num_zero = (eigenvalues.abs() < 1e-6).sum().item()
        assert num_zero == 1


class TestHodgeLaplacian1:
    def test_shape(self):
        cc = make_triangle_complex()
        L1 = hodge_laplacian_1(cc)
        assert L1.shape == (3, 3)

    def test_symmetric(self):
        cc = make_triangle_complex()
        L1 = hodge_laplacian_1(cc)
        assert torch.allclose(L1, L1.T)

    def test_positive_semidefinite(self):
        cc = make_triangle_complex()
        L1 = hodge_laplacian_1(cc)
        eigenvalues = torch.linalg.eigvalsh(L1)
        assert (eigenvalues >= -1e-6).all()

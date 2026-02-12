import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import hodge_decomposition
from src.spectral.laplacian import hodge_laplacian_1


def make_triangle(dim=8, fill=False):
    """Create a triangle graph. If fill=True, add a 2-cell face."""
    cc = CellComplex(embedding_dim=dim)
    n0 = cc.add_0_cell(torch.randn(dim), "node")
    n1 = cc.add_0_cell(torch.randn(dim), "node")
    n2 = cc.add_0_cell(torch.randn(dim), "node")
    e0 = cc.add_1_cell(n0, n1, torch.randn(dim), "edge")
    e1 = cc.add_1_cell(n1, n2, torch.randn(dim), "edge")
    e2 = cc.add_1_cell(n2, n0, torch.randn(dim), "edge")
    if fill:
        cc.add_2_cell([e0, e1, e2], torch.randn(dim))
    return cc


class TestHodgeDecomposition:
    def test_reconstruction(self):
        """gradient + curl + harmonic should equal the original signal."""
        cc = make_triangle(fill=True)
        signal = torch.randn(3)
        g, c, h = hodge_decomposition(cc, signal, dim=1)
        reconstructed = g + c + h
        assert torch.allclose(reconstructed, signal, atol=1e-5)

    def test_orthogonality(self):
        """Gradient and curl components should be approximately orthogonal."""
        cc = make_triangle(fill=True)
        signal = torch.randn(3)
        g, c, h = hodge_decomposition(cc, signal, dim=1)
        # Gradient and curl should be orthogonal
        assert abs(torch.dot(g, c).item()) < 1e-4

    def test_harmonic_in_kernel(self):
        """Harmonic component should be in the kernel of the Hodge Laplacian."""
        cc = make_triangle(fill=True)
        signal = torch.randn(3)
        g, c, h = hodge_decomposition(cc, signal, dim=1)
        L1 = hodge_laplacian_1(cc)
        Lh = L1 @ h
        assert torch.allclose(Lh, torch.zeros_like(Lh), atol=1e-4)

    def test_filled_triangle_no_harmonic(self):
        """A filled triangle has no 1-dimensional holes, so harmonic should be ~0."""
        cc = make_triangle(fill=True)
        signal = torch.randn(3)
        g, c, h = hodge_decomposition(cc, signal, dim=1)
        assert h.norm().item() < 1e-4

    def test_unfilled_triangle_has_harmonic(self):
        """An unfilled triangle has a 1-cycle, so harmonic should be nonzero for generic signals."""
        cc = make_triangle(fill=False)
        # Use a signal that has a harmonic component: constant on all edges
        signal = torch.ones(3)
        g, c, h = hodge_decomposition(cc, signal, dim=1)
        # The cycle [1,1,1] should have a nonzero harmonic component
        assert h.norm().item() > 0.1

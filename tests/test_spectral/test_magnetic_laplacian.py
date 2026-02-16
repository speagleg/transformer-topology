"""Tests for the magnetic Laplacian module."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.spectral.magnetic_laplacian import magnetic_laplacian, magnetic_spectral_decomposition
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


def make_single_edge(dim=4):
    """Simplest graph: 2 nodes, 1 edge."""
    cc = CellComplex(embedding_dim=dim)
    a = cc.add_0_cell(torch.randn(dim), "concept")
    b = cc.add_0_cell(torch.randn(dim), "concept")
    cc.add_1_cell(a, b, torch.randn(dim), "r")
    return cc


class TestMagneticLaplacian:
    def test_hermitian(self):
        """Magnetic Laplacian must be Hermitian: L_q = L_q^H."""
        cc = make_triangle_complex()
        L_q = magnetic_laplacian(cc, q=0.25)
        assert torch.allclose(L_q, L_q.conj().T, atol=1e-6)

    def test_hermitian_various_q(self):
        """Hermitian property must hold for any q value."""
        cc = make_triangle_complex()
        for q in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            L_q = magnetic_laplacian(cc, q=q)
            assert torch.allclose(L_q, L_q.conj().T, atol=1e-6), f"Failed for q={q}"

    def test_q_zero_is_standard_laplacian(self):
        """At q=0, the magnetic Laplacian should equal the standard L0."""
        cc = make_triangle_complex()
        L_mag = magnetic_laplacian(cc, q=0.0)
        L0 = hodge_laplacian_0(cc)

        # At q=0: exp(i*0) = 1, so off-diag = -1*A[u,v], diag = degree
        # This is exactly the standard graph Laplacian
        assert torch.allclose(L_mag.real, L0, atol=1e-6)
        assert torch.allclose(L_mag.imag, torch.zeros_like(L_mag.imag), atol=1e-6)

    def test_shape(self):
        """Output shape should be (N, N) complex."""
        cc = make_triangle_complex()
        L_q = magnetic_laplacian(cc, q=0.25)
        assert L_q.shape == (3, 3)
        assert L_q.is_complex()

    def test_diagonal_is_real(self):
        """Diagonal entries are degree values (real, non-negative)."""
        cc = make_triangle_complex()
        L_q = magnetic_laplacian(cc, q=0.25)
        diag = torch.diagonal(L_q)
        assert torch.allclose(diag.imag, torch.zeros(3), atol=1e-6)
        assert (diag.real >= -1e-6).all()

    def test_diagonal_equals_degree(self):
        """Diagonal should equal node degree."""
        cc = make_triangle_complex()
        L_q = magnetic_laplacian(cc, q=0.3)
        # Triangle: each node has degree 2
        expected_diag = torch.tensor([2.0, 2.0, 2.0])
        assert torch.allclose(torch.diagonal(L_q).real, expected_diag, atol=1e-6)

    def test_triangle_graph(self):
        """Verify structure on a simple triangle: 3 nodes, 3 edges."""
        cc = make_triangle_complex()
        L_q = magnetic_laplacian(cc, q=0.25)

        # All nodes should have degree 2
        for i in range(3):
            assert abs(L_q[i, i].real.item() - 2.0) < 1e-6

        # Off-diagonal entries should be non-zero (edges exist)
        # For q != 0, they should have imaginary parts
        assert L_q[0, 1].abs().item() > 0.5

    def test_q_half_gives_negative_adjacency(self):
        """At q=0.5, exp(i*2*pi*0.5*theta) = exp(i*pi*theta).
        For theta=+1: exp(i*pi) = -1, so off-diag = -(-1) = +1.
        For theta=-1: exp(-i*pi) = -1, so off-diag = -(-1) = +1.
        Result: L_q=0.5 has +1 off-diagonal (like negative standard adjacency)."""
        cc = make_single_edge()
        L_q = magnetic_laplacian(cc, q=0.5)
        # Off-diag: -exp(i*pi) = -(-1) = 1
        assert torch.allclose(L_q[0, 1].real, torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(L_q[0, 1].imag, torch.tensor(0.0), atol=1e-5)

    def test_path_graph(self):
        """Magnetic Laplacian on a path should be Hermitian."""
        cc = make_path_complex(length=5)
        L_q = magnetic_laplacian(cc, q=0.25)
        assert L_q.shape == (5, 5)
        assert torch.allclose(L_q, L_q.conj().T, atol=1e-6)

    def test_device_cpu(self):
        """Verify output is on the correct device."""
        cc = make_triangle_complex()
        L_q = magnetic_laplacian(cc, q=0.25)
        assert L_q.device == torch.device("cpu")


class TestMagneticSpectralDecomposition:
    def test_eigenvalues_real(self):
        """Eigenvalues of a Hermitian matrix must be real."""
        cc = make_triangle_complex()
        eigenvalues, eigenvectors = magnetic_spectral_decomposition(cc, q=0.25)
        # eigh returns real eigenvalues directly
        assert eigenvalues.dtype in (torch.float32, torch.float64)

    def test_eigenvalues_nonnegative(self):
        """Eigenvalues of the magnetic Laplacian should be non-negative."""
        cc = make_triangle_complex()
        eigenvalues, _ = magnetic_spectral_decomposition(cc, q=0.25)
        assert (eigenvalues >= -1e-5).all()

    def test_eigenvalues_sorted(self):
        """Eigenvalues should be in ascending order."""
        cc = make_triangle_complex()
        eigenvalues, _ = magnetic_spectral_decomposition(cc, q=0.25)
        assert (eigenvalues[1:] >= eigenvalues[:-1] - 1e-6).all()

    def test_eigenvectors_complex(self):
        """Eigenvectors of the magnetic Laplacian are complex-valued."""
        cc = make_triangle_complex()
        _, eigenvectors = magnetic_spectral_decomposition(cc, q=0.25)
        assert eigenvectors.is_complex()

    def test_eigenvectors_unitary(self):
        """Eigenvectors should form a unitary basis (V^H @ V = I)."""
        cc = make_triangle_complex()
        _, V = magnetic_spectral_decomposition(cc, q=0.25)
        product = V.conj().T @ V
        identity = torch.eye(V.shape[1], dtype=product.dtype)
        assert torch.allclose(product, identity, atol=1e-5)

    def test_reconstruction(self):
        """L_q = V @ diag(lambda) @ V^H."""
        cc = make_triangle_complex()
        eigenvalues, V = magnetic_spectral_decomposition(cc, q=0.25)
        L_q = magnetic_laplacian(cc, q=0.25)
        reconstructed = V @ torch.diag(eigenvalues.to(V.dtype)) @ V.conj().T
        assert torch.allclose(L_q, reconstructed, atol=1e-4)

    def test_top_k(self):
        """Can request only k smallest eigenvalues."""
        cc = make_triangle_complex()
        eigenvalues, eigenvectors = magnetic_spectral_decomposition(cc, q=0.25, k=2)
        assert eigenvalues.shape == (2,)
        assert eigenvectors.shape == (3, 2)

    def test_q_zero_eigenvalues_match_standard(self):
        """At q=0, eigenvalues should match standard Laplacian eigenvalues."""
        cc = make_triangle_complex()
        mag_evals, _ = magnetic_spectral_decomposition(cc, q=0.0)
        L0 = hodge_laplacian_0(cc)
        std_evals = torch.linalg.eigvalsh(L0)
        assert torch.allclose(mag_evals, std_evals, atol=1e-5)

    def test_device_handling(self):
        """Eigendecomposition should produce tensors on the correct device."""
        cc = make_triangle_complex()
        eigenvalues, eigenvectors = magnetic_spectral_decomposition(cc, q=0.25)
        assert eigenvalues.device == torch.device("cpu")
        assert eigenvectors.device == torch.device("cpu")

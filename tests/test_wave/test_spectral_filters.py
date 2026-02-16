"""Tests for pluggable spectral filters on cell complexes."""

import torch
import pytest

from src.cell_complex.cell_complex import CellComplex
from src.wave.spectral_filters import (
    SpectralFilter,
    HeatFilter,
    BandPassFilter,
    WaveCosineFilter,
    SchrodingerFilter,
    ChebyshevFilter,
    SpectralWaveletFilter,
    MagneticFilter,
    FILTER_REGISTRY,
    create_spectral_filter,
)


def make_chain(dim: int = 16, length: int = 5) -> CellComplex:
    """Build a simple chain graph: 0--1--2--...--length-1."""
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "node") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "edge")
    return cc


def make_triangle(dim: int = 16) -> CellComplex:
    """Triangle graph with a 2-cell for testing L1/L2."""
    cc = CellComplex(embedding_dim=dim)
    a = cc.add_0_cell(torch.randn(dim), "node")
    b = cc.add_0_cell(torch.randn(dim), "node")
    c = cc.add_0_cell(torch.randn(dim), "node")
    e0 = cc.add_1_cell(a, b, torch.randn(dim), "edge")
    e1 = cc.add_1_cell(b, c, torch.randn(dim), "edge")
    e2 = cc.add_1_cell(a, c, torch.randn(dim), "edge")
    cc.add_2_cell([e0, e1, e2], torch.randn(dim), "face")
    return cc


# ---------------------------------------------------------------------------
# HeatFilter
# ---------------------------------------------------------------------------


class TestHeatFilter:
    def test_zero_time_identity(self):
        """At t=0, exp(-lambda*0) = 1 for all lambdas → identity."""
        cc = make_chain(dim=16, length=5)
        signal = torch.randn(5, 16)
        f = HeatFilter(dim=0)
        out = f(cc, signal, diffusion_time=torch.tensor(0.0))
        assert out.shape == signal.shape
        assert torch.allclose(out, signal, atol=1e-5)

    def test_1d_signal(self):
        """Works with 1-D signal (N,)."""
        cc = make_chain(dim=16, length=4)
        signal = torch.randn(4)
        f = HeatFilter(dim=0)
        out = f(cc, signal, diffusion_time=torch.tensor(0.0))
        assert out.shape == (4,)
        assert torch.allclose(out, signal, atol=1e-5)

    def test_large_time_smooths(self):
        """Large t drives all node values toward mean (low-pass)."""
        cc = make_chain(dim=8, length=6)
        signal = torch.randn(6, 8)
        f = HeatFilter(dim=0)
        out = f(cc, signal, diffusion_time=torch.tensor(1000.0))
        col_std = out.std(dim=0)
        assert (col_std < 0.01).all()

    def test_matches_heat_diffusion_structure(self):
        """HeatFilter output should have same structure as HeatDiffusion.

        Note: HeatFilter uses normalized eigenvalues for size-invariance,
        so outputs differ numerically from HeatDiffusion (raw eigenvalues).
        Both should still produce valid smoothing of the signal.
        """
        from src.wave.dynamics import HeatDiffusion

        cc = make_chain(dim=8, length=5)
        signal = torch.randn(5, 8)
        t = torch.tensor(0.5)

        old = HeatDiffusion()(cc, signal, t)
        new = HeatFilter(dim=0)(cc, signal, diffusion_time=t)
        # Both should have the same shape and be finite
        assert old.shape == new.shape
        assert torch.isfinite(old).all()
        assert torch.isfinite(new).all()
        # Both should smooth the signal (reduce variance relative to input)
        assert signal.std() > 0
        assert old.std() <= signal.std() + 0.1
        assert new.std() <= signal.std() + 0.1


# ---------------------------------------------------------------------------
# BandPassFilter
# ---------------------------------------------------------------------------


class TestBandPassFilter:
    def test_output_shape(self):
        cc = make_chain(dim=8, length=5)
        signal = torch.randn(5, 8)
        f = BandPassFilter(dim=0, center=1.0, bandwidth=0.5)
        out = f(cc, signal)
        assert out.shape == (5, 8)

    def test_learnable_params(self):
        f = BandPassFilter(dim=0, center=1.0, bandwidth=0.5)
        params = dict(f.named_parameters())
        assert 'center' in params
        assert 'bandwidth' in params

    def test_gradient_flows(self):
        cc = make_chain(dim=8, length=4)
        signal = torch.randn(4, 8)
        f = BandPassFilter(dim=0)
        out = f(cc, signal)
        out.sum().backward()
        assert f.center.grad is not None
        assert f.bandwidth.grad is not None


# ---------------------------------------------------------------------------
# WaveCosineFilter
# ---------------------------------------------------------------------------


class TestWaveCosineFilter:
    def test_zero_time_identity(self):
        """cos(sqrt(lambda)*0) = 1 → identity."""
        cc = make_chain(dim=8, length=5)
        signal = torch.randn(5, 8)
        f = WaveCosineFilter(dim=0)
        out = f(cc, signal, diffusion_time=torch.tensor(0.0))
        assert torch.allclose(out, signal, atol=1e-5)

    def test_output_shape(self):
        cc = make_chain(dim=8, length=5)
        signal = torch.randn(5, 8)
        f = WaveCosineFilter(dim=0)
        out = f(cc, signal, diffusion_time=torch.tensor(0.3))
        assert out.shape == (5, 8)

    def test_oscillatory_not_smoothing(self):
        """Wave cosine preserves energy, unlike heat diffusion."""
        cc = make_chain(dim=8, length=6)
        signal = torch.randn(6, 8)
        f = WaveCosineFilter(dim=0)
        out = f(cc, signal, diffusion_time=torch.tensor(0.5))
        # Energy should be approximately preserved (cos is norm-preserving)
        assert abs(out.norm().item() - signal.norm().item()) / signal.norm().item() < 0.5


# ---------------------------------------------------------------------------
# SchrodingerFilter
# ---------------------------------------------------------------------------


class TestSchrodingerFilter:
    def test_zero_time_identity(self):
        """cos(lambda*0) = 1 → identity."""
        cc = make_chain(dim=8, length=5)
        signal = torch.randn(5, 8)
        f = SchrodingerFilter(dim=0)
        out = f(cc, signal, diffusion_time=torch.tensor(0.0))
        assert torch.allclose(out, signal, atol=1e-5)

    def test_output_shape(self):
        cc = make_chain(dim=8, length=4)
        signal = torch.randn(4, 8)
        f = SchrodingerFilter(dim=0)
        out = f(cc, signal, diffusion_time=torch.tensor(0.5))
        assert out.shape == (4, 8)


# ---------------------------------------------------------------------------
# ChebyshevFilter
# ---------------------------------------------------------------------------


class TestChebyshevFilter:
    def test_output_shape(self):
        cc = make_chain(dim=8, length=5)
        signal = torch.randn(5, 8)
        f = ChebyshevFilter(dim=0, order=5)
        out = f(cc, signal)
        assert out.shape == (5, 8)

    def test_default_is_approx_identity(self):
        """Initial coefficients [1, 0, 0, ...] → T_0 = 1 → identity-ish."""
        cc = make_chain(dim=8, length=5)
        signal = torch.randn(5, 8)
        f = ChebyshevFilter(dim=0, order=5)
        out = f(cc, signal)
        # T_0(x) = 1 for all x, so this is approximately identity
        assert torch.allclose(out, signal, atol=1e-4)

    def test_learnable_coeffs(self):
        f = ChebyshevFilter(dim=0, order=5)
        params = dict(f.named_parameters())
        assert 'coeffs' in params
        assert params['coeffs'].shape == (5,)

    def test_gradient_flows(self):
        cc = make_chain(dim=8, length=4)
        signal = torch.randn(4, 8)
        f = ChebyshevFilter(dim=0, order=3)
        out = f(cc, signal)
        out.sum().backward()
        assert f.coeffs.grad is not None


# ---------------------------------------------------------------------------
# SpectralWaveletFilter
# ---------------------------------------------------------------------------


class TestSpectralWaveletFilter:
    def test_output_shape(self):
        cc = make_chain(dim=8, length=5)
        signal = torch.randn(5, 8)
        f = SpectralWaveletFilter(dim=0, n_scales=4)
        out = f(cc, signal)
        assert out.shape == (5, 8)

    def test_learnable_scales(self):
        f = SpectralWaveletFilter(dim=0, n_scales=4)
        params = dict(f.named_parameters())
        assert 'log_scales' in params
        assert params['log_scales'].shape == (4,)

    def test_gradient_flows(self):
        cc = make_chain(dim=8, length=4)
        signal = torch.randn(4, 8)
        f = SpectralWaveletFilter(dim=0, n_scales=3)
        out = f(cc, signal)
        out.sum().backward()
        assert f.log_scales.grad is not None


# ---------------------------------------------------------------------------
# MagneticFilter
# ---------------------------------------------------------------------------


class TestMagneticFilter:
    def test_magnetic_filter_forward(self):
        """MagneticFilter runs and produces finite output."""
        cc = make_chain(dim=32, length=10)
        f = create_spectral_filter('magnetic', dim=0, q=0.25)
        signal = cc.get_embeddings(0)
        out = f(cc, signal, diffusion_time=torch.tensor(1.0))
        assert out.shape == signal.shape
        assert torch.isfinite(out).all()

    def test_magnetic_filter_zero_time_identity(self):
        """At t=0, exp(-lambda*0) = 1 → identity."""
        cc = make_chain(dim=16, length=5)
        signal = torch.randn(5, 16)
        f = MagneticFilter(dim=0, q=0.25)
        out = f(cc, signal, diffusion_time=torch.tensor(0.0))
        assert out.shape == signal.shape
        assert torch.allclose(out, signal, atol=1e-4)

    def test_magnetic_filter_q_zero_matches_heat(self):
        """At q=0, magnetic Laplacian reduces to standard Laplacian."""
        cc = make_chain(dim=8, length=5)
        signal = torch.randn(5, 8)
        t = torch.tensor(0.5)
        mag_out = MagneticFilter(dim=0, q=0.0)(cc, signal, diffusion_time=t)
        heat_out = HeatFilter(dim=0)(cc, signal, diffusion_time=t)
        assert torch.allclose(mag_out, heat_out, atol=1e-4)

    def test_magnetic_filter_output_is_real(self):
        """Output should be real-valued even though eigenvectors are complex."""
        cc = make_chain(dim=8, length=6)
        signal = torch.randn(6, 8)
        f = MagneticFilter(dim=0, q=0.25)
        out = f(cc, signal, diffusion_time=torch.tensor(0.5))
        assert out.dtype in (torch.float32, torch.float64)

    def test_magnetic_filter_via_factory(self):
        """create_spectral_filter passes q kwarg correctly."""
        f = create_spectral_filter('magnetic', dim=0, q=0.5)
        assert isinstance(f, MagneticFilter)
        assert f.q == 0.5


# ---------------------------------------------------------------------------
# Registry and Factory
# ---------------------------------------------------------------------------


class TestFilterRegistry:
    def test_all_filters_registered(self):
        expected = {'heat', 'bandpass', 'wave_cosine', 'schrodinger', 'chebyshev', 'wavelet', 'magnetic'}
        assert set(FILTER_REGISTRY.keys()) == expected

    def test_create_spectral_filter_heat(self):
        f = create_spectral_filter('heat', dim=0)
        assert isinstance(f, HeatFilter)
        assert f.dim == 0

    def test_create_spectral_filter_bandpass(self):
        f = create_spectral_filter('bandpass', dim=1, center=2.0, bandwidth=0.3)
        assert isinstance(f, BandPassFilter)
        assert f.dim == 1
        assert f.center.item() == pytest.approx(2.0)

    def test_create_spectral_filter_chebyshev(self):
        f = create_spectral_filter('chebyshev', dim=0, order=7)
        assert isinstance(f, ChebyshevFilter)
        assert f.coeffs.shape == (7,)

    def test_create_spectral_filter_wavelet(self):
        f = create_spectral_filter('wavelet', dim=0, n_scales=6)
        assert isinstance(f, SpectralWaveletFilter)
        assert f.log_scales.shape == (6,)

    def test_invalid_filter_type(self):
        with pytest.raises(ValueError, match="Unknown filter type"):
            create_spectral_filter('nonexistent')


# ---------------------------------------------------------------------------
# L1 / L2 Laplacian support (dim=1, dim=2)
# ---------------------------------------------------------------------------


class TestHigherDimFilters:
    def test_heat_on_edges_dim1(self):
        """HeatFilter on dim=1 (edge Laplacian) should work on a triangle."""
        cc = make_triangle()
        n_edges = cc.num_cells(1)
        signal = torch.randn(n_edges, 8)
        f = HeatFilter(dim=1)
        out = f(cc, signal, diffusion_time=torch.tensor(0.5))
        assert out.shape == (n_edges, 8)

    def test_heat_on_faces_dim2(self):
        """HeatFilter on dim=2 (face Laplacian) should work."""
        cc = make_triangle()
        n_faces = cc.num_cells(2)
        signal = torch.randn(n_faces, 8)
        f = HeatFilter(dim=2)
        out = f(cc, signal, diffusion_time=torch.tensor(0.5))
        assert out.shape == (n_faces, 8)

    def test_chebyshev_on_edges(self):
        """ChebyshevFilter on edges."""
        cc = make_triangle()
        n_edges = cc.num_cells(1)
        signal = torch.randn(n_edges, 8)
        f = ChebyshevFilter(dim=1, order=3)
        out = f(cc, signal)
        assert out.shape == (n_edges, 8)

    def test_wavelet_on_edges(self):
        """SpectralWaveletFilter on edges."""
        cc = make_triangle()
        n_edges = cc.num_cells(1)
        signal = torch.randn(n_edges, 8)
        f = SpectralWaveletFilter(dim=1, n_scales=3)
        out = f(cc, signal)
        assert out.shape == (n_edges, 8)

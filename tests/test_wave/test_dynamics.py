"""Tests for wave/diffusion dynamics on cell complexes."""

import torch
import pytest

from src.cell_complex.cell_complex import CellComplex
from src.wave.dynamics import HeatDiffusion, WavePropagation, WaveDynamics


def make_chain(dim: int = 32, length: int = 5) -> CellComplex:
    """Build a simple chain graph: 0--1--2--...--length-1."""
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "node") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "edge")
    return cc


# ---------------------------------------------------------------------------
# HeatDiffusion
# ---------------------------------------------------------------------------


class TestHeatDiffusion:
    def test_heat_zero_time_identity(self):
        """With diffusion_time=0 the heat kernel is the identity."""
        cc = make_chain(dim=16, length=5)
        signal = torch.randn(5, 16)
        heat = HeatDiffusion()

        t = torch.tensor(0.0)
        out = heat(cc, signal, t)

        assert out.shape == signal.shape
        assert torch.allclose(out, signal, atol=1e-5), (
            f"max diff = {(out - signal).abs().max().item()}"
        )

    def test_heat_zero_time_identity_1d(self):
        """1-D signal variant: diffusion_time=0 returns original."""
        cc = make_chain(dim=16, length=4)
        signal = torch.randn(4)
        heat = HeatDiffusion()

        out = heat(cc, signal, torch.tensor(0.0))
        assert out.shape == signal.shape
        assert torch.allclose(out, signal, atol=1e-5)

    def test_heat_large_time_smoothing(self):
        """Large t drives all node values toward their mean."""
        cc = make_chain(dim=8, length=6)
        signal = torch.randn(6, 8)
        heat = HeatDiffusion()

        t = torch.tensor(1000.0)
        out = heat(cc, signal, t)

        # After long diffusion, each column should be near-constant.
        # The connected-component mean is preserved by the heat equation
        # on the graph Laplacian.  For a connected graph, the constant
        # eigenvector (eigenvalue=0) carries the mean.
        col_std = out.std(dim=0)  # spread per feature
        assert (col_std < 0.01).all(), f"column std too large: {col_std}"


# ---------------------------------------------------------------------------
# WavePropagation
# ---------------------------------------------------------------------------


class TestWavePropagation:
    def test_wave_output_shape(self):
        """Output shape matches input for various graph sizes."""
        for length in [3, 5, 8]:
            dim = 16
            cc = make_chain(dim=dim, length=length)
            signal = torch.randn(length, dim)
            wave = WavePropagation(embedding_dim=dim)

            t = torch.tensor(0.5)
            damping = torch.tensor(0.1)
            out = wave(cc, signal, t, damping)

            assert out.shape == (length, dim), (
                f"Expected ({length}, {dim}), got {out.shape}"
            )

    def test_wave_gradient_flow(self):
        """Gradients flow through diffusion_time and wave_damping."""
        dim = 16
        cc = make_chain(dim=dim, length=4)
        signal = torch.randn(4, dim)

        wave = WavePropagation(embedding_dim=dim)

        t = torch.tensor(0.5, requires_grad=True)
        damping = torch.tensor(0.1, requires_grad=True)

        out = wave(cc, signal, t, damping)
        loss = out.sum()
        loss.backward()

        assert t.grad is not None, "No gradient for diffusion_time"
        assert damping.grad is not None, "No gradient for wave_damping"
        assert t.grad.abs() > 0, "diffusion_time gradient is zero"
        assert damping.grad.abs() > 0, "wave_damping gradient is zero"


# ---------------------------------------------------------------------------
# WaveDynamics (combined)
# ---------------------------------------------------------------------------


class TestWaveDynamics:
    def test_combined_dynamics_shape(self):
        """WaveDynamics output shape matches input."""
        dim = 16
        cc = make_chain(dim=dim, length=5)
        signal = torch.randn(5, dim)

        dyn = WaveDynamics(embedding_dim=dim)

        t = torch.tensor(0.3)
        damping = torch.tensor(0.2)
        out = dyn(cc, signal, t, damping)

        assert out.shape == (5, dim)

    def test_combined_gradient_through_control(self):
        """Gradients flow from WaveDynamics output through control params."""
        dim = 16
        cc = make_chain(dim=dim, length=4)
        signal = torch.randn(4, dim)

        dyn = WaveDynamics(embedding_dim=dim)

        t = torch.tensor(0.5, requires_grad=True)
        damping = torch.tensor(0.1, requires_grad=True)

        out = dyn(cc, signal, t, damping)
        loss = out.sum()
        loss.backward()

        assert t.grad is not None, "No gradient for diffusion_time"
        assert damping.grad is not None, "No gradient for wave_damping"
        assert t.grad.abs() > 0, "diffusion_time gradient is zero"
        assert damping.grad.abs() > 0, "wave_damping gradient is zero"

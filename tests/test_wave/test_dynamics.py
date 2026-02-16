"""Tests for wave/diffusion dynamics on cell complexes."""

import torch
import pytest

from src.cell_complex.cell_complex import CellComplex
from src.wave.dynamics import HeatDiffusion, WavePropagation, WaveDynamics, SheafWaveDynamics


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
        """WaveDynamics output shape matches input (default config)."""
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


# ---------------------------------------------------------------------------
# WaveDynamics with configurable filters
# ---------------------------------------------------------------------------


class TestWaveDynamicsConfigurable:
    def test_wave_cosine_filter(self):
        """WaveDynamics with wave_cosine filter."""
        dim = 16
        cc = make_chain(dim=dim, length=5)
        signal = torch.randn(5, dim)
        dyn = WaveDynamics(embedding_dim=dim, filter_type='wave_cosine')
        out = dyn(cc, signal, torch.tensor(0.3), torch.tensor(0.2))
        assert out.shape == (5, dim)

    def test_chebyshev_filter(self):
        """WaveDynamics with chebyshev filter."""
        dim = 16
        cc = make_chain(dim=dim, length=5)
        signal = torch.randn(5, dim)
        dyn = WaveDynamics(embedding_dim=dim, filter_type='chebyshev', order=3)
        out = dyn(cc, signal, torch.tensor(0.3), torch.tensor(0.2))
        assert out.shape == (5, dim)

    def test_bandpass_filter(self):
        """WaveDynamics with bandpass filter."""
        dim = 16
        cc = make_chain(dim=dim, length=5)
        signal = torch.randn(5, dim)
        dyn = WaveDynamics(embedding_dim=dim, filter_type='bandpass')
        out = dyn(cc, signal, torch.tensor(0.3), torch.tensor(0.2))
        assert out.shape == (5, dim)

    def test_no_neural_ode(self):
        """WaveDynamics with use_neural_ode=False uses only spectral filter."""
        dim = 16
        cc = make_chain(dim=dim, length=5)
        signal = torch.randn(5, dim)
        dyn = WaveDynamics(embedding_dim=dim, use_neural_ode=False)
        out = dyn(cc, signal, torch.tensor(0.3), torch.tensor(0.2))
        assert out.shape == (5, dim)
        # Should not have wave or gate submodules
        assert not hasattr(dyn, 'wave')
        assert not hasattr(dyn, 'gate')


# ---------------------------------------------------------------------------
# Wave strength gate
# ---------------------------------------------------------------------------


class TestWaveStrengthGate:
    def test_gate_creates_parameter(self):
        """use_wave_strength_gate=True adds a learnable parameter."""
        dyn = WaveDynamics(embedding_dim=16, use_wave_strength_gate=True)
        assert hasattr(dyn, 'wave_strength')
        assert dyn.wave_strength.requires_grad

    def test_gate_output_shape(self):
        """Output shape matches with gate enabled."""
        dim = 16
        cc = make_chain(dim=dim, length=5)
        signal = torch.randn(5, dim)
        dyn = WaveDynamics(embedding_dim=dim, use_wave_strength_gate=True)
        out = dyn(cc, signal, torch.tensor(0.3), torch.tensor(0.2))
        assert out.shape == (5, dim)

    def test_gate_gradient_flows(self):
        """Gradient flows through the wave strength gate."""
        dim = 16
        cc = make_chain(dim=dim, length=4)
        signal = torch.randn(4, dim)
        dyn = WaveDynamics(embedding_dim=dim, use_wave_strength_gate=True)
        out = dyn(cc, signal, torch.tensor(0.3), torch.tensor(0.2))
        out.sum().backward()
        assert dyn.wave_strength.grad is not None

    def test_gate_zero_bypasses_dynamics(self):
        """When wave_strength → -inf, sigmoid(strength) → 0, output ≈ signal."""
        dim = 16
        cc = make_chain(dim=dim, length=5)
        signal = torch.randn(5, dim)
        dyn = WaveDynamics(
            embedding_dim=dim, use_wave_strength_gate=True,
            use_neural_ode=False,
        )
        with torch.no_grad():
            dyn.wave_strength.fill_(-100.0)  # sigmoid(-100) ≈ 0
        out = dyn(cc, signal, torch.tensor(0.3), torch.tensor(0.2))
        # strength ≈ 0, so output ≈ (1-0)*signal = signal
        assert torch.allclose(out, signal, atol=1e-4)


# ---------------------------------------------------------------------------
# SheafWaveDynamics
# ---------------------------------------------------------------------------


class TestSheafWaveDynamics:
    def test_sheaf_wave_dynamics_forward(self):
        """SheafWaveDynamics runs and produces correct shape output."""
        dim = 32
        cc = make_chain(dim=dim, length=10)
        swd = SheafWaveDynamics(dim, use_wave_strength_gate=True)
        signal = cc.get_embeddings(0)
        out = swd(cc, signal, diffusion_time=torch.tensor(1.0),
                  wave_damping=torch.tensor(0.1))
        assert out.shape == signal.shape
        assert torch.isfinite(out).all()

    def test_sheaf_wave_dynamics_no_gate(self):
        """SheafWaveDynamics works without wave strength gate."""
        dim = 16
        cc = make_chain(dim=dim, length=5)
        swd = SheafWaveDynamics(dim, use_wave_strength_gate=False)
        signal = torch.randn(5, dim)
        out = swd(cc, signal, diffusion_time=torch.tensor(0.5),
                  wave_damping=torch.tensor(0.1))
        assert out.shape == (5, dim)
        assert torch.isfinite(out).all()

    def test_sheaf_wave_dynamics_gradient_flow(self):
        """Gradients flow through SheafWaveDynamics to restriction_net."""
        dim = 16
        cc = make_chain(dim=dim, length=4)
        swd = SheafWaveDynamics(dim, use_wave_strength_gate=True)
        signal = torch.randn(4, dim)
        out = swd(cc, signal, diffusion_time=torch.tensor(0.5),
                  wave_damping=torch.tensor(0.1))
        out.sum().backward()
        # Check that restriction_net parameters got gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in swd.restriction_net.parameters()
        )
        assert has_grad, "No gradient reached the restriction net"

    def test_sheaf_wave_dynamics_gate_bypass(self):
        """When wave_strength → -inf, output ≈ original signal."""
        dim = 16
        cc = make_chain(dim=dim, length=5)
        swd = SheafWaveDynamics(dim, use_wave_strength_gate=True)
        signal = torch.randn(5, dim)
        with torch.no_grad():
            swd.wave_strength.fill_(-100.0)
        out = swd(cc, signal, diffusion_time=torch.tensor(0.5),
                  wave_damping=torch.tensor(0.1))
        assert torch.allclose(out, signal, atol=1e-4)

    def test_sheaf_wave_dynamics_no_edges(self):
        """SheafWaveDynamics handles a graph with zero edges."""
        dim = 16
        cc = CellComplex(embedding_dim=dim)
        for _ in range(3):
            cc.add_0_cell(torch.randn(dim), "node")
        swd = SheafWaveDynamics(dim)
        signal = cc.get_embeddings(0)
        out = swd(cc, signal, diffusion_time=torch.tensor(0.5),
                  wave_damping=torch.tensor(0.1))
        # With no edges, sheaf Laplacian is zero → diffusion is identity
        assert torch.allclose(out, signal, atol=1e-5)

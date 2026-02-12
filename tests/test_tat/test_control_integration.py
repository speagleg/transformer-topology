"""Tests for GNN executive control signal integration with TAT layers."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.control_head import ControlSignal
from src.tat.transformer import TopologyAwareTransformer


def make_chain(dim=32, length=5):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "node") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "edge")
    return cc


def make_control_signal(num_nodes: int, num_freqs: int,
                        requires_grad: bool = False) -> ControlSignal:
    """Create a ControlSignal with non-trivial values for testing."""
    return ControlSignal(
        frequency_gate=torch.rand(num_freqs, requires_grad=requires_grad),
        spatial_focus=torch.rand(num_nodes, requires_grad=requires_grad),
        confidence_weights=torch.rand(num_nodes),
        diffusion_time=torch.tensor(1.0),
        wave_damping=torch.tensor(0.5),
    )


class TestSpatialFocus:
    def test_spatial_focus_changes_output(self):
        """Output should differ when spatial_focus is applied vs. omitted."""
        torch.manual_seed(42)
        dim, length, num_freqs = 32, 5, 4
        cc = make_chain(dim=dim, length=length)

        tat = TopologyAwareTransformer(
            embedding_dim=dim, num_layers=1, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=64, num_freqs=num_freqs,
        )
        tat.eval()

        with torch.no_grad():
            out_no_ctrl = tat(cc)

            # Build a control signal with spatial focus only (frequency gate = all ones = no-op)
            ctrl = ControlSignal(
                frequency_gate=torch.ones(num_freqs),
                spatial_focus=torch.rand(length),
                confidence_weights=torch.ones(length),
                diffusion_time=torch.tensor(1.0),
                wave_damping=torch.tensor(0.5),
            )
            out_with_ctrl = tat(cc, control_signal=ctrl)

        assert not torch.allclose(out_no_ctrl, out_with_ctrl, atol=1e-5), (
            "Spatial focus should change the output"
        )


class TestFrequencyGate:
    def test_frequency_gate_changes_output(self):
        """Output should differ when frequency_gate attenuates spectral bands."""
        torch.manual_seed(42)
        dim, length, num_freqs = 32, 5, 4
        cc = make_chain(dim=dim, length=length)

        tat = TopologyAwareTransformer(
            embedding_dim=dim, num_layers=1, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=64, num_freqs=num_freqs,
        )
        tat.eval()

        with torch.no_grad():
            out_no_ctrl = tat(cc)

            # Build a control signal with frequency gate only (spatial focus = None-equivalent zeros)
            ctrl = ControlSignal(
                frequency_gate=torch.tensor([1.0, 0.0, 0.0, 0.0]),  # kill 3 of 4 bands
                spatial_focus=torch.zeros(length),  # zero bias = no spatial effect
                confidence_weights=torch.ones(length),
                diffusion_time=torch.tensor(1.0),
                wave_damping=torch.tensor(0.5),
            )
            out_with_ctrl = tat(cc, control_signal=ctrl)

        assert not torch.allclose(out_no_ctrl, out_with_ctrl, atol=1e-5), (
            "Frequency gate should change the output"
        )


class TestGradientThroughControl:
    def test_gradient_through_control_path(self):
        """Gradients should flow back through control signal tensors."""
        torch.manual_seed(42)
        dim, length, num_freqs = 32, 5, 4
        cc = make_chain(dim=dim, length=length)

        tat = TopologyAwareTransformer(
            embedding_dim=dim, num_layers=1, num_spatial_heads=2,
            num_spectral_heads=2, ff_dim=64, num_freqs=num_freqs,
        )

        ctrl = make_control_signal(
            num_nodes=length, num_freqs=num_freqs, requires_grad=True,
        )

        out = tat(cc, control_signal=ctrl)
        loss = out.sum()
        loss.backward()

        assert ctrl.spatial_focus.grad is not None, (
            "Gradient should flow through spatial_focus"
        )
        assert ctrl.spatial_focus.grad.abs().sum() > 0, (
            "spatial_focus gradient should be non-zero"
        )

        assert ctrl.frequency_gate.grad is not None, (
            "Gradient should flow through frequency_gate"
        )
        assert ctrl.frequency_gate.grad.abs().sum() > 0, (
            "frequency_gate gradient should be non-zero"
        )

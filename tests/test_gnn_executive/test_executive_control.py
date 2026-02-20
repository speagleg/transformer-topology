import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.executive import GNNExecutive
from src.gnn_executive.control_head import ControlSignal


def make_chain(dim: int = 32, length: int = 5) -> CellComplex:
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "node") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "edge")
    return cc


DIM = 16
HIDDEN = 32
SPATIAL_LAYERS = 2
SPECTRAL_LAYERS = 2
MAX_FREQS = 5
CHAIN_LEN = 5


def _make_executive(*, control: bool = False) -> GNNExecutive:
    return GNNExecutive(
        embedding_dim=DIM, hidden_dim=HIDDEN,
        num_spatial_layers=SPATIAL_LAYERS,
        num_spectral_layers=SPECTRAL_LAYERS,
        max_freqs=MAX_FREQS,
        produce_control_signals=control,
    )


class TestForwardBackwardCompat:
    """Ensure forward() returns (Tensor, Tensor|None) regardless of control flag."""

    def test_forward_without_control(self):
        executive = _make_executive(control=False)
        cc = make_chain(dim=DIM, length=CHAIN_LEN)
        result = executive(cc)
        assert len(result) == 2
        node_out, edge_out = result
        assert isinstance(node_out, torch.Tensor)
        assert node_out.shape == (CHAIN_LEN, DIM)
        assert edge_out is None

    def test_forward_with_control_enabled(self):
        """forward() still returns 2-tuple even when control is enabled."""
        executive = _make_executive(control=True)
        cc = make_chain(dim=DIM, length=CHAIN_LEN)
        result = executive(cc)
        assert len(result) == 2
        node_out, edge_out = result
        assert isinstance(node_out, torch.Tensor)
        assert node_out.shape == (CHAIN_LEN, DIM)
        assert edge_out is None

    def test_forward_output_unchanged_by_control_flag(self):
        """Node embeddings from forward() are identical with same weights."""
        executive = _make_executive(control=True)
        cc = make_chain(dim=DIM, length=CHAIN_LEN)
        node_a, edge_a = executive(cc)
        node_b, edge_b = executive(cc)
        assert torch.allclose(node_a, node_b, atol=1e-6)


class TestForwardWithControl:
    """Tests for the new forward_with_control() method."""

    def test_returns_three_tuple(self):
        executive = _make_executive(control=True)
        cc = make_chain(dim=DIM, length=CHAIN_LEN)
        result = executive.forward_with_control(cc)
        assert len(result) == 3
        node_out, edge_out, ctrl = result
        assert isinstance(node_out, torch.Tensor)
        assert edge_out is None
        assert isinstance(ctrl, ControlSignal)

    def test_control_signal_shapes(self):
        executive = _make_executive(control=True)
        cc = make_chain(dim=DIM, length=CHAIN_LEN)
        _, _, ctrl = executive.forward_with_control(cc)
        assert ctrl.frequency_gate.shape == (MAX_FREQS,)
        assert ctrl.spatial_focus.shape == (CHAIN_LEN,)
        assert ctrl.confidence_weights.shape == (CHAIN_LEN,)
        assert ctrl.diffusion_time.shape == ()
        assert ctrl.wave_damping.shape == ()

    def test_control_signal_value_ranges(self):
        executive = _make_executive(control=True)
        cc = make_chain(dim=DIM, length=CHAIN_LEN)
        _, _, ctrl = executive.forward_with_control(cc)
        # Sigmoid outputs in [0, 1]
        assert (ctrl.frequency_gate >= 0).all() and (ctrl.frequency_gate <= 1).all()
        assert (ctrl.spatial_focus >= 0).all() and (ctrl.spatial_focus <= 1).all()
        assert (ctrl.confidence_weights >= 0).all() and (ctrl.confidence_weights <= 1).all()
        # Softplus outputs are positive
        assert ctrl.diffusion_time > 0
        assert ctrl.wave_damping > 0

    def test_with_harmonic_energy(self):
        executive = _make_executive(control=True)
        cc = make_chain(dim=DIM, length=CHAIN_LEN)
        _, _, ctrl_without = executive.forward_with_control(cc)
        _, _, ctrl_with = executive.forward_with_control(
            cc, harmonic_energy=torch.tensor(5.0),
        )
        # Harmonic energy should influence the frequency gate
        assert not torch.allclose(
            ctrl_without.frequency_gate, ctrl_with.frequency_gate,
        )

    def test_node_embeddings_match_forward(self):
        """forward_with_control() node output should match forward() output."""
        executive = _make_executive(control=True)
        cc = make_chain(dim=DIM, length=CHAIN_LEN)
        node_fwd, edge_fwd = executive(cc)
        node_ctrl, edge_ctrl, _ = executive.forward_with_control(cc)
        assert torch.allclose(node_fwd, node_ctrl, atol=1e-6)

    def test_raises_without_control_flag(self):
        """forward_with_control() should raise when control is not enabled."""
        executive = _make_executive(control=False)
        cc = make_chain(dim=DIM, length=CHAIN_LEN)
        with pytest.raises(RuntimeError, match="produce_control_signals"):
            executive.forward_with_control(cc)


class TestGradientFlow:
    """Verify gradients propagate through the control signal path."""

    def test_gradients_reach_all_control_head_params(self):
        executive = _make_executive(control=True)
        cc = make_chain(dim=DIM, length=CHAIN_LEN)
        _, _, ctrl = executive.forward_with_control(cc)
        loss = (
            ctrl.frequency_gate.sum()
            + ctrl.spatial_focus.sum()
            + ctrl.confidence_weights.sum()
            + ctrl.diffusion_time
            + ctrl.wave_damping
            + ctrl.semantic_weight
        )
        loss.backward()
        # Every parameter in the control head should have a gradient
        for name, p in executive.control_head.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for control_head.{name}"
                assert p.grad.abs().sum() > 0, f"Zero gradient for control_head.{name}"

    def test_gradients_reach_gnn_through_control(self):
        """Control head loss should back-propagate into the spatial/spectral GNNs."""
        executive = _make_executive(control=True)
        cc = make_chain(dim=DIM, length=CHAIN_LEN)
        _, _, ctrl = executive.forward_with_control(cc)
        loss = ctrl.spatial_focus.sum() + ctrl.confidence_weights.sum()
        loss.backward()
        # Spatial GNN should receive gradients via fused -> control_head path
        spatial_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in executive.spatial_gnn.parameters()
            if p.requires_grad
        )
        assert spatial_has_grad, "No gradient reached spatial GNN from control path"
        # Spectral GNN should also receive gradients
        spectral_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in executive.spectral_gnn.parameters()
            if p.requires_grad
        )
        assert spectral_has_grad, "No gradient reached spectral GNN from control path"

    def test_gradients_through_frequency_gate(self):
        """Frequency gate specifically should carry gradients back."""
        executive = _make_executive(control=True)
        cc = make_chain(dim=DIM, length=CHAIN_LEN)
        _, _, ctrl = executive.forward_with_control(cc)
        loss = ctrl.frequency_gate.sum()
        loss.backward()
        # The trunk of control_head feeds frequency gate
        trunk_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in executive.control_head.trunk.parameters()
        )
        assert trunk_has_grad, "No gradient through control head trunk"


class TestDifferentGraphSizes:
    """Control signals should adapt to varying graph sizes."""

    @pytest.mark.parametrize("length", [3, 7, 12])
    def test_variable_chain_length(self, length: int):
        executive = _make_executive(control=True)
        cc = make_chain(dim=DIM, length=length)
        node_out, _, ctrl = executive.forward_with_control(cc)
        assert node_out.shape == (length, DIM)
        assert ctrl.spatial_focus.shape == (length,)
        assert ctrl.confidence_weights.shape == (length,)
        # Global signals should be fixed shape regardless of graph size
        assert ctrl.frequency_gate.shape == (MAX_FREQS,)
        assert ctrl.diffusion_time.shape == ()
        assert ctrl.wave_damping.shape == ()

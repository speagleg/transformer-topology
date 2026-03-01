"""Integration test: MetaCognitiveController wired into GNNExecutive."""
import torch
from src.gnn_executive.executive import GNNExecutive
from src.gnn_executive.control_head import ControlSignal
from src.cell_complex.cell_complex import CellComplex


def _make_cc(n=10, dim=32):
    cc = CellComplex(dim)
    for i in range(n):
        cc.add_0_cell(torch.randn(dim), f"node_{i}")
    for i in range(n - 1):
        cc.add_1_cell(i, i + 1, torch.randn(dim), "edge")
    return cc


def test_gnn_executive_metacog_mode():
    """GNNExecutive with use_metacog=True should return metacog control signals."""
    exe = GNNExecutive(
        embedding_dim=32, hidden_dim=64,
        num_spatial_layers=2, num_spectral_layers=2,
        max_freqs=8, produce_control_signals=True,
        num_filters=4, use_topo_feedback=True,
        use_embedding_topo_feedback=True,
        use_metacog=True, num_tasks=19,
    )
    cc = _make_cc()
    fused, edge_out, control = exe.forward_with_control(cc, task_id=3)
    assert fused.shape == (10, 32)
    assert control.text_gate is not None
    assert control.strategy_weights is not None
    assert control.strategy_weights.shape == (4,)


def test_gnn_executive_metacog_off_by_default():
    """Without use_metacog, loop should work exactly as before (no task_id)."""
    exe = GNNExecutive(
        embedding_dim=32, hidden_dim=64,
        num_spatial_layers=2, num_spectral_layers=2,
        max_freqs=8, produce_control_signals=True,
    )
    cc = _make_cc()
    fused, edge_out, control = exe.forward_with_control(cc)
    assert control.text_gate is None


def test_gnn_executive_metacog_with_iteration_context():
    """forward_with_control should accept and pass iteration_context."""
    exe = GNNExecutive(
        embedding_dim=32, hidden_dim=64,
        num_spatial_layers=2, num_spectral_layers=2,
        max_freqs=8, produce_control_signals=True,
        use_metacog=True, num_tasks=19,
    )
    cc = _make_cc()
    iter_ctx = torch.tensor([0.5, 0.3, 0.01])
    fused, edge_out, control = exe.forward_with_control(
        cc, task_id=5, iteration_context=iter_ctx,
    )
    assert isinstance(control, ControlSignal)
    assert control.uncertainty is not None

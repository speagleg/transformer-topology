"""Tests for metacognition integration in ExecutiveReasoningLoop."""
import torch
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop


def _make_cc(n=10, dim=32):
    cc = CellComplex(dim)
    for i in range(n):
        cc.add_0_cell(torch.randn(dim), f"node_{i}")
    for i in range(n - 1):
        cc.add_1_cell(i, i + 1, torch.randn(dim), "edge")
    return cc


def test_loop_accepts_task_id():
    """ExecutiveReasoningLoop.forward() should accept task_id kwarg."""
    loop = ExecutiveReasoningLoop(
        embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
        gnn_spectral_layers=2, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_iterations=2, use_wave_dynamics=False,
        use_metacog=True, num_tasks=19,
    )
    cc = _make_cc()
    output, num_iters, diagnostics = loop(cc, task_id=5)
    assert output.shape == (10, 32)
    cs = diagnostics['control_signals'][-1]
    assert cs.text_gate is not None


def test_loop_metacog_off_compat():
    """Without use_metacog, loop should work exactly as before."""
    loop = ExecutiveReasoningLoop(
        embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
        gnn_spectral_layers=2, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_iterations=2, use_wave_dynamics=False,
    )
    cc = _make_cc()
    output, num_iters, diagnostics = loop(cc)
    assert output.shape == (10, 32)
    cs = diagnostics['control_signals'][-1]
    assert cs.text_gate is None


def test_loop_batched_accepts_task_id():
    """forward_batched() should accept task_id."""
    loop = ExecutiveReasoningLoop(
        embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
        gnn_spectral_layers=2, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_iterations=2, use_wave_dynamics=False,
        use_metacog=True, num_tasks=19,
    )
    ccs = [_make_cc() for _ in range(3)]
    results = loop.forward_batched(ccs, task_id=3)
    assert len(results) == 3
    for embs, n_iters, diag in results:
        assert embs.shape == (10, 32)
        assert diag['control_signals'][-1].text_gate is not None

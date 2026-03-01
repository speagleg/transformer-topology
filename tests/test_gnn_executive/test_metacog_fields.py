"""Tests for MetaCognitive Controller ControlSignal extensions."""
import torch
from src.gnn_executive.control_head import ControlSignal


def test_control_signal_has_metacog_fields():
    """ControlSignal should accept new metacog fields with None defaults."""
    cs = ControlSignal(
        frequency_gate=torch.randn(8),
        spatial_focus=torch.randn(20),
        confidence_weights=torch.randn(20),
        diffusion_time=torch.tensor(1.0),
        wave_damping=torch.tensor(0.5),
    )
    assert cs.text_gate is None
    assert cs.structure_gate is None
    assert cs.uncertainty is None
    assert cs.iteration_budget is None
    assert cs.strategy_weights is None


def test_control_signal_metacog_fields_populated():
    """ControlSignal should store metacog fields when provided."""
    cs = ControlSignal(
        frequency_gate=torch.randn(8),
        spatial_focus=torch.randn(20),
        confidence_weights=torch.randn(20),
        diffusion_time=torch.tensor(1.0),
        wave_damping=torch.tensor(0.5),
        text_gate=torch.tensor(0.8),
        structure_gate=torch.tensor(0.3),
        uncertainty=torch.tensor(0.2),
        iteration_budget=torch.tensor(3.0),
        strategy_weights=torch.softmax(torch.randn(4), dim=0),
    )
    assert abs(cs.text_gate.item() - 0.8) < 1e-5
    assert abs(cs.structure_gate.item() - 0.3) < 1e-5
    assert abs(cs.uncertainty.item() - 0.2) < 1e-5
    assert abs(cs.iteration_budget.item() - 3.0) < 1e-5
    assert cs.strategy_weights.shape == (4,)
    assert abs(cs.strategy_weights.sum().item() - 1.0) < 1e-5

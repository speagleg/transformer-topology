"""Tests for metacognition auxiliary losses."""
import torch
from src.gnn_executive.control_head import ControlSignal


def _make_control(text_gate=0.7, uncertainty=0.3):
    return ControlSignal(
        frequency_gate=torch.randn(8),
        spatial_focus=torch.randn(20),
        confidence_weights=torch.randn(20),
        diffusion_time=torch.tensor(1.0),
        wave_damping=torch.tensor(0.5),
        text_gate=torch.tensor(text_gate, requires_grad=True),
        structure_gate=torch.tensor(0.8, requires_grad=True),
        uncertainty=torch.tensor(uncertainty, requires_grad=True),
        iteration_budget=torch.tensor(3.0, requires_grad=True),
        strategy_weights=torch.softmax(torch.randn(4, requires_grad=True), dim=0),
    )


def test_calibration_loss():
    from src.training.metacog_losses import calibration_loss
    control = _make_control(uncertainty=0.3)
    is_correct = torch.tensor(1.0)
    temp = torch.tensor(1.5)
    loss = calibration_loss(control, is_correct, temp)
    assert loss.dim() == 0
    assert loss.item() >= 0.0
    loss.backward()
    assert control.uncertainty.grad is not None


def test_calibration_loss_correct_direction():
    """When model is correct and confident, loss should be low."""
    from src.training.metacog_losses import calibration_loss
    temp = torch.tensor(1.5)
    c_correct = _make_control(uncertainty=0.1)
    loss_correct = calibration_loss(c_correct, torch.tensor(1.0), temp)
    c_wrong = _make_control(uncertainty=0.1)
    loss_wrong = calibration_loss(c_wrong, torch.tensor(0.0), temp)
    assert loss_correct.item() < loss_wrong.item()


def test_calibration_loss_none_uncertainty():
    """Should return 0 when uncertainty is None."""
    from src.training.metacog_losses import calibration_loss
    cs = ControlSignal(
        frequency_gate=torch.randn(8),
        spatial_focus=torch.randn(20),
        confidence_weights=torch.randn(20),
        diffusion_time=torch.tensor(1.0),
        wave_damping=torch.tensor(0.5),
    )
    loss = calibration_loss(cs, torch.tensor(1.0), torch.tensor(1.5))
    assert loss.item() == 0.0


def test_efficiency_loss():
    from src.training.metacog_losses import efficiency_loss
    control = _make_control(uncertainty=0.2)
    loss = efficiency_loss(control, num_iters_used=4)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_efficiency_loss_none():
    """Should return 0 when no metacog fields."""
    from src.training.metacog_losses import efficiency_loss
    cs = ControlSignal(
        frequency_gate=torch.randn(8),
        spatial_focus=torch.randn(20),
        confidence_weights=torch.randn(20),
        diffusion_time=torch.tensor(1.0),
        wave_damping=torch.tensor(0.5),
    )
    loss = efficiency_loss(cs, num_iters_used=4)
    assert loss.item() == 0.0


def test_gating_diversity_loss():
    from src.training.metacog_losses import gating_diversity_loss
    controls = [_make_control(text_gate=tg) for tg in [0.1, 0.9, 0.5, 0.3]]
    loss = gating_diversity_loss(controls)
    assert loss.dim() == 0
    assert loss.item() <= 0.0


def test_gating_diversity_loss_uniform_is_worst():
    """If all text_gates are the same, diversity loss should be 0 (worst)."""
    from src.training.metacog_losses import gating_diversity_loss
    controls = [_make_control(text_gate=0.5) for _ in range(4)]
    loss = gating_diversity_loss(controls)
    assert abs(loss.item()) < 1e-5


def test_gating_diversity_loss_few_samples():
    """Should return 0 with fewer than 2 samples."""
    from src.training.metacog_losses import gating_diversity_loss
    loss = gating_diversity_loss([_make_control()])
    assert loss.item() == 0.0

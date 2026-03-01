"""Auxiliary losses for MetaCognitive Controller training.

- calibration_loss: predicted confidence should match actual correctness
- efficiency_loss: penalize excess iterations when confident
- gating_diversity_loss: encourage different text_gate values across tasks
"""

import torch
import torch.nn.functional as F
from src.gnn_executive.control_head import ControlSignal


def calibration_loss(control: ControlSignal, is_correct: torch.Tensor,
                     confidence_temperature: torch.Tensor) -> torch.Tensor:
    """Confidence calibration: predicted confidence should match correctness.

    Args:
        control: ControlSignal with uncertainty field.
        is_correct: scalar 0.0 or 1.0.
        confidence_temperature: learned temperature parameter.

    Returns:
        Scalar calibration loss + temperature regularization.
    """
    if control.uncertainty is None:
        return torch.tensor(0.0, device=is_correct.device)

    predicted_confidence = 1.0 - control.uncertainty
    predicted_confidence = predicted_confidence.clamp(1e-6, 1 - 1e-6)
    bce = F.binary_cross_entropy(predicted_confidence, is_correct)

    temp_reg = -0.01 * torch.log(confidence_temperature.clamp(min=0.1))
    return bce + temp_reg


def efficiency_loss(control: ControlSignal, num_iters_used: int) -> torch.Tensor:
    """Iteration efficiency: penalize excess iterations when confident.

    Args:
        control: ControlSignal with iteration_budget and uncertainty fields.
        num_iters_used: actual number of iterations executed.

    Returns:
        Scalar efficiency loss.
    """
    if control.iteration_budget is None or control.uncertainty is None:
        return torch.tensor(0.0)

    predicted_confidence = 1.0 - control.uncertainty
    excess = max(0.0, num_iters_used - control.iteration_budget.item())
    excess_t = torch.tensor(excess, device=control.iteration_budget.device)
    return excess_t.pow(2) * predicted_confidence


def gating_diversity_loss(controls: list[ControlSignal]) -> torch.Tensor:
    """Gating diversity: encourage variance in text_gate across tasks.

    Returns negative variance (minimize to maximize diversity).

    Args:
        controls: list of ControlSignal from different tasks in a batch.

    Returns:
        Scalar negative variance of text_gate values.
    """
    text_gates = []
    for c in controls:
        if c.text_gate is not None:
            text_gates.append(c.text_gate)

    if len(text_gates) < 2:
        return torch.tensor(0.0)

    stacked = torch.stack(text_gates)
    return -torch.var(stacked)

"""Focal loss for imbalanced classification with GNN/LLM integration.

Standard focal loss re-weights the cross-entropy by (1 - p_t)^gamma,
focusing gradient signal on hard/misclassified examples. Compatible
with class_weights and label_smoothing.

Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
"""

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute focal loss.

    Args:
        logits: (B, C) raw logits
        targets: (B,) integer class labels
        gamma: focusing parameter (0 = standard CE, 2 = typical focal)
        alpha: optional (C,) per-class weights
        label_smoothing: label smoothing factor (0-1)

    Returns:
        Scalar loss (mean-reduced).
    """
    ce = F.cross_entropy(
        logits, targets, weight=alpha,
        label_smoothing=label_smoothing, reduction='none',
    )
    pt = torch.exp(-ce)
    focal_weight = (1 - pt) ** gamma
    return (focal_weight * ce).mean()

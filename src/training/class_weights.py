"""Compute class weights for imbalanced datasets."""
import torch


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to sum to num_classes.

    Missing classes get max weight (treated as count=1).
    """
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()

    counts = counts.clamp(min=1)

    weights = 1.0 / counts
    weights = weights * num_classes / weights.sum()
    return weights

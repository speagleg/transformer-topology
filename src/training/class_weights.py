"""Compute class weights for imbalanced datasets."""
import torch


def compute_class_weights(labels: torch.Tensor, num_classes: int,
                          max_weight_ratio: float = 10.0) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to sum to num_classes.

    Missing classes get weight 0.0 (no samples → no contribution to loss).
    Present classes are inverse-frequency weighted, clamped so no weight
    exceeds max_weight_ratio × the minimum present-class weight, then
    normalized so present-class weights sum to num_classes.
    """
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()

    present = counts > 0
    weights = torch.zeros(num_classes)
    weights[present] = 1.0 / counts[present]
    # Clamp extreme ratios (e.g. class with 1 sample vs class with 2000)
    if present.sum() > 1:
        min_w = weights[present].min()
        weights = weights.clamp(max=min_w * max_weight_ratio)
    # Normalize so present-class weights sum to num_classes
    if weights.sum() > 0:
        weights = weights * num_classes / weights.sum()
    return weights

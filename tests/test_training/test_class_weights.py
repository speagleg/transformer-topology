"""Tests for class weight computation."""
import torch
from src.training.class_weights import compute_class_weights


class TestClassWeights:
    def test_uniform_distribution(self):
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        weights = compute_class_weights(labels, num_classes=3)
        assert weights.shape == (3,)
        assert torch.allclose(weights[0], weights[1], atol=1e-5)

    def test_imbalanced_distribution(self):
        labels = torch.tensor([0, 0, 0, 0, 1])
        weights = compute_class_weights(labels, num_classes=2)
        assert weights[1] > weights[0]

    def test_missing_class_gets_max_weight(self):
        labels = torch.tensor([0, 0, 1, 1])
        weights = compute_class_weights(labels, num_classes=3)
        assert weights[2] >= weights[0]
        assert weights[2] >= weights[1]

    def test_normalized(self):
        labels = torch.tensor([0, 0, 0, 1, 2, 2])
        weights = compute_class_weights(labels, num_classes=3)
        assert abs(weights.sum().item() - 3.0) < 1e-5

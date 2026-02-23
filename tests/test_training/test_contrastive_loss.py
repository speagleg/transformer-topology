"""Tests for semantic contrastive loss."""
import torch
from src.training.contrastive_loss import SemanticContrastiveLoss


class TestSemanticContrastiveLoss:
    def test_returns_scalar(self):
        loss_fn = SemanticContrastiveLoss(temperature=0.1)
        features = torch.randn(10, 32)
        adjacency = torch.randint(0, 2, (10, 10)).float()
        loss = loss_fn(features, adjacency)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_identical_features_low_loss(self):
        loss_fn = SemanticContrastiveLoss(temperature=0.1)
        features = torch.ones(5, 32)
        adjacency = torch.ones(5, 5)
        loss = loss_fn(features, adjacency)
        assert loss.item() >= 0

    def test_gradient_flows(self):
        loss_fn = SemanticContrastiveLoss(temperature=0.1)
        features = torch.randn(8, 32, requires_grad=True)
        adjacency = torch.randint(0, 2, (8, 8)).float()
        loss = loss_fn(features, adjacency)
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_no_positive_pairs_returns_zero(self):
        loss_fn = SemanticContrastiveLoss(temperature=0.1)
        features = torch.randn(5, 32)
        adjacency = torch.zeros(5, 5)
        loss = loss_fn(features, adjacency)
        assert loss.item() == 0.0

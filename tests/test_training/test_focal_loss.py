"""Tests for focal loss implementation."""

import torch
import torch.nn.functional as F
import pytest

from src.training.focal_loss import focal_loss


class TestFocalLoss:
    """Test focal loss correctness and properties."""

    def test_gamma_zero_equals_cross_entropy(self):
        """With gamma=0, focal loss should equal standard cross-entropy."""
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        fl = focal_loss(logits, targets, gamma=0.0)
        ce = F.cross_entropy(logits, targets)
        assert torch.allclose(fl, ce, atol=1e-5)

    def test_gamma_zero_with_smoothing(self):
        """With gamma=0 and label_smoothing, should match smoothed CE."""
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        fl = focal_loss(logits, targets, gamma=0.0, label_smoothing=0.1)
        ce = F.cross_entropy(logits, targets, label_smoothing=0.1)
        assert torch.allclose(fl, ce, atol=1e-5)

    def test_focal_loss_lower_than_ce_for_easy_examples(self):
        """Focal loss should be lower than CE when predictions are confident."""
        # Create easy examples: logits strongly favor correct class
        logits = torch.zeros(8, 5)
        logits[range(8), torch.zeros(8, dtype=torch.long)] = 10.0
        targets = torch.zeros(8, dtype=torch.long)

        fl = focal_loss(logits, targets, gamma=2.0)
        ce = F.cross_entropy(logits, targets)
        assert fl < ce

    def test_focal_loss_similar_for_hard_examples(self):
        """For uncertain predictions, focal and CE should be closer."""
        # Random logits = hard examples
        torch.manual_seed(42)
        logits = torch.randn(100, 10) * 0.1  # very uncertain
        targets = torch.randint(0, 10, (100,))

        fl = focal_loss(logits, targets, gamma=2.0)
        ce = F.cross_entropy(logits, targets)
        # Focal should be slightly lower but not dramatically
        ratio = fl / ce
        assert 0.3 < ratio < 1.0

    def test_with_class_weights(self):
        """Should accept class weights (alpha parameter)."""
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        weights = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0])
        fl = focal_loss(logits, targets, gamma=2.0, alpha=weights)
        assert torch.isfinite(fl)

    def test_gradient_flows(self):
        """Gradients should flow through focal loss."""
        logits = torch.randn(4, 10, requires_grad=True)
        targets = torch.randint(0, 10, (4,))
        fl = focal_loss(logits, targets, gamma=2.0)
        fl.backward()
        assert logits.grad is not None
        assert torch.all(torch.isfinite(logits.grad))

    def test_single_sample(self):
        """Should work with batch size 1."""
        logits = torch.randn(1, 10)
        targets = torch.tensor([3])
        fl = focal_loss(logits, targets, gamma=2.0)
        assert torch.isfinite(fl)

    def test_higher_gamma_reduces_easy_loss_more(self):
        """Higher gamma should reduce loss more for easy (confident) examples."""
        logits = torch.zeros(8, 5)
        logits[range(8), torch.zeros(8, dtype=torch.long)] = 10.0
        targets = torch.zeros(8, dtype=torch.long)

        fl_1 = focal_loss(logits, targets, gamma=1.0)
        fl_2 = focal_loss(logits, targets, gamma=2.0)
        fl_5 = focal_loss(logits, targets, gamma=5.0)
        assert fl_5 < fl_2 < fl_1

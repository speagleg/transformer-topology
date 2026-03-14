"""Test spectral gap regularization loss."""
import torch
import pytest


def test_spectral_gap_loss_penalizes_small_gap():
    from src.training.spectral_reg import spectral_gap_loss
    small_gap = torch.tensor(0.01)
    large_gap = torch.tensor(0.5)
    loss_small = spectral_gap_loss(small_gap)
    loss_large = spectral_gap_loss(large_gap)
    assert loss_small > loss_large


def test_spectral_gap_loss_is_differentiable():
    from src.training.spectral_reg import spectral_gap_loss
    gap = torch.tensor(0.1, requires_grad=True)
    loss = spectral_gap_loss(gap)
    loss.backward()
    assert gap.grad is not None


def test_compute_batch_spectral_gap():
    from src.training.spectral_reg import compute_batch_spectral_gap
    from src.cell_complex.cell_complex import CellComplex
    cc = CellComplex(embedding_dim=32)
    for i in range(5):
        cc.add_0_cell(torch.randn(32), cell_type='node')
    cc.add_1_cell(0, 1, torch.randn(32), relation_type='edge')
    cc.add_1_cell(1, 2, torch.randn(32), relation_type='edge')
    cc.add_1_cell(2, 3, torch.randn(32), relation_type='edge')
    gap = compute_batch_spectral_gap([cc])
    assert gap.shape == ()
    assert gap.item() >= 0

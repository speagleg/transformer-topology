"""Tests for filter_weights field in ControlSignal and ControlHead."""

import torch
import pytest

from src.gnn_executive.control_head import ControlSignal, ControlHead


class TestFilterWeightsField:
    def test_filter_weights_field_defaults_none(self):
        """Backward compat: filter_weights can be omitted."""
        cs = ControlSignal(
            frequency_gate=torch.rand(8),
            spatial_focus=torch.rand(5),
            confidence_weights=torch.rand(5),
            diffusion_time=torch.tensor(1.0),
            wave_damping=torch.tensor(0.5),
        )
        assert cs.filter_weights is None

    def test_filter_weights_field_set(self):
        cs = ControlSignal(
            frequency_gate=torch.rand(8),
            spatial_focus=torch.rand(5),
            confidence_weights=torch.rand(5),
            diffusion_time=torch.tensor(1.0),
            wave_damping=torch.tensor(0.5),
            filter_weights=torch.tensor([0.25, 0.25, 0.25, 0.25]),
        )
        assert cs.filter_weights.shape == (4,)


class TestFilterWeightsHead:
    def test_no_filter_weights_when_num_filters_zero(self):
        """Default ControlHead (num_filters=0) produces None."""
        head = ControlHead(embedding_dim=32, num_freqs=8, num_filters=0)
        cs = head(torch.randn(5, 32))
        assert cs.filter_weights is None

    def test_filter_weights_produced_when_num_filters_set(self):
        head = ControlHead(embedding_dim=32, num_freqs=8, num_filters=4)
        cs = head(torch.randn(5, 32))
        assert cs.filter_weights is not None
        assert cs.filter_weights.shape == (4,)

    def test_filter_weights_softmax_sums_to_one(self):
        head = ControlHead(embedding_dim=32, num_freqs=8, num_filters=4)
        for _ in range(5):
            cs = head(torch.randn(7, 32))
            assert torch.allclose(cs.filter_weights.sum(),
                                  torch.tensor(1.0), atol=1e-5)

    def test_filter_weights_gradient_flow(self):
        head = ControlHead(embedding_dim=32, num_freqs=8, num_filters=4)
        emb = torch.randn(5, 32, requires_grad=True)
        cs = head(emb)
        loss = cs.filter_weights.sum()
        loss.backward()
        assert emb.grad is not None
        for p in head.filter_weights_head.parameters():
            assert p.grad is not None

    def test_filter_weights_varies_with_input(self):
        head = ControlHead(embedding_dim=32, num_freqs=8, num_filters=4)
        cs1 = head(torch.randn(5, 32))
        cs2 = head(torch.randn(5, 32))
        assert not torch.allclose(cs1.filter_weights, cs2.filter_weights)

    def test_existing_fields_unchanged(self):
        """Adding num_filters doesn't break existing signal production."""
        head = ControlHead(embedding_dim=32, num_freqs=8, num_filters=4)
        emb = torch.randn(5, 32)
        cs = head(emb)
        assert cs.frequency_gate.shape == (8,)
        assert cs.spatial_focus.shape == (5,)
        assert cs.confidence_weights.shape == (5,)
        assert cs.diffusion_time.shape == ()
        assert cs.wave_damping.shape == ()
        assert cs.semantic_weight.shape == ()

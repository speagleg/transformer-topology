"""Tests for semantic_weight field in ControlSignal and ControlHead."""

import torch
import pytest
from src.gnn_executive.control_head import ControlSignal, ControlHead


class TestSemanticWeightField:
    def test_control_signal_has_semantic_weight(self):
        cs = ControlSignal(
            frequency_gate=torch.rand(8),
            spatial_focus=torch.rand(5),
            confidence_weights=torch.rand(5),
            diffusion_time=torch.tensor(1.0),
            wave_damping=torch.tensor(0.5),
            semantic_weight=torch.tensor(0.7),
        )
        assert cs.semantic_weight.shape == ()
        assert cs.semantic_weight.item() == pytest.approx(0.7)

    def test_semantic_weight_defaults_to_none(self):
        cs = ControlSignal(
            frequency_gate=torch.rand(8),
            spatial_focus=torch.rand(5),
            confidence_weights=torch.rand(5),
            diffusion_time=torch.tensor(1.0),
            wave_damping=torch.tensor(0.5),
        )
        assert cs.semantic_weight is None

    def test_existing_fields_unchanged(self):
        cs = ControlSignal(
            frequency_gate=torch.ones(8),
            spatial_focus=torch.ones(5) * 0.3,
            confidence_weights=torch.ones(5) * 0.9,
            diffusion_time=torch.tensor(2.0),
            wave_damping=torch.tensor(1.5),
            semantic_weight=torch.tensor(0.5),
        )
        assert cs.frequency_gate.shape == (8,)
        assert cs.spatial_focus.shape == (5,)
        assert cs.confidence_weights.shape == (5,)
        assert cs.diffusion_time.item() == pytest.approx(2.0)
        assert cs.wave_damping.item() == pytest.approx(1.5)


class TestSemanticWeightHead:
    def test_control_head_produces_semantic_weight(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        node_emb = torch.randn(5, 32)
        cs = head(node_emb)
        assert cs.semantic_weight is not None
        assert cs.semantic_weight.shape == ()

    def test_semantic_weight_in_zero_one(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        for _ in range(10):
            node_emb = torch.randn(7, 32)
            cs = head(node_emb)
            assert cs.semantic_weight >= 0.0
            assert cs.semantic_weight <= 1.0

    def test_semantic_weight_gradient_flow(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        node_emb = torch.randn(5, 32, requires_grad=True)
        cs = head(node_emb)
        cs.semantic_weight.backward()
        assert node_emb.grad is not None
        assert node_emb.grad.abs().sum() > 0

    def test_semantic_weight_head_has_parameters(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        assert hasattr(head, 'semantic_weight_head')
        param_names = [n for n, _ in head.named_parameters()]
        assert any('semantic_weight_head' in n for n in param_names)

    def test_no_llm_gate_head(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        assert not hasattr(head, 'llm_gate_head')

    def test_semantic_weight_varies_with_input(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        cs1 = head(torch.randn(5, 32))
        cs2 = head(torch.randn(5, 32))
        assert not torch.allclose(cs1.semantic_weight, cs2.semantic_weight)

    def test_semantic_weight_finite_for_various_sizes(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        for n in [3, 10, 20, 50, 100]:
            cs = head(torch.randn(n, 32))
            assert torch.isfinite(cs.semantic_weight), f"Non-finite at n={n}"

    def test_existing_outputs_still_correct(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        node_emb = torch.randn(5, 32)
        cs = head(node_emb)
        assert cs.frequency_gate.shape == (8,)
        assert cs.spatial_focus.shape == (5,)
        assert cs.confidence_weights.shape == (5,)
        assert cs.diffusion_time.shape == ()
        assert cs.wave_damping.shape == ()
        assert (cs.frequency_gate >= 0).all() and (cs.frequency_gate <= 1).all()
        assert cs.diffusion_time > 0
        assert cs.wave_damping > 0

    def test_full_gradient_flow_includes_semantic_weight(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        node_emb = torch.randn(5, 32, requires_grad=True)
        cs = head(node_emb)
        loss = (cs.frequency_gate.sum() + cs.spatial_focus.sum() +
                cs.confidence_weights.sum() + cs.diffusion_time +
                cs.wave_damping + cs.semantic_weight)
        loss.backward()
        assert node_emb.grad is not None
        for p in head.semantic_weight_head.parameters():
            assert p.grad is not None

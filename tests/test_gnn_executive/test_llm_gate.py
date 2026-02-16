"""Tests for llm_gate field in ControlSignal and ControlHead."""

import torch
import pytest
from src.gnn_executive.control_head import ControlSignal, ControlHead


class TestLLMGateField:
    def test_control_signal_has_llm_gate(self):
        cs = ControlSignal(
            frequency_gate=torch.rand(8),
            spatial_focus=torch.rand(5),
            confidence_weights=torch.rand(5),
            diffusion_time=torch.tensor(1.0),
            wave_damping=torch.tensor(0.5),
            llm_gate=torch.tensor(0.7),
        )
        assert cs.llm_gate.shape == ()
        assert cs.llm_gate.item() == pytest.approx(0.7)

    def test_llm_gate_defaults_to_none(self):
        cs = ControlSignal(
            frequency_gate=torch.rand(8),
            spatial_focus=torch.rand(5),
            confidence_weights=torch.rand(5),
            diffusion_time=torch.tensor(1.0),
            wave_damping=torch.tensor(0.5),
        )
        assert cs.llm_gate is None

    def test_existing_fields_unchanged(self):
        cs = ControlSignal(
            frequency_gate=torch.ones(8),
            spatial_focus=torch.ones(5) * 0.3,
            confidence_weights=torch.ones(5) * 0.9,
            diffusion_time=torch.tensor(2.0),
            wave_damping=torch.tensor(1.5),
            llm_gate=torch.tensor(0.5),
        )
        assert cs.frequency_gate.shape == (8,)
        assert cs.spatial_focus.shape == (5,)
        assert cs.confidence_weights.shape == (5,)
        assert cs.diffusion_time.item() == pytest.approx(2.0)
        assert cs.wave_damping.item() == pytest.approx(1.5)


class TestLLMGateHead:
    def test_control_head_produces_llm_gate(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        node_emb = torch.randn(5, 32)
        cs = head(node_emb)
        assert cs.llm_gate is not None
        assert cs.llm_gate.shape == ()

    def test_llm_gate_in_zero_one(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        for _ in range(10):
            node_emb = torch.randn(7, 32)
            cs = head(node_emb)
            assert cs.llm_gate >= 0.0
            assert cs.llm_gate <= 1.0

    def test_llm_gate_gradient_flow(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        node_emb = torch.randn(5, 32, requires_grad=True)
        cs = head(node_emb)
        cs.llm_gate.backward()
        assert node_emb.grad is not None
        assert node_emb.grad.abs().sum() > 0

    def test_llm_gate_head_has_parameters(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        assert hasattr(head, 'llm_gate_head')
        param_names = [n for n, _ in head.named_parameters()]
        assert any('llm_gate_head' in n for n in param_names)

    def test_llm_gate_varies_with_input(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        cs1 = head(torch.randn(5, 32))
        cs2 = head(torch.randn(5, 32))
        # Different inputs should (almost certainly) give different gates
        assert not torch.allclose(cs1.llm_gate, cs2.llm_gate)

    def test_llm_gate_finite_for_various_sizes(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        for n in [3, 10, 20, 50, 100]:
            cs = head(torch.randn(n, 32))
            assert torch.isfinite(cs.llm_gate), f"Non-finite llm_gate at n={n}"

    def test_existing_outputs_still_correct(self):
        """Adding llm_gate head does not break existing signal outputs."""
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

    def test_full_gradient_flow_includes_llm_gate(self):
        """All signals including llm_gate produce gradients."""
        head = ControlHead(embedding_dim=32, num_freqs=8)
        node_emb = torch.randn(5, 32, requires_grad=True)
        cs = head(node_emb)
        loss = (cs.frequency_gate.sum() + cs.spatial_focus.sum() +
                cs.confidence_weights.sum() + cs.diffusion_time +
                cs.wave_damping + cs.llm_gate)
        loss.backward()
        assert node_emb.grad is not None
        # llm_gate_head params have gradients
        for p in head.llm_gate_head.parameters():
            assert p.grad is not None

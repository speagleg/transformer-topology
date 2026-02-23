"""Tests for TopoBridge encoder, decoder, and full bridge."""

import torch
import pytest
from src.llm.backend import MockLLMBackend
from src.llm.topo_bridge import TopoBridgeEncoder, TopoBridgeDecoder, TopoBridge


TOPO_DIM = 32
LLM_DIM = 128  # Small for fast testing
NUM_PREFIX = 4


class TestTopoBridgeEncoder:
    def test_output_shapes(self):
        enc = TopoBridgeEncoder(topo_dim=TOPO_DIM, llm_dim=LLM_DIM, num_prefix=NUM_PREFIX)
        nodes = torch.randn(10, TOPO_DIM)
        topo_memory, prefix = enc(nodes)
        assert topo_memory.shape == (10, LLM_DIM)
        assert prefix.shape == (NUM_PREFIX, LLM_DIM)

    def test_different_graph_sizes(self):
        enc = TopoBridgeEncoder(topo_dim=TOPO_DIM, llm_dim=LLM_DIM, num_prefix=NUM_PREFIX)
        for n in [3, 10, 25, 50]:
            nodes = torch.randn(n, TOPO_DIM)
            topo_memory, prefix = enc(nodes)
            assert topo_memory.shape == (n, LLM_DIM)
            assert prefix.shape == (NUM_PREFIX, LLM_DIM)

    def test_gradient_flow(self):
        enc = TopoBridgeEncoder(topo_dim=TOPO_DIM, llm_dim=LLM_DIM, num_prefix=NUM_PREFIX)
        nodes = torch.randn(10, TOPO_DIM, requires_grad=True)
        topo_memory, prefix = enc(nodes)
        loss = topo_memory.sum() + prefix.sum()
        loss.backward()
        assert nodes.grad is not None
        assert nodes.grad.abs().sum() > 0

    def test_prefix_queries_are_learnable(self):
        enc = TopoBridgeEncoder(topo_dim=TOPO_DIM, llm_dim=LLM_DIM, num_prefix=NUM_PREFIX)
        nodes = torch.randn(10, TOPO_DIM)
        _, prefix = enc(nodes)
        prefix.sum().backward()
        assert enc.prefix_queries.grad is not None

    def test_outputs_finite(self):
        enc = TopoBridgeEncoder(topo_dim=TOPO_DIM, llm_dim=LLM_DIM, num_prefix=NUM_PREFIX)
        nodes = torch.randn(20, TOPO_DIM)
        topo_memory, prefix = enc(nodes)
        assert torch.isfinite(topo_memory).all()
        assert torch.isfinite(prefix).all()


class TestTopoBridgeDecoder:
    def test_output_shape(self):
        dec = TopoBridgeDecoder(topo_dim=TOPO_DIM, llm_dim=LLM_DIM)
        topo_memory = torch.randn(10, LLM_DIM)
        llm_hidden = torch.randn(18, LLM_DIM)  # prefix + nodes
        llm_out, semantic_bias, sem_feat, graph_emb = dec(topo_memory, llm_hidden)
        assert llm_out.shape == (10, TOPO_DIM)
        assert semantic_bias.shape == (10, 10)

    def test_different_graph_sizes(self):
        dec = TopoBridgeDecoder(topo_dim=TOPO_DIM, llm_dim=LLM_DIM)
        for n in [3, 10, 25]:
            topo_memory = torch.randn(n, LLM_DIM)
            llm_hidden = torch.randn(n + NUM_PREFIX, LLM_DIM)
            llm_out, semantic_bias, _, _ = dec(topo_memory, llm_hidden)
            assert llm_out.shape == (n, TOPO_DIM)
            assert semantic_bias.shape == (n, n)

    def test_gradient_flow(self):
        dec = TopoBridgeDecoder(topo_dim=TOPO_DIM, llm_dim=LLM_DIM)
        topo_memory = torch.randn(10, LLM_DIM, requires_grad=True)
        llm_hidden = torch.randn(14, LLM_DIM, requires_grad=True)
        llm_out, semantic_bias, _, _ = dec(topo_memory, llm_hidden)
        (llm_out.sum() + semantic_bias.sum()).backward()
        assert topo_memory.grad is not None
        assert llm_hidden.grad is not None


class TestMockLLMBackend:
    def test_output_shape(self):
        backend = MockLLMBackend(llm_dim=LLM_DIM, hidden_dim=64)
        prefix = torch.randn(NUM_PREFIX, LLM_DIM)
        topo_mem = torch.randn(10, LLM_DIM)
        out = backend(prefix, topo_mem)
        assert out.shape == (NUM_PREFIX + 10, LLM_DIM)

    def test_gradient_flow(self):
        backend = MockLLMBackend(llm_dim=LLM_DIM, hidden_dim=64)
        prefix = torch.randn(NUM_PREFIX, LLM_DIM, requires_grad=True)
        topo_mem = torch.randn(10, LLM_DIM, requires_grad=True)
        out = backend(prefix, topo_mem)
        out.sum().backward()
        assert prefix.grad is not None
        assert topo_mem.grad is not None

    def test_task_text_ignored(self):
        backend = MockLLMBackend(llm_dim=LLM_DIM, hidden_dim=64)
        prefix = torch.randn(NUM_PREFIX, LLM_DIM)
        topo_mem = torch.randn(10, LLM_DIM)
        out1 = backend(prefix, topo_mem, task_text=None)
        out2 = backend(prefix, topo_mem, task_text="some task text")
        assert torch.allclose(out1, out2)


class TestTopoBridge:
    def _make_bridge(self):
        backend = MockLLMBackend(llm_dim=LLM_DIM, hidden_dim=64)
        return TopoBridge(
            backend=backend,
            topo_dim=TOPO_DIM,
            llm_dim=LLM_DIM,
            num_prefix=NUM_PREFIX,
        )

    def test_end_to_end_shape(self):
        bridge = self._make_bridge()
        nodes = torch.randn(10, TOPO_DIM)
        gate = torch.tensor(0.8)
        llm_out, semantic_bias, _, _ = bridge(nodes, gate)
        assert llm_out.shape == (10, TOPO_DIM)
        assert semantic_bias.shape == (10, 10)

    def test_always_runs_even_low_weight(self):
        """No gate threshold skip — DSM always runs now."""
        bridge = self._make_bridge()
        nodes = torch.randn(10, TOPO_DIM)
        gate = torch.tensor(0.05)  # Previously below threshold
        llm_out, semantic_bias, _, _ = bridge(nodes, gate)
        assert llm_out.shape == (10, TOPO_DIM)
        assert llm_out.abs().sum() > 0
        assert semantic_bias.abs().sum() > 0

    def test_returns_nonzero(self):
        bridge = self._make_bridge()
        nodes = torch.randn(10, TOPO_DIM)
        gate = torch.tensor(0.8)
        llm_out, semantic_bias, _, _ = bridge(nodes, gate)
        assert llm_out.abs().sum() > 0

    def test_gradient_flows_through_bridge(self):
        torch.manual_seed(42)  # Fix seed for reproducible attention init
        bridge = self._make_bridge()
        nodes = torch.randn(10, TOPO_DIM, requires_grad=True)
        llm_out, semantic_bias, _, _ = bridge(nodes, torch.tensor(0.8))
        (llm_out.sum() + semantic_bias.sum()).backward()
        assert nodes.grad is not None
        # Bridge params gradient: at least some encoder/decoder params get gradients
        grads_found = sum(
            1 for p in bridge.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grads_found > 0, "No bridge parameters received gradients"

    def test_different_graph_sizes(self):
        bridge = self._make_bridge()
        for n in [3, 10, 25, 50]:
            nodes = torch.randn(n, TOPO_DIM)
            llm_out, semantic_bias, _, _ = bridge(nodes, torch.tensor(0.5))
            assert llm_out.shape == (n, TOPO_DIM)
            assert semantic_bias.shape == (n, n)

    def test_outputs_finite(self):
        bridge = self._make_bridge()
        nodes = torch.randn(20, TOPO_DIM)
        llm_out, semantic_bias, _, _ = bridge(nodes, torch.tensor(0.9))
        assert torch.isfinite(llm_out).all()
        assert torch.isfinite(semantic_bias).all()

    def test_parameter_count_reasonable(self):
        bridge = self._make_bridge()
        total = sum(p.numel() for p in bridge.parameters())
        # With LLM_DIM=128, should be well under 1M
        assert total < 500_000


class TestTopoBridgeWithSemanticBias:
    def _make_bridge(self):
        backend = MockLLMBackend(llm_dim=LLM_DIM, hidden_dim=64)
        return TopoBridge(
            backend=backend,
            topo_dim=TOPO_DIM,
            llm_dim=LLM_DIM,
            num_prefix=NUM_PREFIX,
        )

    def test_forward_returns_tuple(self):
        bridge = self._make_bridge()
        nodes = torch.randn(10, TOPO_DIM)
        llm_out, semantic_bias, _, _ = bridge(nodes, torch.tensor(0.5))
        assert llm_out.shape == (10, TOPO_DIM)
        assert semantic_bias.shape == (10, 10)

    def test_semantic_bias_variable_sizes(self):
        bridge = self._make_bridge()
        for n in [3, 10, 25]:
            nodes = torch.randn(n, TOPO_DIM)
            _, semantic_bias, _, _ = bridge(nodes, torch.tensor(0.5))
            assert semantic_bias.shape == (n, n)

    def test_semantic_bias_gradient_flow(self):
        bridge = self._make_bridge()
        nodes = torch.randn(10, TOPO_DIM, requires_grad=True)
        _, semantic_bias, _, _ = bridge(nodes, torch.tensor(0.5))
        semantic_bias.sum().backward()
        assert nodes.grad is not None

    def test_no_gate_threshold_skip(self):
        bridge = self._make_bridge()
        nodes = torch.randn(10, TOPO_DIM)
        llm_out, semantic_bias, _, _ = bridge(nodes, torch.tensor(0.01))
        assert llm_out.abs().sum() > 0
        assert semantic_bias.abs().sum() > 0

"""Tests for the Distilled Semantic Model (DSM)."""

import torch
import pytest
from src.llm.dsm import DSMBlock, DSMCrossAttentionBlock, DistilledSemanticModel


class TestDSMBlock:
    def test_output_shape(self):
        block = DSMBlock(hidden_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(12, 64)
        out = block(x)
        assert out.shape == (12, 64)

    def test_gradient_flow(self):
        block = DSMBlock(hidden_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(12, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_output_finite(self):
        block = DSMBlock(hidden_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(20, 64)
        out = block(x)
        assert torch.isfinite(out).all()


class TestDSMCrossAttentionBlock:
    def test_output_shape(self):
        block = DSMCrossAttentionBlock(hidden_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(12, 64)
        memory = torch.randn(10, 64)
        out = block(x, memory)
        assert out.shape == (12, 64)

    def test_gradient_flows_to_memory(self):
        block = DSMCrossAttentionBlock(hidden_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(12, 64, requires_grad=True)
        memory = torch.randn(10, 64, requires_grad=True)
        out = block(x, memory)
        out.sum().backward()
        assert x.grad is not None
        assert memory.grad is not None

    def test_different_memory_sizes(self):
        block = DSMCrossAttentionBlock(hidden_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(12, 64)
        for n in [3, 10, 25, 50]:
            memory = torch.randn(n, 64)
            out = block(x, memory)
            assert out.shape == (12, 64)


class TestDistilledSemanticModel:
    def test_output_shape(self):
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )
        prefix = torch.randn(4, 64)
        topo_memory = torch.randn(10, 64)
        task_tokens = torch.randn(8, 64)
        out = model(prefix, topo_memory, task_tokens)
        assert out.shape == (4 + 8, 64)

    def test_no_task_tokens(self):
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )
        prefix = torch.randn(4, 64)
        topo_memory = torch.randn(10, 64)
        out = model(prefix, topo_memory)
        assert out.shape == (4, 64)

    def test_gradient_flow(self):
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )
        prefix = torch.randn(4, 64, requires_grad=True)
        topo_memory = torch.randn(10, 64, requires_grad=True)
        out = model(prefix, topo_memory)
        out.sum().backward()
        assert prefix.grad is not None
        assert topo_memory.grad is not None
        grads = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads > 0

    def test_cross_attn_at_correct_layer(self):
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=6, cross_attn_layer=2,
        )
        assert hasattr(model.layers[2], 'cross_attn')
        assert not hasattr(model.layers[0], 'cross_attn')
        assert not hasattr(model.layers[1], 'cross_attn')

    def test_parameter_count_250m_config(self):
        model = DistilledSemanticModel(
            hidden_dim=1024, num_heads=16, ff_dim=4096,
            num_layers=16, cross_attn_layer=4,
        )
        total = sum(p.numel() for p in model.parameters())
        assert 150_000_000 < total < 350_000_000, f"Got {total:,} params"

    def test_small_config(self):
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )
        total = sum(p.numel() for p in model.parameters())
        assert total < 1_000_000

    def test_output_finite(self):
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )
        prefix = torch.randn(4, 64)
        topo_memory = torch.randn(10, 64)
        out = model(prefix, topo_memory)
        assert torch.isfinite(out).all()

    def test_variable_graph_sizes(self):
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )
        prefix = torch.randn(4, 64)
        for n in [3, 10, 25, 50]:
            topo_memory = torch.randn(n, 64)
            out = model(prefix, topo_memory)
            assert out.shape == (4, 64)

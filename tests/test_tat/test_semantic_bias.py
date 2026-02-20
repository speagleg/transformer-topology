"""Tests for semantic bias injection in TAT spatial attention."""

import torch
import pytest
from src.tat.spatial_attention import TopologicalSpatialAttention


class TestSemanticBiasInjection:
    def test_accepts_semantic_bias(self):
        attn = TopologicalSpatialAttention(embed_dim=32, num_heads=4)
        x = torch.randn(10, 32)
        adj = torch.ones(10, 10)
        semantic_bias = torch.randn(10, 10)
        out = attn(x, adj, semantic_bias=semantic_bias)
        assert out.shape == (10, 32)

    def test_semantic_bias_none_is_no_op(self):
        attn = TopologicalSpatialAttention(embed_dim=32, num_heads=4)
        attn.eval()  # disable dropout for determinism
        x = torch.randn(10, 32)
        adj = torch.ones(10, 10)
        with torch.no_grad():
            out_none = attn(x, adj, semantic_bias=None)
            out_default = attn(x, adj)
        assert torch.allclose(out_none, out_default)

    def test_semantic_bias_changes_output(self):
        attn = TopologicalSpatialAttention(embed_dim=32, num_heads=4)
        attn.eval()
        x = torch.randn(10, 32)
        adj = torch.ones(10, 10)
        with torch.no_grad():
            out_no_bias = attn(x, adj)
            out_with_bias = attn(x, adj, semantic_bias=torch.randn(10, 10) * 10)
        assert not torch.allclose(out_no_bias, out_with_bias)

    def test_semantic_bias_gradient_flow(self):
        attn = TopologicalSpatialAttention(embed_dim=32, num_heads=4)
        x = torch.randn(10, 32)
        adj = torch.ones(10, 10)
        bias = torch.randn(10, 10, requires_grad=True)
        out = attn(x, adj, semantic_bias=bias)
        out.sum().backward()
        assert bias.grad is not None
        assert bias.grad.abs().sum() > 0

    def test_semantic_bias_with_semantic_weight(self):
        attn = TopologicalSpatialAttention(embed_dim=32, num_heads=4)
        attn.eval()
        x = torch.randn(10, 32)
        adj = torch.ones(10, 10)
        bias = torch.randn(10, 10)
        with torch.no_grad():
            out_low = attn(x, adj, semantic_bias=bias, semantic_weight=torch.tensor(0.01))
            out_high = attn(x, adj, semantic_bias=bias, semantic_weight=torch.tensor(10.0))
        assert not torch.allclose(out_low, out_high)

    def test_semantic_weight_gradient_flow(self):
        attn = TopologicalSpatialAttention(embed_dim=32, num_heads=4)
        x = torch.randn(10, 32)
        adj = torch.ones(10, 10)
        bias = torch.randn(10, 10)
        weight = torch.tensor(0.5, requires_grad=True)
        out = attn(x, adj, semantic_bias=bias, semantic_weight=weight)
        out.sum().backward()
        assert weight.grad is not None

    def test_variable_sizes(self):
        attn = TopologicalSpatialAttention(embed_dim=32, num_heads=4)
        for n in [3, 10, 25]:
            x = torch.randn(n, 32)
            adj = torch.ones(n, n)
            bias = torch.randn(n, n)
            out = attn(x, adj, semantic_bias=bias, semantic_weight=torch.tensor(0.5))
            assert out.shape == (n, 32)

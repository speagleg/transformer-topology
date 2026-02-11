import torch
import pytest
from src.tat.spatial_attention import TopologicalSpatialAttention


class TestTopologicalSpatialAttention:
    def test_output_shape(self):
        attn = TopologicalSpatialAttention(embed_dim=64, num_heads=4)
        x = torch.randn(8, 64)
        adj = torch.ones(8, 8)
        out = attn(x, adjacency=adj)
        assert out.shape == (8, 64)

    def test_adjacency_masking(self):
        attn = TopologicalSpatialAttention(embed_dim=64, num_heads=4)
        x = torch.randn(4, 64)
        adj = torch.zeros(4, 4)
        adj[0, 1] = adj[1, 0] = 1.0
        adj[2, 3] = adj[3, 2] = 1.0
        adj.fill_diagonal_(1.0)
        out = attn(x, adjacency=adj)
        assert out.shape == (4, 64)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        attn = TopologicalSpatialAttention(embed_dim=64, num_heads=4)
        x = torch.randn(8, 64, requires_grad=True)
        adj = torch.ones(8, 8)
        out = attn(x, adjacency=adj)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

import pytest
import torch
from src.reasoning_loop.cross_attention import CrossAttentionBlock


class TestCrossAttentionBlock:

    def test_init(self):
        block = CrossAttentionBlock(embed_dim=32, num_heads=4)
        assert block.embed_dim == 32

    def test_forward_shape(self):
        block = CrossAttentionBlock(embed_dim=32, num_heads=4)
        h_struct = torch.randn(10, 32)
        h_text = torch.randn(10, 32)
        out = block(h_struct, h_text)
        assert out.shape == (10, 32)

    def test_forward_different_n(self):
        """Text can have different N if nodes were filtered."""
        block = CrossAttentionBlock(embed_dim=32, num_heads=4)
        h_struct = torch.randn(10, 32)
        h_text = torch.randn(8, 32)
        out = block(h_struct, h_text)
        assert out.shape == (10, 32)

    def test_residual_connection(self):
        """Output should be close to input when attention weights are small."""
        block = CrossAttentionBlock(embed_dim=32, num_heads=4)
        with torch.no_grad():
            for p in block.cross_attn.parameters():
                p.zero_()
        h_struct = torch.randn(5, 32)
        h_text = torch.randn(5, 32)
        out = block(h_struct, h_text)
        expected = block.norm(h_struct)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_gradients_flow_to_both_inputs(self):
        block = CrossAttentionBlock(embed_dim=32, num_heads=4)
        h_struct = torch.randn(5, 32, requires_grad=True)
        h_text = torch.randn(5, 32, requires_grad=True)
        out = block(h_struct, h_text)
        out.sum().backward()
        assert h_struct.grad is not None
        assert h_text.grad is not None

    def test_gradients_flow_to_params(self):
        block = CrossAttentionBlock(embed_dim=32, num_heads=4)
        h_struct = torch.randn(5, 32)
        h_text = torch.randn(5, 32)
        out = block(h_struct, h_text)
        out.sum().backward()
        has_grad = any(p.grad is not None for p in block.parameters())
        assert has_grad

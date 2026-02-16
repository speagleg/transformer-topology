"""Tests for Llama backend and cross-attention module.

Most tests skip if transformers is not installed.
"""

import torch
import pytest

from src.llm.llama_backend import (
    HAS_TRANSFORMERS,
    HAS_PEFT,
    TopoCrossAttention,
)


class TestTopoCrossAttention:
    def test_output_shape(self):
        cross_attn = TopoCrossAttention(hidden_dim=64, num_heads=4)
        hidden = torch.randn(1, 12, 64)
        topo_mem = torch.randn(1, 8, 64)
        out = cross_attn(hidden, topo_mem)
        assert out.shape == (1, 12, 64)

    def test_gradient_flow(self):
        cross_attn = TopoCrossAttention(hidden_dim=64, num_heads=4)
        hidden = torch.randn(1, 12, 64, requires_grad=True)
        topo_mem = torch.randn(1, 8, 64, requires_grad=True)
        out = cross_attn(hidden, topo_mem)
        out.sum().backward()
        assert hidden.grad is not None
        assert topo_mem.grad is not None

    def test_residual_connection(self):
        """Output should differ from input (cross-attention adds info)."""
        torch.manual_seed(42)
        cross_attn = TopoCrossAttention(hidden_dim=64, num_heads=4)
        hidden = torch.randn(1, 12, 64)
        topo_mem = torch.randn(1, 8, 64)
        out = cross_attn(hidden, topo_mem)
        assert not torch.allclose(hidden, out)

    def test_different_memory_sizes(self):
        cross_attn = TopoCrossAttention(hidden_dim=64, num_heads=4)
        hidden = torch.randn(1, 10, 64)
        for n_mem in [3, 8, 20, 50]:
            topo_mem = torch.randn(1, n_mem, 64)
            out = cross_attn(hidden, topo_mem)
            assert out.shape == (1, 10, 64)

    def test_outputs_finite(self):
        cross_attn = TopoCrossAttention(hidden_dim=64, num_heads=4)
        hidden = torch.randn(1, 10, 64)
        topo_mem = torch.randn(1, 8, 64)
        out = cross_attn(hidden, topo_mem)
        assert torch.isfinite(out).all()


class TestImportGuards:
    def test_has_transformers_is_bool(self):
        assert isinstance(HAS_TRANSFORMERS, bool)

    def test_has_peft_is_bool(self):
        assert isinstance(HAS_PEFT, bool)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestLlamaBackend:
    """These tests only run when transformers is installed."""

    def test_import_llama_backend(self):
        from src.llm.llama_backend import LlamaBackend
        assert LlamaBackend is not None

    def test_backend_requires_transformers(self):
        """If transformers is available, LlamaBackend should be importable."""
        from src.llm.llama_backend import LlamaBackend
        # Just checking the class exists and is importable
        assert hasattr(LlamaBackend, 'forward')

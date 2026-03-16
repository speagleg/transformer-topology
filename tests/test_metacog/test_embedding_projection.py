"""Tests for EmbeddingProjection."""

import torch
from src.metacog.embedding_projection import EmbeddingProjection


def test_output_shape():
    """Projection maps (batch, 2048) -> (batch, 128)."""
    proj = EmbeddingProjection(llm_dim=2048, embed_dim=128)
    x = torch.randn(4, 2048)
    out = proj(x)
    assert out.shape == (4, 128)


def test_single_vector():
    """Projection works on a single unbatched vector."""
    proj = EmbeddingProjection(llm_dim=2048, embed_dim=128)
    x = torch.randn(2048)
    out = proj(x)
    assert out.shape == (128,)


def test_gradients_flow():
    """Gradients propagate back through the projection."""
    proj = EmbeddingProjection(llm_dim=2048, embed_dim=128)
    x = torch.randn(1, 2048, requires_grad=True)
    out = proj(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (1, 2048)
    assert proj.proj.weight.grad is not None

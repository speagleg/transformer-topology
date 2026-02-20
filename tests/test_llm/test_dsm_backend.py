"""Tests for DSMBackend -- wraps DSM to implement BaseLLMBackend interface."""

import torch
import pytest
from src.llm.dsm_backend import DSMBackend


class TestDSMBackend:
    def _make_backend(self, dsm_dim=64):
        return DSMBackend(
            dsm_dim=dsm_dim, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )

    def test_output_shape(self):
        backend = self._make_backend()
        prefix = torch.randn(4, 64)
        topo_mem = torch.randn(10, 64)
        out = backend.forward(prefix, topo_mem)
        assert out.shape[1] == 64
        assert out.shape[0] == 4

    def test_with_task_text(self):
        backend = self._make_backend()
        prefix = torch.randn(4, 64)
        topo_mem = torch.randn(10, 64)
        out = backend.forward(prefix, topo_mem, task_text="some task")
        assert out.shape == (4, 64)

    def test_gradient_flow(self):
        backend = self._make_backend()
        prefix = torch.randn(4, 64, requires_grad=True)
        topo_mem = torch.randn(10, 64, requires_grad=True)
        out = backend.forward(prefix, topo_mem)
        out.sum().backward()
        assert prefix.grad is not None
        assert topo_mem.grad is not None

    def test_all_params_trainable(self):
        backend = self._make_backend()
        params = list(backend.parameters())
        assert len(params) > 0
        for p in params:
            assert p.requires_grad

    def test_variable_graph_sizes(self):
        backend = self._make_backend()
        prefix = torch.randn(4, 64)
        for n in [3, 10, 25, 50]:
            topo_mem = torch.randn(n, 64)
            out = backend.forward(prefix, topo_mem)
            assert out.shape == (4, 64)

    def test_output_finite(self):
        backend = self._make_backend()
        prefix = torch.randn(4, 64)
        topo_mem = torch.randn(10, 64)
        out = backend.forward(prefix, topo_mem)
        assert torch.isfinite(out).all()

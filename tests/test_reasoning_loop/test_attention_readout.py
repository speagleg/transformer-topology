import pytest
import torch
from src.reasoning_loop.attention_readout import AttentionReadout


class TestAttentionReadout:

    def test_init(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        assert readout.embed_dim == 32

    def test_forward_shape(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        context = readout(h_out, query_idx=0, target_idx=3, task_id=5)
        assert context.shape == (32,)

    def test_forward_without_task_id(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        context = readout(h_out, 0, 3, task_id=None)
        assert context.shape == (32,)

    def test_different_tasks_different_queries(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        c1 = readout(h_out, 0, 3, task_id=0)
        c2 = readout(h_out, 0, 3, task_id=1)
        assert not torch.allclose(c1, c2)

    def test_gradients_flow(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32, requires_grad=True)
        context = readout(h_out, 0, 3, task_id=5)
        context.sum().backward()
        assert h_out.grad is not None

    def test_attention_focuses_on_query_target_region(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.zeros(10, 32)
        h_out[2] = torch.ones(32)
        h_out[5] = torch.ones(32) * 2
        context = readout(h_out, 2, 5, task_id=0)
        assert context.abs().sum() > 0

    def test_classifier_input_shape(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        topo_features = torch.randn(4)
        fusion_weight = torch.tensor(0.5)
        combined = readout.build_classifier_input(
            h_out, 0, 3, task_id=5,
            topo_features=topo_features,
            fusion_weight=fusion_weight,
        )
        assert combined.shape == (101,)

    def test_classifier_input_without_optional(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        combined = readout.build_classifier_input(h_out, 0, 3, task_id=5)
        assert combined.shape == (96,)

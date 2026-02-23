"""Tests for QwenGraphBackend (uses mock LLM for CI)."""
import torch
from src.llm.qwen_backend import QwenGraphBackend


class TestQwenGraphBackend:
    def test_mock_mode_forward(self):
        config = {"topo_dim": 32, "llm_dim": 64, "num_tokens": 16,
                  "adapter_layers": 2, "num_tasks": 14, "use_mock": True}
        backend = QwenGraphBackend(config)
        node_embs = torch.randn(10, 32)
        task_id = torch.tensor(0)
        features, bias, graph_emb = backend.forward_graph(node_embs, task_id)
        assert features.shape == (10, 32)
        assert bias.shape == (10, 10)
        assert graph_emb.shape == (32,)

    def test_trainable_params_exclude_llm(self):
        config = {"topo_dim": 32, "llm_dim": 64, "num_tokens": 16,
                  "adapter_layers": 2, "num_tasks": 14, "use_mock": True}
        backend = QwenGraphBackend(config)
        trainable = list(backend.trainable_parameters())
        assert len(trainable) > 0
        for p in trainable:
            assert p.requires_grad

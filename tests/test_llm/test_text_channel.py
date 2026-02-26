"""Test node_texts text channel through QwenGraphBackend and QwenBridgeAdapter."""
import torch


def test_forward_graph_accepts_node_texts():
    from src.llm.qwen_backend import QwenGraphBackend
    backend = QwenGraphBackend({
        "topo_dim": 8, "llm_dim": 32, "num_tokens": 4,
        "adapter_layers": 1, "num_tasks": 19, "use_mock": True,
    })
    node_emb = torch.randn(5, 8)
    task_id = torch.tensor(0)
    feat, bias, graph_emb = backend.forward_graph(
        node_emb, task_id, node_texts=["dog", "cat", "bird", "fish", "tree"]
    )
    assert feat.shape == (5, 8)
    assert bias.shape == (5, 5)


def test_forward_graph_without_node_texts():
    """Backward compat: node_texts=None still works."""
    from src.llm.qwen_backend import QwenGraphBackend
    backend = QwenGraphBackend({
        "topo_dim": 8, "llm_dim": 32, "num_tokens": 4,
        "adapter_layers": 1, "num_tasks": 19, "use_mock": True,
    })
    node_emb = torch.randn(5, 8)
    task_id = torch.tensor(0)
    feat, bias, graph_emb = backend.forward_graph(node_emb, task_id)
    assert feat.shape == (5, 8)


def test_qwen_bridge_adapter_passes_node_texts():
    from src.llm.qwen_bridge_adapter import QwenBridgeAdapter
    adapter = QwenBridgeAdapter(
        {"llm_dim": 32, "num_tokens": 4, "adapter_layers": 1,
         "num_tasks": 19, "use_mock": True},
        topo_dim=8,
    )
    node_emb = torch.randn(5, 8)
    out = adapter(node_emb, 0.5, task_text="kg_relation",
                  node_texts=["a", "b", "c", "d", "e"])
    assert len(out) == 4
    assert out[0].shape == (5, 8)


def test_topo_bridge_accepts_node_texts():
    """TopoBridge (DSM path) accepts node_texts kwarg without error."""
    from src.llm.backend import MockLLMBackend
    from src.llm.topo_bridge import TopoBridge
    backend = MockLLMBackend(llm_dim=32)
    bridge = TopoBridge(backend, topo_dim=8, llm_dim=32, num_prefix=4)
    node_emb = torch.randn(5, 8)
    out = bridge(node_emb, torch.tensor(0.5), task_text="test",
                 node_texts=["a", "b", "c", "d", "e"])
    assert len(out) == 4
    assert out[0].shape == (5, 8)

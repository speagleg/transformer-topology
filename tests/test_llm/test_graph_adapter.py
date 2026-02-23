"""Tests for GraphFormer adapter."""
import torch
from src.llm.graph_adapter import GraphFormerEncoder, GraphFormerDecoder


class TestGraphFormerEncoder:
    def test_output_shape(self):
        enc = GraphFormerEncoder(topo_dim=32, llm_dim=64, num_tokens=16, num_layers=2)
        node_embs = torch.randn(10, 32)
        task_id = torch.tensor(0)
        tokens = enc(node_embs, task_id)
        assert tokens.shape == (16, 64)

    def test_different_node_counts(self):
        enc = GraphFormerEncoder(topo_dim=32, llm_dim=64, num_tokens=16, num_layers=2)
        for n in [5, 10, 30]:
            tokens = enc(torch.randn(n, 32), torch.tensor(0))
            assert tokens.shape == (16, 64)


class TestGraphFormerDecoder:
    def test_output_shapes(self):
        dec = GraphFormerDecoder(llm_dim=64, topo_dim=32, num_layers=2)
        node_queries = torch.randn(10, 64)
        llm_hidden = torch.randn(16, 64)
        features, bias, graph_emb = dec(node_queries, llm_hidden)
        assert features.shape == (10, 32)
        assert bias.shape == (10, 10)
        assert graph_emb.shape == (32,)

    def test_gradient_flows(self):
        dec = GraphFormerDecoder(llm_dim=64, topo_dim=32, num_layers=2)
        node_queries = torch.randn(10, 64, requires_grad=True)
        llm_hidden = torch.randn(16, 64)
        features, bias, graph_emb = dec(node_queries, llm_hidden)
        loss = features.sum() + bias.sum() + graph_emb.sum()
        loss.backward()
        assert node_queries.grad is not None

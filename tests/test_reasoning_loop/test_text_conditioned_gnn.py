import pytest
import torch
from src.reasoning_loop.text_conditioned_gnn import TextConditionedGNN


class TestTextConditionedGNN:

    def test_init(self):
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        assert gnn.embed_dim == 32

    def test_forward_shape(self):
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        text_embs = torch.randn(10, 32)
        out = gnn(x, edge_index, text_embs)
        assert out.shape == (10, 32)

    def test_forward_without_text(self):
        """Should work without text (no modulation)."""
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        out = gnn(x, edge_index, text_embs=None)
        assert out.shape == (10, 32)

    def test_text_modulation_changes_output(self):
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        text_embs = torch.randn(10, 32)
        out_with = gnn(x, edge_index, text_embs)
        out_without = gnn(x, edge_index, text_embs=None)
        assert not torch.allclose(out_with, out_without)

    def test_gradients_flow_through_text(self):
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        text_embs = torch.randn(10, 32, requires_grad=True)
        out = gnn(x, edge_index, text_embs)
        out.sum().backward()
        assert text_embs.grad is not None

    def test_gradients_flow_through_x(self):
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(10, 32, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        text_embs = torch.randn(10, 32)
        out = gnn(x, edge_index, text_embs)
        out.sum().backward()
        assert x.grad is not None

    def test_single_node_graph(self):
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(1, 32)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        text_embs = torch.randn(1, 32)
        out = gnn(x, edge_index, text_embs)
        assert out.shape == (1, 32)

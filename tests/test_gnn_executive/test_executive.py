import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.executive import GNNExecutive


def make_chain(dim=16, length=5):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "r")
    return cc


class TestGNNExecutive:
    def test_output_shape(self):
        executive = GNNExecutive(
            embedding_dim=16, hidden_dim=32, num_spatial_layers=2,
            num_spectral_layers=2, max_freqs=5
        )
        cc = make_chain(dim=16, length=5)
        node_out, edge_out = executive(cc)
        assert node_out.shape == (5, 16)

    def test_dual_path(self):
        executive = GNNExecutive(
            embedding_dim=16, hidden_dim=32, num_spatial_layers=2,
            num_spectral_layers=2, max_freqs=5
        )
        cc = make_chain(dim=16, length=5)
        node_out, _ = executive(cc)
        assert not torch.isnan(node_out).any()
        input_emb = cc.get_embeddings(0)
        assert not torch.allclose(node_out, input_emb, atol=0.1)

    def test_gradient_flow(self):
        executive = GNNExecutive(
            embedding_dim=16, hidden_dim=32, num_spatial_layers=2,
            num_spectral_layers=2, max_freqs=5
        )
        cc = make_chain(dim=16, length=5)
        node_out, _ = executive(cc)
        loss = node_out.sum()
        loss.backward()
        for param in executive.parameters():
            if param.requires_grad:
                assert param.grad is not None

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.spatial import SpatialMessagePassingLayer, SpatialGNN


def make_chain(dim=16, length=4):
    cc = CellComplex(embedding_dim=dim)
    nodes = [cc.add_0_cell(torch.randn(dim), "concept") for _ in range(length)]
    for i in range(length - 1):
        cc.add_1_cell(nodes[i], nodes[i + 1], torch.randn(dim), "r")
    return cc


class TestSpatialMessagePassingLayer:
    def test_output_shape(self):
        layer = SpatialMessagePassingLayer(in_dim=16, out_dim=16)
        cc = make_chain(dim=16, length=4)
        x = cc.get_embeddings(0)
        edge_index = cc.edge_index()
        out = layer(x, edge_index)
        assert out.shape == (4, 16)

    def test_gradient_flow(self):
        layer = SpatialMessagePassingLayer(in_dim=16, out_dim=16)
        cc = make_chain(dim=16, length=4)
        x = cc.get_embeddings(0).requires_grad_(True)
        edge_index = cc.edge_index()
        out = layer(x, edge_index)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 16)


class TestSpatialGNN:
    def test_multi_layer(self):
        gnn = SpatialGNN(in_dim=16, hidden_dim=16, out_dim=16, num_layers=3)
        cc = make_chain(dim=16, length=5)
        x = cc.get_embeddings(0)
        edge_index = cc.edge_index()
        out = gnn(x, edge_index)
        assert out.shape == (5, 16)

    def test_information_propagation(self):
        gnn = SpatialGNN(in_dim=16, hidden_dim=16, out_dim=16, num_layers=4)
        cc = make_chain(dim=16, length=5)
        x = cc.get_embeddings(0)
        edge_index = cc.edge_index()

        x_perturbed = x.clone()
        x_perturbed[0] += 10.0

        out_original = gnn(x, edge_index)
        out_perturbed = gnn(x_perturbed, edge_index)

        diff = (out_original[-1] - out_perturbed[-1]).abs().sum()
        assert diff > 0.01, "Signal should propagate through entire chain"

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.gnn_executive.executive import GNNExecutive


def make_cc_with_face(dim=16):
    cc = CellComplex(embedding_dim=dim)
    n0 = cc.add_0_cell(torch.randn(dim), "node")
    n1 = cc.add_0_cell(torch.randn(dim), "node")
    n2 = cc.add_0_cell(torch.randn(dim), "node")
    n3 = cc.add_0_cell(torch.randn(dim), "node")
    e0 = cc.add_1_cell(n0, n1, torch.randn(dim), "edge")
    e1 = cc.add_1_cell(n1, n2, torch.randn(dim), "edge")
    e2 = cc.add_1_cell(n2, n0, torch.randn(dim), "edge")
    cc.add_1_cell(n2, n3, torch.randn(dim), "edge")
    cc.add_2_cell([e0, e1, e2], torch.randn(dim))
    return cc


def make_cc_no_face(dim=16):
    cc = CellComplex(embedding_dim=dim)
    n0 = cc.add_0_cell(torch.randn(dim), "node")
    n1 = cc.add_0_cell(torch.randn(dim), "node")
    n2 = cc.add_0_cell(torch.randn(dim), "node")
    cc.add_1_cell(n0, n1, torch.randn(dim), "edge")
    cc.add_1_cell(n1, n2, torch.randn(dim), "edge")
    return cc


class TestHigherOrderGNN:
    def test_output_shape_with_face(self):
        cc = make_cc_with_face()
        gnn = GNNExecutive(embedding_dim=16, hidden_dim=32, num_spatial_layers=1,
                           num_spectral_layers=1, max_freqs=4, use_higher_order=True)
        node_out, edge_out = gnn(cc)
        assert node_out.shape == (4, 16)
        assert edge_out is not None
        assert edge_out.shape == (4, 16)

    def test_output_shape_without_face(self):
        cc = make_cc_no_face()
        gnn = GNNExecutive(embedding_dim=16, hidden_dim=32, num_spatial_layers=1,
                           num_spectral_layers=1, max_freqs=4, use_higher_order=True)
        node_out, edge_out = gnn(cc)
        assert node_out.shape == (3, 16)
        assert edge_out is not None
        assert edge_out.shape == (2, 16)

    def test_gradient_flow_through_hierarchy(self):
        cc = make_cc_with_face()
        gnn = GNNExecutive(embedding_dim=16, hidden_dim=32, num_spatial_layers=1,
                           num_spectral_layers=1, max_freqs=4, use_higher_order=True)
        node_out, edge_out = gnn(cc)
        loss = node_out.sum() + edge_out.sum()
        loss.backward()
        # Check gradients flow through higher-order components
        ho_grad_count = sum(
            1 for n, p in gnn.named_parameters()
            if 'higher_order' in n and p.grad is not None and p.grad.abs().sum() > 0
        )
        assert ho_grad_count > 0

    def test_backward_compatible_no_higher_order(self):
        cc = make_cc_no_face()
        gnn = GNNExecutive(embedding_dim=16, hidden_dim=32, num_spatial_layers=1,
                           num_spectral_layers=1, max_freqs=4, use_higher_order=False)
        node_out, edge_out = gnn(cc)
        assert node_out.shape == (3, 16)
        assert edge_out is None

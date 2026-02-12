import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.spectral.positional_encoding import (
    cell_membership_encoding,
    TopologicalPositionalEncoding,
)


def make_triangle_with_face(dim=16):
    cc = CellComplex(embedding_dim=dim)
    n0 = cc.add_0_cell(torch.randn(dim), "node")
    n1 = cc.add_0_cell(torch.randn(dim), "node")
    n2 = cc.add_0_cell(torch.randn(dim), "node")
    # Add an extra node not in the triangle
    n3 = cc.add_0_cell(torch.randn(dim), "node")
    e0 = cc.add_1_cell(n0, n1, torch.randn(dim), "edge")
    e1 = cc.add_1_cell(n1, n2, torch.randn(dim), "edge")
    e2 = cc.add_1_cell(n2, n0, torch.randn(dim), "edge")
    cc.add_1_cell(n2, n3, torch.randn(dim), "edge")
    cc.add_2_cell([e0, e1, e2], torch.randn(dim))
    return cc


class TestCellMembership:
    def test_correctness(self):
        cc = make_triangle_with_face()
        membership = cell_membership_encoding(cc)
        assert membership.shape == (4, 1)
        # Nodes 0, 1, 2 are in the face; node 3 is not
        assert membership[0, 0] == 1.0
        assert membership[1, 0] == 1.0
        assert membership[2, 0] == 1.0
        assert membership[3, 0] == 0.0


class TestTopologicalPE:
    def test_output_shape(self):
        cc = make_triangle_with_face()
        pe = TopologicalPositionalEncoding(embedding_dim=16, num_eigenvectors=4,
                                           num_persistence_features=4, max_2_cells=8)
        out = pe(cc)
        assert out.shape == (4, 16)

    def test_gradient_flow(self):
        cc = make_triangle_with_face()
        pe = TopologicalPositionalEncoding(embedding_dim=16, num_eigenvectors=4,
                                           num_persistence_features=4, max_2_cells=8)
        out = pe(cc)
        loss = out.sum()
        loss.backward()
        grad_count = sum(1 for p in pe.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0

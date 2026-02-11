import torch
import pytest
from src.cell_complex.cell_complex import CellComplex


class TestCellComplexConstruction:
    def test_empty_complex(self):
        cc = CellComplex(embedding_dim=64)
        assert cc.num_cells(0) == 0
        assert cc.num_cells(1) == 0
        assert cc.num_cells(2) == 0

    def test_add_0_cells(self):
        cc = CellComplex(embedding_dim=4)
        idx = cc.add_0_cell(embedding=torch.randn(4), cell_type="concept")
        assert idx == 0
        assert cc.num_cells(0) == 1

    def test_add_1_cell(self):
        cc = CellComplex(embedding_dim=4)
        a = cc.add_0_cell(torch.randn(4), "concept")
        b = cc.add_0_cell(torch.randn(4), "concept")
        e = cc.add_1_cell(source=a, target=b, embedding=torch.randn(4), relation_type="causes")
        assert e == 0
        assert cc.num_cells(1) == 1

    def test_boundary_operator_1(self):
        """d_1 maps 1-cells to their boundary 0-cells."""
        cc = CellComplex(embedding_dim=4)
        a = cc.add_0_cell(torch.randn(4), "concept")
        b = cc.add_0_cell(torch.randn(4), "concept")
        c = cc.add_0_cell(torch.randn(4), "concept")
        cc.add_1_cell(a, b, torch.randn(4), "r1")
        cc.add_1_cell(b, c, torch.randn(4), "r2")

        B1 = cc.boundary_operator(1)
        assert B1.shape == (3, 2)
        assert B1[a, 0].item() == -1.0
        assert B1[b, 0].item() == 1.0
        assert B1[b, 1].item() == -1.0
        assert B1[c, 1].item() == 1.0

    def test_get_embeddings(self):
        cc = CellComplex(embedding_dim=4)
        emb = torch.tensor([1.0, 2.0, 3.0, 4.0])
        cc.add_0_cell(emb, "concept")
        retrieved = cc.get_embeddings(0)
        assert torch.allclose(retrieved[0], emb)

    def test_adjacency_0_cells(self):
        cc = CellComplex(embedding_dim=4)
        a = cc.add_0_cell(torch.randn(4), "concept")
        b = cc.add_0_cell(torch.randn(4), "concept")
        c = cc.add_0_cell(torch.randn(4), "concept")
        cc.add_1_cell(a, b, torch.randn(4), "r1")
        cc.add_1_cell(b, c, torch.randn(4), "r2")

        A = cc.adjacency_matrix(0)
        assert A.shape == (3, 3)
        assert A[a, b].item() == 1.0
        assert A[b, a].item() == 1.0
        assert A[a, c].item() == 0.0

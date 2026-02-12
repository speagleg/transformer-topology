import torch
import pytest
from src.cell_complex.cell_complex import CellComplex


def make_triangle(embedding_dim: int = 8) -> tuple[CellComplex, list[int], list[int]]:
    """Create a triangle (3 nodes, 3 edges) and return (cc, node_ids, edge_ids)."""
    cc = CellComplex(embedding_dim=embedding_dim)
    n0 = cc.add_0_cell(torch.randn(embedding_dim), "node")
    n1 = cc.add_0_cell(torch.randn(embedding_dim), "node")
    n2 = cc.add_0_cell(torch.randn(embedding_dim), "node")
    e0 = cc.add_1_cell(n0, n1, torch.randn(embedding_dim), "edge")
    e1 = cc.add_1_cell(n1, n2, torch.randn(embedding_dim), "edge")
    e2 = cc.add_1_cell(n2, n0, torch.randn(embedding_dim), "edge")
    return cc, [n0, n1, n2], [e0, e1, e2]


class TestAdd2Cell:
    def test_add_2_cell_triangle(self):
        cc, nodes, edges = make_triangle()
        face = cc.add_2_cell(edges, torch.randn(8), "triangle")
        assert face == 0
        assert cc.num_cells(2) == 1

    def test_invalid_boundary_not_cycle(self):
        cc = CellComplex(embedding_dim=8)
        n0 = cc.add_0_cell(torch.randn(8), "node")
        n1 = cc.add_0_cell(torch.randn(8), "node")
        n2 = cc.add_0_cell(torch.randn(8), "node")
        e0 = cc.add_1_cell(n0, n1, torch.randn(8), "edge")
        e1 = cc.add_1_cell(n1, n2, torch.randn(8), "edge")
        # e0, e1 form a path, not a cycle
        with pytest.raises(AssertionError):
            cc.add_2_cell([e0, e1], torch.randn(8))


class TestBoundaryOperator2:
    def test_b2_shape(self):
        cc, nodes, edges = make_triangle()
        cc.add_2_cell(edges, torch.randn(8))
        B2 = cc.boundary_operator(2)
        assert B2.shape == (3, 1)  # 3 edges, 1 face

    def test_b2_values(self):
        cc, nodes, edges = make_triangle()
        cc.add_2_cell(edges, torch.randn(8))
        B2 = cc.boundary_operator(2)
        # All boundary edges should have nonzero entries (+1 or -1)
        assert (B2.abs() > 0).sum().item() == 3

    def test_chain_complex_property(self):
        """B1 @ B2 must equal zero."""
        cc, nodes, edges = make_triangle()
        cc.add_2_cell(edges, torch.randn(8))
        assert cc.verify_chain_complex()


class TestEdgeAdjacency:
    def test_edge_adjacency(self):
        cc, nodes, edges = make_triangle()
        A1 = cc.adjacency_matrix(1)
        assert A1.shape == (3, 3)
        # In a triangle, all edges share endpoints with all other edges
        assert A1.sum().item() == 6.0  # 3 pairs * 2 (symmetric)


class TestTwoCellEmbeddings:
    def test_get_set_embeddings_dim2(self):
        cc, nodes, edges = make_triangle()
        cc.add_2_cell(edges, torch.randn(8))
        embs = cc.get_embeddings(2)
        assert embs.shape == (1, 8)
        new_embs = torch.ones(1, 8)
        cc.set_embeddings(2, new_embs)
        assert torch.allclose(cc.get_embeddings(2), new_embs)

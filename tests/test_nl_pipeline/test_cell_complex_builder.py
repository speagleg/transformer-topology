"""Tests for CellComplexBuilder: GraphSpec -> CellComplex."""
import pytest
from src.nl_pipeline.data_types import NodeSpec, EdgeSpec, GraphSpec
from src.nl_pipeline.cell_complex_builder import CellComplexBuilder


@pytest.fixture
def builder():
    return CellComplexBuilder(embedding_dim=32)


def _tech_spec():
    return GraphSpec(
        nodes=[NodeSpec("server", "component"), NodeSpec("database", "component"), NodeSpec("cache", "component")],
        edges=[EdgeSpec("server", "cache", "connects"), EdgeSpec("cache", "database", "causes")],
        query_node="server", target_node="database", domain="technology",
    )


class TestBuild:
    def test_correct_node_count(self, builder):
        cc, node_map = builder.build(_tech_spec())
        assert cc.num_cells(0) == 3

    def test_correct_edge_count(self, builder):
        cc, node_map = builder.build(_tech_spec())
        assert cc.num_cells(1) == 2

    def test_node_map_keys(self, builder):
        cc, node_map = builder.build(_tech_spec())
        assert set(node_map.keys()) == {"server", "database", "cache"}

    def test_node_map_indices_valid(self, builder):
        cc, node_map = builder.build(_tech_spec())
        for idx in node_map.values():
            assert 0 <= idx < cc.num_cells(0)

    def test_source_cell_type(self, builder):
        spec = _tech_spec()
        cc, node_map = builder.build(spec)
        src_idx = node_map[spec.query_node]
        assert cc._0_cell_types[src_idx] == "source"

    def test_target_cell_type(self, builder):
        spec = _tech_spec()
        cc, node_map = builder.build(spec)
        tgt_idx = node_map[spec.target_node]
        assert cc._0_cell_types[tgt_idx] == "target"

    def test_embeddings_have_correct_dim(self, builder):
        cc, _ = builder.build(_tech_spec())
        embs = cc.get_embeddings(0)
        assert embs.shape == (3, 32)

    def test_structural_embeddings_populated(self, builder):
        cc, _ = builder.build(_tech_spec())
        embs = cc.get_embeddings(0)
        assert embs[0, 0].item() > 0  # normalized degree > 0


class TestEdgeRelationEncoding:
    def test_causal_edge_encoded(self, builder):
        spec = _tech_spec()
        cc, _ = builder.build(spec)
        edge_embs = cc.get_embeddings(1)
        assert any(edge_embs[i, 1].item() != 0 for i in range(cc.num_cells(1)))


class TestTriangleFilling:
    def test_triangle_creates_2cells(self, builder):
        spec = GraphSpec(
            nodes=[NodeSpec("a", "n"), NodeSpec("b", "n"), NodeSpec("c", "n")],
            edges=[EdgeSpec("a", "b", "connects"), EdgeSpec("b", "c", "connects"), EdgeSpec("a", "c", "connects")],
            query_node="a", target_node="c", domain="test",
        )
        cc, _ = builder.build(spec)
        assert cc.num_cells(2) >= 1


class TestSingleNode:
    def test_single_node_no_crash(self, builder):
        spec = GraphSpec(
            nodes=[NodeSpec("lonely", "entity")], edges=[],
            query_node="lonely", target_node=None, domain="test",
        )
        cc, node_map = builder.build(spec)
        assert cc.num_cells(0) == 1
        assert cc.num_cells(1) == 0

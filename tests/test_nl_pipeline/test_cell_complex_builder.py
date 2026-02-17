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


class TestTopologyAwareBuilding:
    def test_ba_topology_generates_larger_graph(self, builder):
        spec = GraphSpec(
            nodes=[NodeSpec("hub", "entity"), NodeSpec("client_a", "entity"),
                   NodeSpec("client_b", "entity")],
            edges=[EdgeSpec("hub", "client_a", "connects")],
            query_node="hub", target_node="client_a", domain="technology",
            topology_hint="ba", topology_confidence=0.8,
            node_roles={"hub": "hub", "client_a": "leaf", "client_b": "leaf"},
        )
        cc, name_to_cc = builder.build(spec)
        # BA with min_size=8 should produce >= 8 nodes
        assert cc.num_cells(0) >= 8
        # All named nodes mapped
        assert "hub" in name_to_cc
        assert "client_a" in name_to_cc
        assert "client_b" in name_to_cc

    def test_tree_topology_generates_tree(self, builder):
        spec = GraphSpec(
            nodes=[NodeSpec("root", "entity"), NodeSpec("leaf", "entity")],
            edges=[], query_node="root", target_node="leaf", domain="general",
            topology_hint="tree", topology_confidence=0.7,
            node_roles={"root": "root", "leaf": "leaf"},
        )
        cc, name_to_cc = builder.build(spec)
        assert cc.num_cells(0) >= 8
        assert "root" in name_to_cc
        assert "leaf" in name_to_cc

    def test_fallback_no_topology(self, builder):
        """Without topology hint, builds from edges (original behavior)."""
        spec = GraphSpec(
            nodes=[NodeSpec("a", "entity"), NodeSpec("b", "entity")],
            edges=[EdgeSpec("a", "b", "connects")],
            query_node="a", target_node="b", domain="general",
        )
        cc, name_to_cc = builder.build(spec)
        assert cc.num_cells(0) == 2  # Only parsed nodes

    def test_low_confidence_uses_fallback(self, builder):
        """Low confidence should fall back to edge-based construction."""
        spec = GraphSpec(
            nodes=[NodeSpec("x", "entity"), NodeSpec("y", "entity")],
            edges=[EdgeSpec("x", "y", "connects")],
            query_node="x", target_node="y", domain="general",
            topology_hint="ba", topology_confidence=0.2,  # below threshold
        )
        cc, _ = builder.build(spec)
        assert cc.num_cells(0) == 2  # fallback

    def test_sbm_produces_connected_graph(self, builder):
        spec = GraphSpec(
            nodes=[NodeSpec("team_a", "entity"), NodeSpec("team_b", "entity")],
            edges=[], query_node="team_a", target_node="team_b", domain="general",
            topology_hint="sbm", topology_confidence=0.9,
            node_roles={"team_a": "member", "team_b": "member"},
        )
        cc, name_to_cc = builder.build(spec)
        assert cc.num_cells(0) >= 9  # min SBM size
        assert cc.num_cells(1) > 0  # has edges

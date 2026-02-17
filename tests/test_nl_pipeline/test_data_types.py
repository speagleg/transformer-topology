"""Tests for Phase 5 NL pipeline data structures."""
from src.nl_pipeline.data_types import (
    NodeSpec, EdgeSpec, GraphSpec, TaskRoute, NLResult, TopologyHint,
)

class TestNodeSpec:
    def test_creation(self):
        n = NodeSpec(name="server", type="component")
        assert n.name == "server"
        assert n.type == "component"

class TestEdgeSpec:
    def test_creation(self):
        e = EdgeSpec(source="server", target="database", relation="causes")
        assert e.source == "server"
        assert e.relation == "causes"

class TestGraphSpec:
    def test_creation_with_target(self):
        spec = GraphSpec(
            nodes=[NodeSpec("a", "node"), NodeSpec("b", "node")],
            edges=[EdgeSpec("a", "b", "connects")],
            query_node="a", target_node="b", domain="technology",
        )
        assert len(spec.nodes) == 2
        assert len(spec.edges) == 1
        assert spec.query_node == "a"
        assert spec.target_node == "b"
        assert spec.domain == "technology"

    def test_creation_without_target(self):
        spec = GraphSpec(
            nodes=[NodeSpec("x", "entity")], edges=[],
            query_node="x", target_node=None, domain="general",
        )
        assert spec.target_node is None

    def test_node_names(self):
        spec = GraphSpec(
            nodes=[NodeSpec("a", "n"), NodeSpec("b", "n"), NodeSpec("c", "n")],
            edges=[EdgeSpec("a", "b", "r")],
            query_node="a", target_node="b", domain="test",
        )
        assert {n.name for n in spec.nodes} == {"a", "b", "c"}

class TestTaskRoute:
    def test_creation(self):
        route = TaskRoute(
            task_type="bfs", query_node_idx=0, target_node_idx=3,
            max_classes=16, metadata={"task_prompt": "test"},
        )
        assert route.task_type == "bfs"
        assert route.max_classes == 16

class TestTopologyHint:
    def test_creation(self):
        hint = TopologyHint(topology="ba", confidence=0.85, node_roles={"hub": "hub"})
        assert hint.topology == "ba"
        assert hint.confidence == 0.85
        assert hint.node_roles == {"hub": "hub"}
        assert hint.properties == {}

    def test_none_topology(self):
        hint = TopologyHint(topology=None, confidence=0.0)
        assert hint.topology is None
        assert hint.node_roles == {}

    def test_with_properties(self):
        hint = TopologyHint(
            topology="sbm", confidence=0.7,
            node_roles={"a": "member"},
            properties={"num_communities": 3},
        )
        assert hint.properties["num_communities"] == 3


class TestGraphSpecTopology:
    def test_backward_compat(self):
        """Existing GraphSpec creation works without new fields."""
        spec = GraphSpec(
            nodes=[NodeSpec(name="a", type="entity")],
            edges=[], query_node="a", target_node=None, domain="general",
        )
        assert spec.topology_hint is None
        assert spec.topology_confidence == 0.0
        assert spec.node_roles is None

    def test_with_topology(self):
        spec = GraphSpec(
            nodes=[NodeSpec(name="hub", type="entity")],
            edges=[], query_node="hub", target_node=None, domain="technology",
            topology_hint="ba", topology_confidence=0.85,
            node_roles={"hub": "hub"},
        )
        assert spec.topology_hint == "ba"
        assert spec.topology_confidence == 0.85
        assert spec.node_roles == {"hub": "hub"}


class TestNLResult:
    def test_creation(self):
        spec = GraphSpec(nodes=[NodeSpec("a", "n")], edges=[],
                         query_node="a", target_node=None, domain="test")
        result = NLResult(answer="The answer is 3.", class_idx=3,
                          task_type="bfs", graph_spec=spec, confidence=0.92)
        assert result.answer == "The answer is 3."
        assert result.confidence == 0.92

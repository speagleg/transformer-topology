"""Tests for Phase 5 NL pipeline data structures."""
from src.nl_pipeline.data_types import (
    NodeSpec, EdgeSpec, GraphSpec, TaskRoute, NLResult,
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

class TestNLResult:
    def test_creation(self):
        spec = GraphSpec(nodes=[NodeSpec("a", "n")], edges=[],
                         query_node="a", target_node=None, domain="test")
        result = NLResult(answer="The answer is 3.", class_idx=3,
                          task_type="bfs", graph_spec=spec, confidence=0.92)
        assert result.answer == "The answer is 3."
        assert result.confidence == 0.92

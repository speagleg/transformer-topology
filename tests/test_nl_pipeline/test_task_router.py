"""Tests for TaskRouter heuristic routing."""
import pytest
from src.nl_pipeline.data_types import NodeSpec, EdgeSpec, GraphSpec
from src.nl_pipeline.task_router import TaskRouter


@pytest.fixture
def router():
    return TaskRouter()


def _make_spec(edges=None, domain="test"):
    nodes = [NodeSpec("a", "node"), NodeSpec("b", "node"), NodeSpec("c", "node")]
    return GraphSpec(nodes=nodes, edges=edges or [EdgeSpec("a", "b", "connects")],
                     query_node="a", target_node="b", domain=domain)


class TestCausalRouting:
    def test_causes_relation(self, router):
        spec = _make_spec([EdgeSpec("a", "b", "causes")])
        route = router.route(spec, "Why does A cause B?")
        assert route.task_type == "labeled_reasoning"

    def test_prevents_relation(self, router):
        spec = _make_spec([EdgeSpec("a", "b", "prevents")])
        route = router.route(spec, "Does A prevent B?")
        assert route.task_type == "labeled_reasoning"

    def test_enables_relation(self, router):
        spec = _make_spec([EdgeSpec("a", "b", "enables")])
        route = router.route(spec, "How does A enable B?")
        assert route.task_type == "labeled_reasoning"


class TestKeywordRouting:
    def test_shortest_path(self, router):
        spec = _make_spec()
        route = router.route(spec, "What is the shortest path from A to B?")
        assert route.task_type == "bfs"

    def test_distance(self, router):
        spec = _make_spec()
        route = router.route(spec, "How far is A from B?")
        assert route.task_type == "bfs"

    def test_count_paths(self, router):
        spec = _make_spec()
        route = router.route(spec, "How many paths are there from A to B?")
        assert route.task_type == "path_counting"

    def test_cycle(self, router):
        spec = _make_spec()
        route = router.route(spec, "Is there a cycle in this graph?")
        assert route.task_type == "cycle_detection"

    def test_loop(self, router):
        spec = _make_spec()
        route = router.route(spec, "Does this network have any loops?")
        assert route.task_type == "cycle_detection"

    def test_missing_edge(self, router):
        spec = _make_spec()
        route = router.route(spec, "What connects A to B?")
        assert route.task_type == "graph_completion"

    def test_similar(self, router):
        spec = _make_spec()
        route = router.route(spec, "Is the biology domain similar to the technology domain?")
        assert route.task_type == "analogical_transfer"

    def test_connected(self, router):
        spec = _make_spec()
        route = router.route(spec, "How well connected is this graph?")
        assert route.task_type == "spectral_gap"


class TestDefaultRouting:
    def test_unknown_query_defaults_to_diverse(self, router):
        spec = _make_spec()
        route = router.route(spec, "Tell me about this graph.")
        assert route.task_type == "diverse"


class TestTopologyAffinity:
    def test_tree_topology_routes_to_bfs(self, router):
        spec = GraphSpec(
            nodes=[NodeSpec("a", "node"), NodeSpec("b", "node")],
            edges=[EdgeSpec("a", "b", "connects")],
            query_node="a", target_node="b", domain="test",
            topology_hint="tree", topology_confidence=0.8,
        )
        route = router.route(spec, "what is the structure of this hierarchy")
        assert route.task_type == "bfs"

    def test_ba_topology_routes_to_spectral(self, router):
        spec = GraphSpec(
            nodes=[NodeSpec("hub", "node"), NodeSpec("spoke", "node")],
            edges=[EdgeSpec("hub", "spoke", "connects")],
            query_node="hub", target_node="spoke", domain="test",
            topology_hint="ba", topology_confidence=0.7,
        )
        route = router.route(spec, "tell me about this network")
        assert route.task_type == "spectral_gap"

    def test_topology_hint_in_metadata(self, router):
        spec = GraphSpec(
            nodes=[NodeSpec("a", "node"), NodeSpec("b", "node")],
            edges=[EdgeSpec("a", "b", "connects")],
            query_node="a", target_node="b", domain="test",
            topology_hint="ba", topology_confidence=0.8,
        )
        route = router.route(spec, "tell me about this network")
        assert route.metadata.get("topology_hint") == "ba"

    def test_low_confidence_falls_through(self, router):
        spec = GraphSpec(
            nodes=[NodeSpec("a", "node"), NodeSpec("b", "node")],
            edges=[EdgeSpec("a", "b", "connects")],
            query_node="a", target_node="b", domain="test",
            topology_hint="tree", topology_confidence=0.1,
        )
        route = router.route(spec, "tell me about this")
        assert route.task_type == "diverse"

    def test_keyword_takes_priority_over_topology(self, router):
        spec = GraphSpec(
            nodes=[NodeSpec("a", "node"), NodeSpec("b", "node")],
            edges=[EdgeSpec("a", "b", "connects")],
            query_node="a", target_node="b", domain="test",
            topology_hint="tree", topology_confidence=0.9,
        )
        # "cycle" keyword should override tree→bfs affinity
        route = router.route(spec, "is there a cycle in this tree")
        assert route.task_type == "cycle_detection"


class TestRouteMetadata:
    def test_route_has_correct_max_classes(self, router):
        spec = _make_spec()
        route = router.route(spec, "What is the shortest path?")
        assert route.max_classes == 16

    def test_route_metadata_has_task_prompt(self, router):
        spec = _make_spec()
        route = router.route(spec, "What is the shortest path?")
        assert "task_prompt" in route.metadata

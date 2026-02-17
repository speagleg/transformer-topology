"""End-to-end integration tests: NL queries with implied topologies."""
import pytest
from src.nl_pipeline.graph_parser import MockGraphParser
from src.nl_pipeline.cell_complex_builder import CellComplexBuilder
from src.nl_pipeline.task_router import TaskRouter
from src.nl_pipeline.answer_generator import AnswerGenerator
from src.nl_pipeline.data_types import TaskRoute


@pytest.fixture
def parser():
    return MockGraphParser()


@pytest.fixture
def builder():
    return CellComplexBuilder(embedding_dim=32)


@pytest.fixture
def router():
    return TaskRouter()


@pytest.fixture
def answerer():
    return AnswerGenerator()


class TestHubSpokeQueries:
    def test_central_hub_failure(self, parser, builder, router):
        spec = parser.parse(
            "what happens when the central hub fails in a distributed system"
        )
        assert spec.topology_hint == "ba"
        cc, names = builder.build(spec)
        assert cc.num_cells(0) >= 8  # topology generation
        route = router.route(spec, "what happens when the central hub fails")
        assert route.metadata.get("topology_hint") == "ba"

    def test_load_balancer_query(self, parser, builder):
        spec = parser.parse(
            "how does the load balancer distribute traffic to servers"
        )
        assert spec.topology_hint == "ba"
        cc, _ = builder.build(spec)
        assert cc.num_cells(0) >= 8


class TestHierarchyQueries:
    def test_org_chart(self, parser, builder):
        spec = parser.parse(
            "how does information flow from manager to subordinate in the org chart"
        )
        assert spec.topology_hint == "tree"
        cc, _ = builder.build(spec)
        assert cc.num_cells(0) >= 8

    def test_decision_tree(self, parser):
        spec = parser.parse(
            "what is the depth of this decision tree from root to leaf"
        )
        assert spec.topology_hint == "tree"


class TestCommunityQueries:
    def test_department_clusters(self, parser, builder):
        spec = parser.parse(
            "how do different departments communicate in the company"
        )
        assert spec.topology_hint == "sbm"
        cc, _ = builder.build(spec)
        assert cc.num_cells(0) >= 9

    def test_community_detection(self, parser):
        spec = parser.parse(
            "identify the main communities in this social group"
        )
        assert spec.topology_hint == "sbm"


class TestCyclicQueries:
    def test_feedback_loop(self, parser, builder):
        spec = parser.parse(
            "is there a feedback loop between production and testing"
        )
        assert spec.topology_hint == "ws"
        cc, _ = builder.build(spec)
        assert cc.num_cells(0) >= 8

    def test_circular_dependency(self, parser):
        spec = parser.parse(
            "detect the circular dependency in the module imports"
        )
        assert spec.topology_hint == "ws"


class TestFallbackBehavior:
    def test_generic_query_no_topology(self, parser, builder):
        spec = parser.parse(
            "tell me about the relationship between apples and oranges"
        )
        assert spec.topology_hint is None
        cc, _ = builder.build(spec)
        # Falls back to edge-based construction (small graph)
        assert cc.num_cells(0) <= 6

    def test_topology_propagates_to_route_metadata(self, parser, router):
        spec = parser.parse(
            "how tight-knit is each clique in the organization"
        )
        assert spec.topology_hint == "caveman"
        route = router.route(spec, "how tight-knit is each clique")
        assert route.metadata.get("topology_hint") == "caveman"


class TestEndToEndAnswerFlow:
    def test_hub_query_produces_topology_answer(self, parser, builder, router, answerer):
        query = "what happens when the central hub fails in a distributed system"
        spec = parser.parse(query)
        cc, names = builder.build(spec)
        route = router.route(spec, query)
        answer = answerer.generate(0, route, spec, query)
        # Answer should mention hub-spoke since topology=ba
        assert "hub-spoke" in answer.lower()

    def test_tree_query_produces_topology_answer(self, parser, builder, router, answerer):
        query = "how deep is the hierarchy from root to leaf"
        spec = parser.parse(query)
        cc, names = builder.build(spec)
        route = router.route(spec, query)
        answer = answerer.generate(3, route, spec, query)
        assert "hierarchical tree" in answer.lower()

    def test_generic_query_no_topology_prefix(self, parser, builder, router, answerer):
        query = "tell me about foo and bar"
        spec = parser.parse(query)
        cc, names = builder.build(spec)
        route = router.route(spec, query)
        answer = answerer.generate(5, route, spec, query)
        # No topology prefix for generic queries
        assert "in the" not in answer.lower().split(":")[0] if ":" in answer else True

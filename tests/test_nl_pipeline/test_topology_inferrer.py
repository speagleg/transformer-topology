"""Tests for TopologyInferrer pattern-matching topology inference."""
import pytest
from src.nl_pipeline.data_types import NodeSpec, EdgeSpec, GraphSpec
from src.nl_pipeline.topology_inferrer import TopologyInferrer


@pytest.fixture
def inferrer():
    return TopologyInferrer()


def _make_spec(nodes, edges=None, domain="general"):
    ns = [NodeSpec(name=n, type="entity") for n in nodes]
    es = edges or []
    return GraphSpec(
        nodes=ns,
        edges=es,
        query_node=nodes[0],
        target_node=nodes[-1] if len(nodes) > 1 else None,
        domain=domain,
    )


# ------------------------------------------------------------------
# 8 topology pattern tests
# ------------------------------------------------------------------
class TestTopologyPatterns:
    def test_ba_hub(self, inferrer):
        spec = _make_spec(["server", "client1", "client2"])
        hint = inferrer.infer(spec, "what happens when the central hub fails")
        assert hint.topology == "ba"

    def test_tree_hierarchy(self, inferrer):
        spec = _make_spec(["ceo", "vp", "engineer"])
        hint = inferrer.infer(spec, "how does information flow through the hierarchy")
        assert hint.topology == "tree"

    def test_sbm_community(self, inferrer):
        spec = _make_spec(["alice", "bob", "carol"])
        hint = inferrer.infer(spec, "how do clusters communicate between departments")
        assert hint.topology == "sbm"

    def test_ws_cycle(self, inferrer):
        spec = _make_spec(["a", "b", "c"])
        hint = inferrer.infer(spec, "is there a feedback loop in this cycle")
        assert hint.topology == "ws"

    def test_grid(self, inferrer):
        spec = _make_spec(["cell1", "cell2", "cell3"])
        hint = inferrer.infer(spec, "how do values propagate through the grid")
        assert hint.topology == "grid"

    def test_ladder_parallel(self, inferrer):
        spec = _make_spec(["a", "b", "c"])
        hint = inferrer.infer(spec, "which parallel path is the redundant backup")
        assert hint.topology == "ladder"

    def test_caveman_clique(self, inferrer):
        spec = _make_spec(["a", "b", "c"])
        hint = inferrer.infer(spec, "how tight-knit is each clique in the network")
        assert hint.topology == "caveman"

    def test_er_random(self, inferrer):
        spec = _make_spec(["a", "b", "c"])
        hint = inferrer.infer(spec, "what is the structure of this random network")
        assert hint.topology == "er"


# ------------------------------------------------------------------
# Confidence tests
# ------------------------------------------------------------------
class TestConfidence:
    def test_multi_pattern_high_confidence(self, inferrer):
        spec = _make_spec(["hub", "spoke1", "spoke2"])
        hint = inferrer.infer(
            spec,
            "the central hub in this star network broadcasts to every spoke",
        )
        assert hint.confidence > 0.5

    def test_no_match_zero_confidence(self, inferrer):
        spec = _make_spec(["alpha", "beta"])
        hint = inferrer.infer(spec, "the quick brown fox jumps over the lazy dog")
        assert hint.topology is None
        assert hint.confidence == 0.0


# ------------------------------------------------------------------
# Node role tests
# ------------------------------------------------------------------
class TestNodeRoles:
    def test_hub_node_role(self, inferrer):
        spec = _make_spec(["hub", "device1", "device2"])
        hint = inferrer.infer(spec, "the central hub connects all devices")
        assert hint.node_roles["hub"] == "hub"

    def test_root_node_role(self, inferrer):
        spec = _make_spec(["root", "child1", "child2"])
        hint = inferrer.infer(spec, "the tree has a root and two children")
        assert hint.node_roles["root"] == "root"

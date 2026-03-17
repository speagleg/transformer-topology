"""Tests for src/metacog/graph_templates.py"""

import pytest

from src.metacog.graph_templates import create_template
from src.metacog.problem_classifier import PROBLEM_TYPES


class TestAllProblemTypesExist:
    def test_all_problem_types_exist(self):
        """create_template should accept every declared problem type."""
        for ptype in PROBLEM_TYPES:
            cc = create_template(ptype, embedding_dim=16, num_steps=4)
            assert cc is not None, f"create_template returned None for {ptype}"
            assert cc.num_cells(0) > 0, f"No nodes in template for {ptype}"


class TestSequentialTemplate:
    def test_sequential_template_default_num_steps(self):
        """Default num_steps=6 → 6 nodes, 5 edges."""
        cc = create_template("SEQUENTIAL", embedding_dim=32)
        assert cc.num_cells(0) == 6
        assert cc.num_cells(1) == 5

    def test_sequential_template_five_nodes(self):
        """num_steps=5 → 5 nodes, 4 edges (linear chain)."""
        cc = create_template("SEQUENTIAL", embedding_dim=32, num_steps=5)
        assert cc.num_cells(0) == 5
        assert cc.num_cells(1) == 4

    def test_sequential_template_chain_structure(self):
        """Edges form a path: node i → node i+1."""
        cc = create_template("SEQUENTIAL", embedding_dim=8, num_steps=4)
        sources = cc._1_cell_sources
        targets = cc._1_cell_targets
        for i, (s, t) in enumerate(zip(sources, targets)):
            assert s == i
            assert t == i + 1


class TestExplorationTemplate:
    def test_exploration_template_seven_nodes(self):
        """num_steps=6 → depth=3 binary tree → 7 nodes, 6 edges."""
        cc = create_template("EXPLORATION", embedding_dim=16, num_steps=6)
        assert cc.num_cells(0) == 7
        assert cc.num_cells(1) == 6

    def test_exploration_template_tree_structure(self):
        """Every non-root node has exactly one parent edge."""
        cc = create_template("EXPLORATION", embedding_dim=16, num_steps=6)
        # Each non-root node (1..n-1) should appear exactly once as a target
        n = cc.num_cells(0)
        targets = cc._1_cell_targets
        for node_idx in range(1, n):
            assert targets.count(node_idx) == 1, (
                f"Node {node_idx} should have exactly one parent edge"
            )
        # Root (node 0) should never appear as a target
        assert 0 not in targets


class TestUnknownTypeDefaultsToSequential:
    def test_unknown_type_defaults_to_sequential(self):
        """Unknown problem type should produce the same structure as SEQUENTIAL."""
        cc_unknown = create_template("NONSENSE_TYPE", embedding_dim=16, num_steps=5)
        cc_seq = create_template("SEQUENTIAL", embedding_dim=16, num_steps=5)
        assert cc_unknown.num_cells(0) == cc_seq.num_cells(0)
        assert cc_unknown.num_cells(1) == cc_seq.num_cells(1)


class TestConstraintTemplate:
    def test_constraint_star_structure(self):
        """Star: 1 center + num_steps leaves, num_steps edges."""
        n = 4
        cc = create_template("CONSTRAINT", embedding_dim=16, num_steps=n)
        assert cc.num_cells(0) == n + 1
        assert cc.num_cells(1) == n
        # All edges should originate from node 0 (the goal/center)
        for s in cc._1_cell_sources:
            assert s == 0


class TestMultiHopTemplate:
    def test_multi_hop_structure(self):
        """Two evidence streams (depth nodes each) + 1 conclusion, proper edge count."""
        depth = 4
        cc = create_template("MULTI_HOP", embedding_dim=16, num_steps=depth)
        expected_nodes = depth + depth + 1  # stream_a + stream_b + conclusion
        # (depth-1) chain edges per stream + 2 merge edges
        expected_edges = (depth - 1) * 2 + 2
        assert cc.num_cells(0) == expected_nodes
        assert cc.num_cells(1) == expected_edges


class TestVerificationTemplate:
    def test_verification_bipartite_structure(self):
        """num_steps claims + num_steps evidence nodes, num_steps edges."""
        n = 5
        cc = create_template("VERIFICATION", embedding_dim=16, num_steps=n)
        assert cc.num_cells(0) == 2 * n
        assert cc.num_cells(1) == n
        # Claims are nodes 0..n-1, evidence are nodes n..2n-1
        for i, (s, t) in enumerate(zip(cc._1_cell_sources, cc._1_cell_targets)):
            assert s == i          # claim index
            assert t == n + i      # corresponding evidence index

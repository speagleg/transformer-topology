# tests/test_nl_pipeline/test_answer_generator.py
"""Tests for AnswerGenerator: class_idx + context → NL answer."""

import pytest

from src.nl_pipeline.data_types import NodeSpec, EdgeSpec, GraphSpec, TaskRoute
from src.nl_pipeline.answer_generator import AnswerGenerator, CLASS_INTERPRETATIONS


def _make_spec():
    return GraphSpec(
        nodes=[NodeSpec("server", "component"), NodeSpec("database", "component")],
        edges=[EdgeSpec("server", "database", "causes")],
        query_node="server",
        target_node="database",
        domain="technology",
    )


def _make_route(task_type, max_classes):
    return TaskRoute(
        task_type=task_type,
        query_node_idx=0,
        target_node_idx=1,
        max_classes=max_classes,
        metadata={"task_prompt": "test"},
    )


class TestClassInterpretations:
    def test_bfs_interpretation_exists(self):
        assert "bfs" in CLASS_INTERPRETATIONS

    def test_labeled_reasoning_interpretation_exists(self):
        assert "labeled_reasoning" in CLASS_INTERPRETATIONS

    def test_cycle_detection_interpretation_exists(self):
        assert "cycle_detection" in CLASS_INTERPRETATIONS

    def test_graph_completion_interpretation_exists(self):
        assert "graph_completion" in CLASS_INTERPRETATIONS


class TestAnswerGenerator:
    @pytest.fixture
    def gen(self):
        return AnswerGenerator()

    def test_bfs_answer(self, gen):
        answer = gen.generate(
            class_idx=3,
            task_route=_make_route("bfs", 16),
            graph_spec=_make_spec(),
            original_query="How far is server from database?",
        )
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert "3" in answer

    def test_labeled_reasoning_causal(self, gen):
        answer = gen.generate(
            class_idx=0,
            task_route=_make_route("labeled_reasoning", 3),
            graph_spec=_make_spec(),
            original_query="Does server cause database issues?",
        )
        assert "causal" in answer.lower()

    def test_cycle_detection_yes(self, gen):
        answer = gen.generate(
            class_idx=1,
            task_route=_make_route("cycle_detection", 2),
            graph_spec=_make_spec(),
            original_query="Is there a cycle?",
        )
        assert "cycle" in answer.lower()

    def test_graph_completion_no_edge(self, gen):
        answer = gen.generate(
            class_idx=0,
            task_route=_make_route("graph_completion", 2),
            graph_spec=_make_spec(),
            original_query="Should there be an edge?",
        )
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_diverse_fallback(self, gen):
        answer = gen.generate(
            class_idx=5,
            task_route=_make_route("diverse", 11),
            graph_spec=_make_spec(),
            original_query="Tell me about this graph.",
        )
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_answer_includes_node_names(self, gen):
        answer = gen.generate(
            class_idx=3,
            task_route=_make_route("bfs", 16),
            graph_spec=_make_spec(),
            original_query="How far is server from database?",
        )
        assert "server" in answer.lower() or "database" in answer.lower()

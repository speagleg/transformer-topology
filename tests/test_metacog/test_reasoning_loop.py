"""Tests for MetacognitiveReasoner (Task 6 — full reasoning loop)."""

import pytest
from src.metacog.reasoning_loop import MetacognitiveReasoner


def test_reasoner_init():
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    assert reasoner is not None


def test_mock_reasoning():
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    result = reasoner.solve("What is 2 + 3?")
    assert "answer" in result
    assert "steps" in result
    assert "health_reports" in result
    assert len(result["steps"]) > 0


def test_mock_reasoning_has_topology():
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    result = reasoner.solve("What is 2 + 3?")
    assert len(result["health_reports"]) > 0
    for report in result["health_reports"]:
        assert hasattr(report, "action")
        assert report.action in ("CONTINUE", "INTERVENE", "BACKTRACK", "STOP")


def test_mock_produces_answer():
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    result = reasoner.solve("What is 2 + 3?")
    assert result["answer"] != ""


def test_mock_problem_type():
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    result = reasoner.solve("What is 2 + 3?")
    assert result["problem_type"] == "SEQUENTIAL"  # default without LLM


def test_result_has_all_keys():
    """solve() returns all expected keys."""
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    result = reasoner.solve("Explain how sorting works.")
    expected_keys = {"answer", "steps", "health_reports", "interventions", "graph", "problem_type"}
    assert expected_keys.issubset(result.keys())


def test_result_graph_is_cell_complex():
    """The returned graph is a CellComplex."""
    from src.cell_complex.cell_complex import CellComplex
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    result = reasoner.solve("What is 2 + 3?")
    assert isinstance(result["graph"], CellComplex)


def test_result_graph_has_nodes():
    """Returned graph has at least one node (step)."""
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    result = reasoner.solve("What is 2 + 3?")
    assert result["graph"].num_cells(0) > 0


def test_interventions_is_list():
    """Interventions field is a list."""
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    result = reasoner.solve("What is 2 + 3?")
    assert isinstance(result["interventions"], list)


def test_steps_are_dicts():
    """Each step is a dict with at least a 'step' key."""
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    result = reasoner.solve("What is 2 + 3?")
    for step in result["steps"]:
        assert isinstance(step, dict)
        assert "step" in step


def test_max_steps_respected():
    """Reasoner never exceeds max_steps steps."""
    reasoner = MetacognitiveReasoner(embedding_dim=128, max_steps=3, use_mock=True)
    result = reasoner.solve("What is 2 + 3?")
    assert len(result["steps"]) <= 3


def test_health_reports_count_matches_steps():
    """One health report per step added."""
    reasoner = MetacognitiveReasoner(embedding_dim=128, use_mock=True)
    result = reasoner.solve("What is 2 + 3?")
    assert len(result["health_reports"]) == len(result["steps"])

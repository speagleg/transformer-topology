# tests/test_nl_pipeline/test_pipeline.py
"""Tests for NLReasoningPipeline end-to-end."""

import tempfile

import pytest
import torch

from src.nl_pipeline.data_types import NLResult, GraphSpec
from src.nl_pipeline.pipeline import NLReasoningPipeline, load_frozen_model
from src.benchmarks.run_benchmark_suite import _build_model


@pytest.fixture
def small_config():
    """Minimal model config for testing."""
    return {
        "model": {
            "embedding_dim": 32,
            "gnn_hidden": 64,
            "gnn_spatial_layers": 2,
            "gnn_spectral_layers": 2,
            "max_freqs": 8,
            "tat_layers": 2,
            "tat_spatial_heads": 2,
            "tat_spectral_heads": 2,
            "tat_ff_dim": 128,
            "max_iterations": 3,
            "convergence_threshold": 0.1,
            "use_wave_dynamics": True,
            "use_higher_order": True,
            "use_topological_pe": False,
            "use_structural_features": True,
        },
        "wave": {
            "filter_type": "heat",
            "wave_mode": "spectral",
            "laplacian_dim": 0,
            "wave_strength_gate": False,
            "use_neural_ode": False,
        },
        "llm": {
            "backend": "mock",
            "llm_dim": 128,
            "num_prefix": 4,
            "gate_threshold": 0.1,
            "mock_hidden": 64,
        },
    }


@pytest.fixture
def checkpoint_path(small_config):
    """Create a temporary checkpoint from a fresh model."""
    model = _build_model(
        "hierarchical_llm",
        small_config["model"],
        max_classes=16,
        device=torch.device("cpu"),
        wave_config=small_config["wave"],
        llm_config=small_config["llm"],
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        return f.name


@pytest.fixture
def pipeline(checkpoint_path, small_config):
    return NLReasoningPipeline(
        checkpoint_path=checkpoint_path,
        model_config=small_config["model"],
        wave_config=small_config["wave"],
        llm_config=small_config["llm"],
        max_classes=16,
        device="cpu",
    )


class TestLoadFrozenModel:
    def test_all_params_frozen(self, checkpoint_path, small_config):
        model = load_frozen_model(
            checkpoint_path,
            small_config["model"],
            small_config["wave"],
            small_config["llm"],
            max_classes=16,
            device=torch.device("cpu"),
        )
        for param in model.parameters():
            assert not param.requires_grad

    def test_model_in_eval_mode(self, checkpoint_path, small_config):
        model = load_frozen_model(
            checkpoint_path,
            small_config["model"],
            small_config["wave"],
            small_config["llm"],
            max_classes=16,
            device=torch.device("cpu"),
        )
        assert not model.training


class TestPipeline:
    def test_query_returns_nl_result(self, pipeline):
        result = pipeline.query("What is the shortest path from server to database?")
        assert isinstance(result, NLResult)

    def test_result_has_answer_string(self, pipeline):
        result = pipeline.query("How far is A from B?")
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_result_has_valid_class_idx(self, pipeline):
        result = pipeline.query("Is there a cycle in this network?")
        assert isinstance(result.class_idx, int)
        assert 0 <= result.class_idx < 16

    def test_result_has_graph_spec(self, pipeline):
        result = pipeline.query("Why does rain cause flooding?")
        assert isinstance(result.graph_spec, GraphSpec)
        assert len(result.graph_spec.nodes) >= 2

    def test_result_has_confidence(self, pipeline):
        result = pipeline.query("Any question here")
        assert 0.0 <= result.confidence <= 1.0

    def test_result_has_task_type(self, pipeline):
        result = pipeline.query("What is the shortest path from A to B?")
        assert result.task_type == "bfs"

    def test_causal_query_routes_correctly(self, pipeline):
        result = pipeline.query("Why does the server crash when the database fails?")
        # MockGraphParser detects "cause" → edges with "causes" → labeled_reasoning
        assert result.task_type in ("labeled_reasoning", "diverse")

"""Tests for prompt builder."""

import torch
import pytest

from src.llm.prompt_builder import build_topo_prompt
from src.gnn_executive.control_head import ControlSignal
from src.cell_complex.cell_complex import CellComplex


def _make_cc(n_nodes=10, embedding_dim=16):
    """Create a simple chain cell complex for testing."""
    cc = CellComplex(embedding_dim)
    for i in range(n_nodes):
        cc.add_0_cell(torch.randn(embedding_dim), cell_type="node")
    for i in range(n_nodes - 1):
        cc.add_1_cell(i, i + 1, torch.randn(embedding_dim), relation_type="edge")
    return cc


def _make_control_signal(n_nodes=10):
    return ControlSignal(
        frequency_gate=torch.rand(8),
        spatial_focus=torch.rand(n_nodes),
        confidence_weights=torch.ones(n_nodes) * 0.7,
        diffusion_time=torch.tensor(1.0),
        wave_damping=torch.tensor(0.5),
        semantic_weight=torch.tensor(0.8),
    )


class TestBuildTopoPrompt:
    def test_basic_prompt(self):
        cc = _make_cc(10)
        prompt = build_topo_prompt(cc, task_type="bfs")
        assert "[TOPO]" in prompt
        assert "[/TOPO]" in prompt
        assert "nodes=10" in prompt
        assert "task=bfs" in prompt

    def test_with_control_signal(self):
        cc = _make_cc(10)
        ctrl = _make_control_signal(10)
        prompt = build_topo_prompt(cc, control_signal=ctrl, task_type="hodge_class")
        assert "conf=" in prompt
        assert "gate=" in prompt

    def test_with_metadata_prompt(self):
        cc = _make_cc(10)
        metadata = {'task_prompt': 'custom prompt text'}
        prompt = build_topo_prompt(cc, metadata=metadata)
        assert "custom prompt text" in prompt

    def test_iteration_info(self):
        cc = _make_cc(10)
        prompt = build_topo_prompt(cc, iteration=3, max_iterations=5)
        assert "iter=3/5" in prompt

    def test_with_faces(self):
        # Build a triangle: 3 nodes, 3 edges forming a cycle
        cc = CellComplex(16)
        cc.add_0_cell(torch.randn(16), "node")
        cc.add_0_cell(torch.randn(16), "node")
        cc.add_0_cell(torch.randn(16), "node")
        e0 = cc.add_1_cell(0, 1, torch.randn(16), "edge")
        e1 = cc.add_1_cell(1, 2, torch.randn(16), "edge")
        e2 = cc.add_1_cell(2, 0, torch.randn(16), "edge")
        cc.add_2_cell([e0, e1, e2], torch.randn(16))
        prompt = build_topo_prompt(cc)
        assert "faces=1" in prompt

    def test_returns_string(self):
        cc = _make_cc(10)
        prompt = build_topo_prompt(cc)
        assert isinstance(prompt, str)

    def test_no_control_signal(self):
        cc = _make_cc(10)
        prompt = build_topo_prompt(cc)
        assert "conf=" not in prompt

    def test_semantic_weight_none(self):
        cc = _make_cc(10)
        ctrl = ControlSignal(
            frequency_gate=torch.rand(8),
            spatial_focus=torch.rand(10),
            confidence_weights=torch.ones(10) * 0.5,
            diffusion_time=torch.tensor(1.0),
            wave_damping=torch.tensor(0.5),
        )
        prompt = build_topo_prompt(cc, control_signal=ctrl)
        assert "gate=" not in prompt

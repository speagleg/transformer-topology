"""Tests for MetaCognitive Controller ControlSignal extensions."""
import torch
from src.gnn_executive.control_head import ControlSignal


def test_control_signal_has_metacog_fields():
    """ControlSignal should accept new metacog fields with None defaults."""
    cs = ControlSignal(
        frequency_gate=torch.randn(8),
        spatial_focus=torch.randn(20),
        confidence_weights=torch.randn(20),
        diffusion_time=torch.tensor(1.0),
        wave_damping=torch.tensor(0.5),
    )
    assert cs.text_gate is None
    assert cs.structure_gate is None
    assert cs.uncertainty is None
    assert cs.iteration_budget is None
    assert cs.strategy_weights is None


def test_control_signal_metacog_fields_populated():
    """ControlSignal should store metacog fields when provided."""
    cs = ControlSignal(
        frequency_gate=torch.randn(8),
        spatial_focus=torch.randn(20),
        confidence_weights=torch.randn(20),
        diffusion_time=torch.tensor(1.0),
        wave_damping=torch.tensor(0.5),
        text_gate=torch.tensor(0.8),
        structure_gate=torch.tensor(0.3),
        uncertainty=torch.tensor(0.2),
        iteration_budget=torch.tensor(3.0),
        strategy_weights=torch.softmax(torch.randn(4), dim=0),
    )
    assert abs(cs.text_gate.item() - 0.8) < 1e-5
    assert abs(cs.structure_gate.item() - 0.3) < 1e-5
    assert abs(cs.uncertainty.item() - 0.2) < 1e-5
    assert abs(cs.iteration_budget.item() - 3.0) < 1e-5
    assert cs.strategy_weights.shape == (4,)
    assert abs(cs.strategy_weights.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# MetaCognitiveController tests
# ---------------------------------------------------------------------------
import pytest
from src.gnn_executive.metacognitive_controller import MetaCognitiveController


class TestMetaCognitiveController:

    @pytest.fixture
    def controller(self):
        return MetaCognitiveController(
            embedding_dim=32, num_freqs=8, num_filters=4,
            num_tasks=19, task_embed_dim=128,
            use_topo_feedback=True, use_embedding_topo_feedback=True,
        )

    def test_output_is_control_signal(self, controller):
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert isinstance(cs, ControlSignal)

    def test_existing_fields_present(self, controller):
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert cs.frequency_gate.shape == (8,)
        assert cs.spatial_focus.shape == (20,)
        assert cs.confidence_weights.shape == (20,)
        assert cs.diffusion_time.dim() == 0
        assert cs.wave_damping.dim() == 0
        assert cs.semantic_weight.dim() == 0
        assert cs.filter_weights.shape == (4,)

    def test_metacog_fields_present(self, controller):
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert cs.text_gate is not None
        assert cs.structure_gate is not None
        assert cs.uncertainty is not None
        assert cs.iteration_budget is not None
        assert cs.strategy_weights is not None

    def test_text_gate_range(self, controller):
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert 0.0 <= cs.text_gate.item() <= 1.0
        assert 0.0 <= cs.structure_gate.item() <= 1.0

    def test_uncertainty_range(self, controller):
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert 0.0 <= cs.uncertainty.item() <= 1.0

    def test_strategy_weights_sum_to_one(self, controller):
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert cs.strategy_weights.shape == (4,)
        assert abs(cs.strategy_weights.sum().item() - 1.0) < 1e-5

    def test_iteration_budget_positive(self, controller):
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert cs.iteration_budget.item() > 0

    def test_task_id_changes_output(self, controller):
        node_embs = torch.randn(20, 32)
        cs0 = controller(node_embs, task_id=0)
        cs5 = controller(node_embs, task_id=5)
        assert not torch.allclose(cs0.text_gate, cs5.text_gate) or \
               not torch.allclose(cs0.strategy_weights, cs5.strategy_weights)

    def test_iteration_context(self, controller):
        node_embs = torch.randn(20, 32)
        iter_ctx = torch.tensor([0.4, 0.7, 0.01])
        cs = controller(node_embs, task_id=0, iteration_context=iter_ctx)
        assert isinstance(cs, ControlSignal)

    def test_topo_features_passthrough(self, controller):
        node_embs = torch.randn(20, 32)
        topo = torch.randn(6)
        cs = controller(node_embs, task_id=0, topo_features=topo)
        assert isinstance(cs, ControlSignal)

    def test_gradient_flows_through_task_embedding(self, controller):
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=3)
        loss = cs.text_gate + cs.uncertainty
        loss.backward()
        assert controller.task_embedding.weight.grad is not None
        assert controller.task_embedding.weight.grad[3].abs().sum() > 0

    def test_confidence_temperature_learnable(self, controller):
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        cs.uncertainty.backward()
        assert controller.confidence_temperature.grad is not None

    def test_no_task_id_uses_zeros(self, controller):
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=None)
        assert isinstance(cs, ControlSignal)

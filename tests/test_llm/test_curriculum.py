"""Tests for curriculum training, gate penalty, and LLM diagnostics."""

import torch
import pytest

from src.benchmarks.benchmark_dataset import BenchmarkDataset, get_max_classes
from src.benchmarks.diagnostics import DiagnosticCollector
from src.benchmarks.run_benchmark_suite import _build_model, TIER_3_TASKS
from src.benchmarks.run_comparison import HierarchicalMultiHopModel, evaluate
from scripts.run_phase4c_curriculum import train_epoch_with_gate_penalty


# Small model config for fast tests
MC = {
    'embedding_dim': 16, 'gnn_hidden': 32,
    'gnn_spatial_layers': 1, 'gnn_spectral_layers': 1,
    'max_freqs': 4, 'tat_layers': 1,
    'tat_spatial_heads': 1, 'tat_spectral_heads': 1,
    'tat_ff_dim': 32, 'max_iterations': 2,
    'convergence_threshold': 0.1,
    'use_wave_dynamics': False,
    'use_higher_order': False,
}

LLM_CONFIG = {
    'llm_dim': 64, 'num_prefix': 2,
    'gate_threshold': 0.1, 'mock_hidden': 32,
}


class TestGatePenalty:
    def test_train_with_gate_penalty(self):
        max_cls = get_max_classes('graph_completion')  # 2
        model = HierarchicalMultiHopModel(**MC, use_llm=True, llm_config=LLM_CONFIG, max_classes=max_cls)
        ds = BenchmarkDataset(8, 'graph_completion', 12, 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss, gate_loss, gn = train_epoch_with_gate_penalty(
            model, ds, optimizer,
            gate_penalty_weight=1.0,
            accumulation_steps=2,
        )
        assert loss > 0
        assert gn > 0

    def test_train_without_gate_penalty(self):
        max_cls = get_max_classes('graph_completion')
        model = HierarchicalMultiHopModel(**MC, use_llm=True, llm_config=LLM_CONFIG, max_classes=max_cls)
        ds = BenchmarkDataset(8, 'graph_completion', 12, 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss, gate_loss, gn = train_epoch_with_gate_penalty(
            model, ds, optimizer,
            gate_penalty_weight=0.0,
            accumulation_steps=2,
        )
        assert loss > 0
        assert gate_loss == 0.0

    def test_train_on_labeled_reasoning(self):
        max_cls = get_max_classes('labeled_reasoning')  # 3
        model = HierarchicalMultiHopModel(**MC, use_llm=True, llm_config=LLM_CONFIG, max_classes=max_cls)
        ds = BenchmarkDataset(8, 'labeled_reasoning', 12, 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss, _, gn = train_epoch_with_gate_penalty(
            model, ds, optimizer,
            gate_penalty_weight=0.0,
            accumulation_steps=2,
        )
        assert loss > 0


class TestDiagnosticsLLMGate:
    def test_diagnostics_capture_llm_gate(self):
        model = HierarchicalMultiHopModel(**MC, use_llm=True, llm_config=LLM_CONFIG, max_classes=2)
        ds = BenchmarkDataset(5, 'graph_completion', 12, 16)
        collector = DiagnosticCollector()
        records = collector.collect(model, ds, torch.device('cpu'))
        assert len(records) == 5
        # Every record should have llm_gate
        for r in records:
            assert 'llm_gate' in r
            assert 0.0 <= r['llm_gate'] <= 1.0

    def test_diagnostics_summary_includes_llm_gate(self):
        model = HierarchicalMultiHopModel(**MC, use_llm=True, llm_config=LLM_CONFIG, max_classes=2)
        ds = BenchmarkDataset(5, 'graph_completion', 12, 16)
        collector = DiagnosticCollector()
        collector.collect(model, ds, torch.device('cpu'))
        summary = collector.summarize()
        assert 'llm_gate' in summary
        assert 'mean' in summary['llm_gate']
        assert 'std' in summary['llm_gate']

    def test_diagnostics_5_tuple_dataset(self):
        """Diagnostics work with 5-tuple (LLM task) datasets."""
        model = HierarchicalMultiHopModel(**MC, use_llm=True, llm_config=LLM_CONFIG, max_classes=2)
        ds = BenchmarkDataset(5, 'graph_completion', 12, 16)
        collector = DiagnosticCollector()
        records = collector.collect(model, ds, torch.device('cpu'))
        assert len(records) == 5


class TestTier3Tasks:
    def test_tier3_tasks_defined(self):
        assert "graph_completion" in TIER_3_TASKS
        assert "labeled_reasoning" in TIER_3_TASKS
        assert "analogical_transfer" in TIER_3_TASKS

    def test_build_hierarchical_llm(self):
        model = _build_model('hierarchical_llm', MC, 5, torch.device('cpu'),
                              llm_config=LLM_CONFIG)
        assert model.use_llm
        assert model.topo_bridge is not None


class TestConfigLoading:
    def test_config_loads(self):
        import yaml
        with open("config/benchmark_4c_llm.yaml") as f:
            config = yaml.safe_load(f)
        assert 'llm' in config
        assert 'curriculum' in config
        assert config['llm']['llm_dim'] == 128
        assert config['curriculum']['phase_a']['enabled'] is True
        assert config['curriculum']['phase_b']['enabled'] is True
        assert config['curriculum']['phase_c']['enabled'] is True
        assert config['benchmark']['tier'] == 3

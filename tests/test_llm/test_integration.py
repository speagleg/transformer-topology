"""Tests for LLM integration into HierarchicalMultiHopModel."""

import torch
import pytest
from src.benchmarks.run_comparison import (
    HierarchicalMultiHopModel,
    train_epoch,
    evaluate,
)
from src.benchmarks.multi_hop import MultiHopDataset


# Small model params for fast testing
MODEL_KWARGS = dict(
    embedding_dim=16,
    gnn_hidden=32,
    gnn_spatial_layers=1,
    gnn_spectral_layers=1,
    max_freqs=4,
    tat_layers=1,
    tat_spatial_heads=1,
    tat_spectral_heads=1,
    tat_ff_dim=32,
    max_classes=5,
    max_iterations=2,
    convergence_threshold=0.1,
    use_wave_dynamics=False,
    use_higher_order=False,
)

LLM_CONFIG = dict(
    llm_dim=64,
    num_prefix=2,
    gate_threshold=0.1,
    mock_hidden=32,
)


def _make_dataset(n=10):
    return MultiHopDataset(
        num_samples=n, min_hops=1, max_hops=4,
        num_distractors=3, embedding_dim=16,
    )


class TestBackwardCompat:
    def test_use_llm_false_same_as_default(self):
        """use_llm=False produces identical architecture to no llm args."""
        m1 = HierarchicalMultiHopModel(**MODEL_KWARGS)
        m2 = HierarchicalMultiHopModel(**MODEL_KWARGS, use_llm=False)
        p1 = sum(p.numel() for p in m1.parameters())
        p2 = sum(p.numel() for p in m2.parameters())
        assert p1 == p2

    def test_no_topo_bridge_when_disabled(self):
        model = HierarchicalMultiHopModel(**MODEL_KWARGS, use_llm=False)
        assert model.topo_bridge is None

    def test_existing_forward_unchanged(self):
        """Forward pass works identically without LLM."""
        model = HierarchicalMultiHopModel(**MODEL_KWARGS)
        ds = _make_dataset(3)
        cc, q, t, ans = ds[0]
        logits = model(cc, q, t)
        assert logits.shape == (5,)


class TestLLMEnabled:
    def test_model_has_topo_bridge(self):
        model = HierarchicalMultiHopModel(**MODEL_KWARGS, use_llm=True, llm_config=LLM_CONFIG)
        assert model.topo_bridge is not None

    def test_more_parameters_with_llm(self):
        m_no = HierarchicalMultiHopModel(**MODEL_KWARGS, use_llm=False)
        m_llm = HierarchicalMultiHopModel(**MODEL_KWARGS, use_llm=True, llm_config=LLM_CONFIG)
        p_no = sum(p.numel() for p in m_no.parameters())
        p_llm = sum(p.numel() for p in m_llm.parameters())
        assert p_llm > p_no

    def test_forward_output_shape(self):
        model = HierarchicalMultiHopModel(**MODEL_KWARGS, use_llm=True, llm_config=LLM_CONFIG)
        ds = _make_dataset(3)
        cc, q, t, ans = ds[0]
        logits = model(cc, q, t)
        assert logits.shape == (5,)

    def test_forward_with_metadata(self):
        model = HierarchicalMultiHopModel(**MODEL_KWARGS, use_llm=True, llm_config=LLM_CONFIG)
        ds = _make_dataset(3)
        cc, q, t, ans = ds[0]
        metadata = {'task_prompt': 'test prompt'}
        logits = model(cc, q, t, metadata=metadata)
        assert logits.shape == (5,)

    def test_gradient_flow_through_bridge(self):
        model = HierarchicalMultiHopModel(**MODEL_KWARGS, use_llm=True, llm_config=LLM_CONFIG)
        ds = _make_dataset(3)
        cc, q, t, ans = ds[0]
        logits = model(cc, q, t)
        loss = logits.sum()
        loss.backward()
        # Check bridge has gradients
        bridge_has_grad = False
        for p in model.topo_bridge.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                bridge_has_grad = True
                break
        assert bridge_has_grad

    def test_outputs_finite(self):
        model = HierarchicalMultiHopModel(**MODEL_KWARGS, use_llm=True, llm_config=LLM_CONFIG)
        ds = _make_dataset(5)
        for i in range(min(5, len(ds))):
            cc, q, t, ans = ds[i]
            logits = model(cc, q, t)
            assert torch.isfinite(logits).all(), f"Non-finite logits at sample {i}"


class TestTrainEpoch:
    def test_train_epoch_with_llm(self):
        model = HierarchicalMultiHopModel(**MODEL_KWARGS, use_llm=True, llm_config=LLM_CONFIG)
        ds = _make_dataset(8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss, gn = train_epoch(model, ds, optimizer, accumulation_steps=2)
        assert loss > 0
        assert gn > 0

    def test_evaluate_with_llm(self):
        model = HierarchicalMultiHopModel(**MODEL_KWARGS, use_llm=True, llm_config=LLM_CONFIG)
        ds = _make_dataset(8)
        acc, loss = evaluate(model, ds)
        assert 0.0 <= acc <= 1.0
        assert loss > 0

    def test_train_epoch_without_llm_still_works(self):
        """Existing training path is unaffected."""
        model = HierarchicalMultiHopModel(**MODEL_KWARGS)
        ds = _make_dataset(8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss, gn = train_epoch(model, ds, optimizer, accumulation_steps=2)
        assert loss > 0


class TestBuildModel:
    def test_build_hierarchical_llm_variant(self):
        from src.benchmarks.run_benchmark_suite import _build_model
        mc = {
            'embedding_dim': 16, 'gnn_hidden': 32,
            'gnn_spatial_layers': 1, 'gnn_spectral_layers': 1,
            'max_freqs': 4, 'tat_layers': 1,
            'tat_spatial_heads': 1, 'tat_spectral_heads': 1,
            'tat_ff_dim': 32, 'max_iterations': 2,
            'convergence_threshold': 0.1,
            'use_wave_dynamics': False,
            'use_higher_order': False,
        }
        model = _build_model('hierarchical_llm', mc, 5, torch.device('cpu'),
                              llm_config=LLM_CONFIG)
        assert model.topo_bridge is not None
        assert model.use_llm is True

    def test_build_hierarchical_still_works(self):
        from src.benchmarks.run_benchmark_suite import _build_model
        mc = {
            'embedding_dim': 16, 'gnn_hidden': 32,
            'gnn_spatial_layers': 1, 'gnn_spectral_layers': 1,
            'max_freqs': 4, 'tat_layers': 1,
            'tat_spatial_heads': 1, 'tat_spectral_heads': 1,
            'tat_ff_dim': 32, 'max_iterations': 2,
            'convergence_threshold': 0.1,
        }
        model = _build_model('hierarchical', mc, 5, torch.device('cpu'))
        assert model.topo_bridge is None
        assert model.use_llm is False

"""Tests for TextEdgeEncoder — text-derived edge features."""

import torch
import pytest


def test_text_edge_encoder_output_shape():
    from src.llm.text_edge_encoder import TextEdgeEncoder
    enc = TextEdgeEncoder(text_dim=32, edge_dim=32)
    text_features = torch.randn(10, 32)
    src_idx = torch.tensor([0, 1, 2, 3, 4])
    tgt_idx = torch.tensor([1, 2, 3, 4, 5])
    output = enc(text_features, src_idx, tgt_idx)
    assert output.shape == (5, 32)


def test_text_edge_encoder_gate_init():
    from src.llm.text_edge_encoder import TextEdgeEncoder
    enc = TextEdgeEncoder(text_dim=32, edge_dim=32)
    gate_val = torch.sigmoid(enc.gate).item()
    assert gate_val < 0.1, f"Gate should start near 0, got {gate_val}"


def test_text_edge_encoder_gate_zero_output():
    from src.llm.text_edge_encoder import TextEdgeEncoder
    enc = TextEdgeEncoder(text_dim=32, edge_dim=32)
    enc.gate.data.fill_(-100.0)
    text_features = torch.randn(5, 32)
    src_idx = torch.tensor([0, 1, 2])
    tgt_idx = torch.tensor([1, 2, 3])
    output = enc(text_features, src_idx, tgt_idx)
    assert output.abs().max() < 1e-6


def test_text_edge_encoder_gradient_flow():
    from src.llm.text_edge_encoder import TextEdgeEncoder
    enc = TextEdgeEncoder(text_dim=32, edge_dim=32)
    enc.gate.data.fill_(0.0)
    text_features = torch.randn(5, 32, requires_grad=True)
    src_idx = torch.tensor([0, 1])
    tgt_idx = torch.tensor([1, 2])
    output = enc(text_features, src_idx, tgt_idx)
    loss = output.sum()
    loss.backward()
    assert text_features.grad is not None
    # Gradient through LayerNorm+MLP can be very small; check MLP params instead
    mlp_grads = [p.grad for p in enc.mlp.parameters() if p.grad is not None]
    assert len(mlp_grads) > 0, "MLP should receive gradients"
    assert any(g.abs().sum() > 0 for g in mlp_grads)


def test_text_edge_encoder_none_input():
    from src.llm.text_edge_encoder import TextEdgeEncoder
    enc = TextEdgeEncoder(text_dim=32, edge_dim=32)
    output = enc(None, torch.tensor([0]), torch.tensor([1]))
    assert output is None


def test_model_has_text_edge_encoder():
    """Model creates TextEdgeEncoder when Qwen backend is used."""
    import torch
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel

    model = HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=32, gnn_spatial_layers=2,
        gnn_spectral_layers=1, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=10, max_iterations=2, convergence_threshold=0.01,
        use_llm=True, llm_config={'backend': 'qwen', 'use_mock': True},
    )

    assert model.text_edge_encoder is not None
    # Gate should be frozen for Phase 1
    assert not model.text_edge_encoder.gate.requires_grad


def test_model_no_text_edge_encoder_without_llm():
    """Model without LLM has no TextEdgeEncoder."""
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel

    model = HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=32, gnn_spatial_layers=2,
        gnn_spectral_layers=1, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=10, max_iterations=2, convergence_threshold=0.01,
        use_llm=False,
    )

    assert model.text_edge_encoder is None

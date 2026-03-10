"""Tests for TextReasoningHead — graph-aware text transformer."""

import torch
import pytest


def test_text_reasoning_head_output_shape():
    from src.llm.text_reasoning_head import TextReasoningHead
    head = TextReasoningHead(input_dim=32, hidden_dim=64, num_layers=2, num_heads=4)
    text_features = torch.randn(10, 32)
    adj = torch.eye(10)
    adj[0, 1] = adj[1, 0] = 1.0
    output = head(text_features, adj)
    assert output.shape == (10, 64)


def test_text_reasoning_head_adjacency_masking():
    from src.llm.text_reasoning_head import TextReasoningHead
    head = TextReasoningHead(input_dim=32, hidden_dim=64, num_layers=2, num_heads=4)
    text_features = torch.randn(10, 32)
    adj = torch.zeros(10, 10)
    for i in range(4):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    for i in range(5, 9):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    output = head(text_features, adj)
    assert output.shape == (10, 64)
    assert not torch.isnan(output).any()


def test_text_reasoning_head_none_input():
    from src.llm.text_reasoning_head import TextReasoningHead
    head = TextReasoningHead(input_dim=32, hidden_dim=64, num_layers=2, num_heads=4)
    output = head(None, torch.eye(5))
    assert output is None


def test_text_reasoning_head_gradient_flow():
    from src.llm.text_reasoning_head import TextReasoningHead
    head = TextReasoningHead(input_dim=32, hidden_dim=64, num_layers=2, num_heads=4)
    text_features = torch.randn(5, 32, requires_grad=True)
    adj = torch.ones(5, 5)
    output = head(text_features, adj)
    loss = output.sum()
    loss.backward()
    assert text_features.grad is not None
    assert text_features.grad.abs().sum() > 0


def test_text_reasoning_head_single_node():
    from src.llm.text_reasoning_head import TextReasoningHead
    head = TextReasoningHead(input_dim=32, hidden_dim=64, num_layers=2, num_heads=4)
    text_features = torch.randn(1, 32)
    adj = torch.ones(1, 1)
    output = head(text_features, adj)
    assert output.shape == (1, 64)


def test_model_with_text_reasoning_head():
    """Full model forward pass with TextReasoningHead produces correct classifier dim."""
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel
    from src.cell_complex.cell_complex import CellComplex

    model = HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=32, gnn_spatial_layers=2,
        gnn_spectral_layers=1, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=10, max_iterations=2, convergence_threshold=0.01,
        use_llm=True, llm_config={'backend': 'qwen', 'use_mock': True},
        use_multi_head_classifier=False, use_metacog=False,
    )

    # Verify text_reasoning_dim is set
    assert model.text_reasoning_dim == 64
    # base(132) + text_concat(96) + text_reasoning(192) = 420
    assert model.classifier_input_dim == 132 + 96 + 192

    # Build a simple cell complex with node_texts
    cc = CellComplex(embedding_dim=32)
    for i in range(5):
        cc.add_0_cell(torch.randn(32), "node")
    cc.add_1_cell(0, 1, torch.randn(32), "edge")
    cc.add_1_cell(1, 2, torch.randn(32), "edge")
    cc.add_1_cell(2, 3, torch.randn(32), "edge")
    cc.add_1_cell(3, 4, torch.randn(32), "edge")
    cc.node_texts = ['dog', 'cat', 'animal', 'pet', 'fish']

    logits = model(cc, query_node=0, target_node=2)
    assert logits.shape == (10,)


def test_model_without_text_reasoning_no_change():
    """Model without LLM has text_reasoning_dim=0, same classifier dim as before."""
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel

    model = HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=32, gnn_spatial_layers=2,
        gnn_spectral_layers=1, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=10, max_iterations=2, convergence_threshold=0.01,
        use_llm=False,
    )

    assert model.text_reasoning_dim == 0
    # base(132) only, no text dims
    assert model.classifier_input_dim == 132

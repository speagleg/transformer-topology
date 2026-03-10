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

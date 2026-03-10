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
    assert text_features.grad.abs().sum() > 0


def test_text_edge_encoder_none_input():
    from src.llm.text_edge_encoder import TextEdgeEncoder
    enc = TextEdgeEncoder(text_dim=32, edge_dim=32)
    output = enc(None, torch.tensor([0]), torch.tensor([1]))
    assert output is None

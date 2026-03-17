"""Tests for text-enhanced classifier input and bidirectional cross-attention."""
import torch
import pytest
from src.reasoning_loop.attention_readout import AttentionReadout
from src.reasoning_loop.cross_attention import BidirectionalCrossAttention


def test_output_dim_with_text():
    readout = AttentionReadout(embed_dim=32, num_tasks=19, text_dim=32)
    assert readout.output_dim == 165


def test_output_dim_without_text():
    readout = AttentionReadout(embed_dim=32, num_tasks=19, text_dim=0)
    assert readout.output_dim == 101


def test_output_dim_default_text_dim():
    readout = AttentionReadout(embed_dim=32, num_tasks=19)
    assert readout.output_dim == 101


def test_build_classifier_input_with_text():
    readout = AttentionReadout(embed_dim=32, num_tasks=19, text_dim=32)
    h_out = torch.randn(10, 32)
    text_embs = torch.randn(10, 32)
    topo = torch.randn(4)
    fw = torch.tensor(0.5)
    result = readout.build_classifier_input(
        h_out, query_idx=0, target_idx=1, task_id=0,
        topo_features=topo, fusion_weight=fw,
        text_embeddings=text_embs,
    )
    assert result.shape == (165,)


def test_build_classifier_input_without_text_backward_compat():
    readout = AttentionReadout(embed_dim=32, num_tasks=19, text_dim=0)
    h_out = torch.randn(10, 32)
    topo = torch.randn(4)
    fw = torch.tensor(0.5)
    result = readout.build_classifier_input(
        h_out, query_idx=0, target_idx=1, task_id=0,
        topo_features=topo, fusion_weight=fw,
    )
    assert result.shape == (101,)


def test_text_dim_with_none_text_pads_zeros():
    readout = AttentionReadout(embed_dim=32, num_tasks=19, text_dim=32)
    h_out = torch.randn(10, 32)
    topo = torch.randn(4)
    fw = torch.tensor(0.5)
    result = readout.build_classifier_input(
        h_out, query_idx=0, target_idx=1, task_id=0,
        topo_features=topo, fusion_weight=fw,
        text_embeddings=None,
    )
    assert result.shape == (165,)
    assert result[-64:].abs().sum() == 0.0


def test_bidirectional_cross_attention_output_shape():
    bidir = BidirectionalCrossAttention(embed_dim=32, num_heads=4)
    h_struct = torch.randn(10, 32)
    h_text = torch.randn(10, 32)
    h_fused, h_text_enriched = bidir(h_struct, h_text)
    assert h_fused.shape == (10, 32)
    assert h_text_enriched.shape == (10, 32)


def test_bidirectional_gradients_flow():
    bidir = BidirectionalCrossAttention(embed_dim=32, num_heads=4)
    h_struct = torch.randn(10, 32, requires_grad=True)
    h_text = torch.randn(10, 32, requires_grad=True)
    h_fused, h_text_enriched = bidir(h_struct, h_text)
    loss = h_fused.sum() + h_text_enriched.sum()
    loss.backward()
    assert h_struct.grad is not None
    assert h_text.grad is not None

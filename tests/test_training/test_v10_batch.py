"""Tests for v10 batch forward with attention readout and dual-track support."""

import torch
from src.reasoning_loop.attention_readout import AttentionReadout


class TestV10BatchForward:

    def test_attention_readout_builds_correct_dim(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        topo = torch.randn(4)
        fw = torch.tensor(0.5)
        combined = readout.build_classifier_input(
            h_out, 0, 3, task_id=5, topo_features=topo, fusion_weight=fw,
        )
        assert combined.shape == (101,)

    def test_attention_readout_without_optional_features(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        combined = readout.build_classifier_input(h_out, 0, 3, task_id=5)
        # 32 + 32 + 32 = 96 (no topo, no fusion)
        assert combined.shape == (96,)

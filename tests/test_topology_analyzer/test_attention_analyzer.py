"""Tests for AttentionFlowAnalyzer."""

import torch
import pytest


class TestAttentionFlowAnalyzer:
    def test_build_cell_complex(self):
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer()
        attn = torch.rand(8, 8)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        lcc = analyzer.build_cell_complex(attn, layer_idx=0, head_idx=0)
        assert lcc.cc.num_cells(0) == 8
        assert lcc.source == "attention"
        assert lcc.head_idx == 0

    def test_adaptive_threshold(self):
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer(threshold_mode="adaptive")
        attn = torch.zeros(8, 8) + 0.01
        attn[0, 1] = 0.9
        attn[2, 3] = 0.8
        lcc = analyzer.build_cell_complex(attn, layer_idx=0, head_idx=0)
        assert lcc.cc.num_cells(1) >= 2
        assert lcc.cc.num_cells(1) < 30

    def test_hodge_analysis(self):
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer()
        attn = torch.rand(8, 8)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        lcc = analyzer.build_cell_complex(attn, layer_idx=0, head_idx=0)
        result = analyzer.analyze_head(lcc)
        assert "hodge_ratios" in result
        assert len(result["hodge_ratios"]) == 3

    def test_classify_head_gradient(self):
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer()
        assert analyzer.classify_head((0.7, 0.2, 0.1)) == "gradient"

    def test_classify_head_curl(self):
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer()
        assert analyzer.classify_head((0.2, 0.6, 0.2)) == "curl"

    def test_classify_head_harmonic(self):
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer()
        assert analyzer.classify_head((0.1, 0.2, 0.7)) == "harmonic"

    def test_fixed_threshold(self):
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer(threshold_mode="fixed", threshold=0.3)
        attn = torch.zeros(4, 4) + 0.1
        attn[0, 1] = 0.5
        attn[1, 2] = 0.4
        lcc = analyzer.build_cell_complex(attn, layer_idx=0, head_idx=0)
        assert lcc.cc.num_cells(1) == 2

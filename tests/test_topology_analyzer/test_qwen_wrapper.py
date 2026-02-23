"""Tests for QwenAnalysisWrapper."""
import torch
from src.topology_analyzer.qwen_wrapper import QwenAnalysisWrapper


class TestQwenAnalysisWrapper:
    def test_forward_shape(self):
        wrapper = QwenAnalysisWrapper(llm_dim=64, num_layers=4, use_mock=True)
        x = torch.randn(1, 10, 64)
        out = wrapper(x)
        assert out.dim() == 3

    def test_exposes_layers(self):
        wrapper = QwenAnalysisWrapper(llm_dim=64, num_layers=4, use_mock=True)
        assert hasattr(wrapper, 'layers')
        assert len(wrapper.layers) == 4

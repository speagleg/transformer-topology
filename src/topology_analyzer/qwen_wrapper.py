"""Wrap Qwen (or mock) for TransformerTopologyAnalyzer.

The embedding topology analyzer needs a model with `.layers` attribute
and `forward(x)` -> output. This wraps the frozen Qwen or a mock.
"""
import torch
import torch.nn as nn


class QwenAnalysisWrapper(nn.Module):
    """Wraps a transformer model for topology analysis.

    Provides the `.layers` and single-arg `forward(x)` interface
    that TransformerTopologyAnalyzer expects.
    """

    def __init__(self, llm_dim=2048, num_layers=32, use_mock=False,
                 qwen_model=None):
        super().__init__()
        if use_mock or qwen_model is None:
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    llm_dim, nhead=4, dim_feedforward=4 * llm_dim,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ])
            self._is_mock = True
        else:
            self.layers = qwen_model.model.layers
            self._is_mock = False
            self._qwen = qwen_model

    def forward(self, x):
        if self._is_mock:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            with torch.no_grad():
                out = self._qwen(
                    inputs_embeds=x,
                    output_hidden_states=False,
                )
                return out.last_hidden_state

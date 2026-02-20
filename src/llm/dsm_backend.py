"""DSM backend: wraps DistilledSemanticModel to implement BaseLLMBackend."""

import torch
import torch.nn as nn

from src.llm.backend import BaseLLMBackend
from src.llm.dsm import DistilledSemanticModel


class DSMBackend(nn.Module, BaseLLMBackend):
    """Wraps the DSM as a drop-in replacement for MockLLMBackend/LlamaBackend.

    Unlike LlamaBackend, this is fully trainable (no frozen weights, no LoRA).
    The llm_dim for TopoBridge should match dsm_dim.
    """

    def __init__(self, dsm_dim: int = 1024, num_heads: int = 16,
                 ff_dim: int = 4096, num_layers: int = 16,
                 cross_attn_layer: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dsm_dim = dsm_dim
        self.dsm = DistilledSemanticModel(
            hidden_dim=dsm_dim, num_heads=num_heads, ff_dim=ff_dim,
            num_layers=num_layers, cross_attn_layer=cross_attn_layer,
            dropout=dropout,
        )

    def forward(
        self,
        prefix_tokens: torch.Tensor,
        topo_memory: torch.Tensor,
        task_text: str | None = None,
    ) -> torch.Tensor:
        """Run DSM forward pass.

        Args:
            prefix_tokens: (num_prefix, dsm_dim) from TopoBridge encoder.
            topo_memory: (N, dsm_dim) projected node embeddings.
            task_text: Ignored (no tokenizer in DSM). Kept for interface compat.

        Returns:
            hidden_states: (num_prefix, dsm_dim) DSM hidden states for decoder.
        """
        return self.dsm(prefix_tokens, topo_memory)

    def parameters(self, recurse=True):
        """All DSM parameters are trainable."""
        return self.dsm.parameters(recurse=recurse)

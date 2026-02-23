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

    def __init__(self, config=None, dsm_dim: int = 1024, num_heads: int = 16,
                 ff_dim: int = 4096, num_layers: int = 16,
                 cross_attn_layer: int = 4, dropout: float = 0.1):
        super().__init__()
        # Support both dict config and keyword args
        if config is not None:
            dsm_dim = config.get("dsm_dim", dsm_dim)
            num_heads = config.get("num_heads", num_heads)
            ff_dim = config.get("ff_dim", ff_dim)
            num_layers = config.get("num_layers", num_layers)
            cross_attn_layer = config.get("cross_attn_layer", cross_attn_layer)
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
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run DSM forward pass.

        Args:
            prefix_tokens: (num_prefix, dsm_dim) or (num_prefix, batch, dsm_dim).
            topo_memory: (N, dsm_dim) or (max_N, batch, dsm_dim).
            task_text: Ignored (no tokenizer in DSM). Kept for interface compat.
            memory_key_padding_mask: (batch, max_N) bool mask where True = ignore.

        Returns:
            hidden_states: (num_prefix, dsm_dim) or (num_prefix, batch, dsm_dim).
        """
        return self.dsm(prefix_tokens, topo_memory,
                        memory_key_padding_mask=memory_key_padding_mask)

    def parameters(self, recurse=True):
        """All DSM parameters are trainable."""
        return self.dsm.parameters(recurse=recurse)

    def freeze_all(self):
        """Freeze all DSM parameters."""
        for p in self.dsm.parameters():
            p.requires_grad = False

    def unfreeze_top_n(self, n: int):
        """Unfreeze the top N transformer layers (highest index = closest to output)."""
        num_layers = len(self.dsm.layers)
        for i, layer in enumerate(self.dsm.layers):
            if i >= num_layers - n:
                for p in layer.parameters():
                    p.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all DSM parameters."""
        for p in self.dsm.parameters():
            p.requires_grad = True

    def load_pretrained(self, state_dict: dict):
        """Load pre-trained weights (from WikiText pre-training)."""
        self.dsm.load_state_dict(state_dict, strict=False)

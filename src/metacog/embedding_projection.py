"""Project LLM hidden states into the topology embedding space."""

import torch
import torch.nn as nn


class EmbeddingProjection(nn.Module):
    """Linear projection from LLM hidden dimension to topology embedding dimension.

    Args:
        llm_dim: Dimension of the LLM hidden states (default: 2048).
        embed_dim: Target embedding dimension for the cell complex (default: 128).
    """

    def __init__(self, llm_dim: int = 2048, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(llm_dim, embed_dim)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project hidden states to topology embedding space.

        Args:
            hidden_state: Tensor of shape (..., llm_dim).

        Returns:
            Tensor of shape (..., embed_dim).
        """
        return self.proj(hidden_state)

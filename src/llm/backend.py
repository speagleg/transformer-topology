"""LLM backend interface and mock implementation for CPU testing."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseLLMBackend(ABC):
    """Abstract interface for LLM backends.

    Implementations must accept prefix tokens and topo_memory from the
    TopoBridge encoder and return hidden states for the decoder.
    """

    @abstractmethod
    def forward(
        self,
        prefix_tokens: torch.Tensor,
        topo_memory: torch.Tensor,
        task_text: str | None = None,
    ) -> torch.Tensor:
        """Run LLM forward pass.

        Args:
            prefix_tokens: (num_prefix, llm_dim) prefix embeddings.
            topo_memory: (N, llm_dim) projected node embeddings.
            task_text: Optional text prompt describing the task.

        Returns:
            hidden_states: (seq_len, llm_dim) LLM hidden states for decoder.
        """
        ...

    @abstractmethod
    def parameters(self):
        """Return trainable parameters (for optimizer)."""
        ...


class MockLLMBackend(nn.Module, BaseLLMBackend):
    """Tiny stand-in for a real LLM (~5K params). Shape and gradient correct.

    Uses a 2-layer MLP to transform concatenated prefix+memory tokens.
    No text processing — just ensures the TopoBridge data path works.
    """

    def __init__(self, llm_dim: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.llm_dim = llm_dim
        self.mlp = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(
        self,
        prefix_tokens: torch.Tensor,
        topo_memory: torch.Tensor,
        task_text: str | None = None,
    ) -> torch.Tensor:
        """Process prefix + topo_memory through MLP.

        Returns hidden states with same seq_len as prefix+memory combined.
        """
        # Concatenate prefix tokens and topo memory along sequence dim
        # prefix_tokens: (num_prefix, llm_dim), topo_memory: (N, llm_dim)
        combined = torch.cat([prefix_tokens, topo_memory], dim=0)  # (seq, llm_dim)
        return self.mlp(combined)  # (seq, llm_dim)

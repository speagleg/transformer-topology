"""TopoBridge: encoder/decoder for translating between topo and LLM spaces."""

import torch
import torch.nn as nn

from src.llm.backend import BaseLLMBackend


class TopoBridgeEncoder(nn.Module):
    """Projects TAT node embeddings into LLM space and generates prefix tokens.

    Components:
        NodeProjector: Linear(topo_dim→llm_dim) + LayerNorm → topo_memory (N, llm_dim)
        PrefixGenerator: K learned queries attend over topo_memory → prefix (K, llm_dim)
    """

    def __init__(self, topo_dim: int = 32, llm_dim: int = 2048, num_prefix: int = 8):
        super().__init__()
        self.topo_dim = topo_dim
        self.llm_dim = llm_dim
        self.num_prefix = num_prefix

        # NodeProjector: project each node embedding into LLM hidden space
        self.node_proj = nn.Sequential(
            nn.Linear(topo_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

        # PrefixGenerator: learned queries that attend over projected nodes
        self.prefix_queries = nn.Parameter(torch.randn(num_prefix, llm_dim) * 0.02)
        self.prefix_attn = nn.MultiheadAttention(
            llm_dim, num_heads=8, batch_first=False,
        )

    def forward(self, node_embeddings: torch.Tensor):
        """Encode TAT output into LLM-space representations.

        Args:
            node_embeddings: (N, topo_dim) from TAT output.

        Returns:
            topo_memory: (N, llm_dim) projected node embeddings.
            prefix_tokens: (num_prefix, llm_dim) attention-pooled prefix tokens.
        """
        # Project nodes into LLM space
        topo_memory = self.node_proj(node_embeddings)  # (N, llm_dim)

        # Generate prefix tokens via cross-attention
        # queries: (num_prefix, 1, llm_dim), key/value: (N, 1, llm_dim)
        queries = self.prefix_queries.unsqueeze(1)  # (K, 1, llm_dim)
        kv = topo_memory.unsqueeze(1)  # (N, 1, llm_dim)
        prefix_tokens, _ = self.prefix_attn(queries, kv, kv)  # (K, 1, llm_dim)
        prefix_tokens = prefix_tokens.squeeze(1)  # (K, llm_dim)

        return topo_memory, prefix_tokens


class TopoBridgeDecoder(nn.Module):
    """Extracts node-aligned embeddings and semantic bias from LLM hidden states."""

    def __init__(self, topo_dim: int = 32, llm_dim: int = 2048):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            llm_dim, num_heads=8, batch_first=False,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(llm_dim, topo_dim),
            nn.LayerNorm(topo_dim),
        )
        # Semantic bias: projects node embeddings for pairwise attention bias
        self.semantic_bias_proj = nn.Linear(topo_dim, topo_dim)

    def forward(
        self, topo_memory: torch.Tensor, llm_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode LLM hidden states back to node-aligned topo embeddings + bias.

        Args:
            topo_memory: (N, llm_dim) from encoder — used as queries.
            llm_hidden: (seq_len, llm_dim) LLM hidden states — keys/values.

        Returns:
            llm_out: (N, topo_dim) node-aligned output embeddings.
            semantic_bias: (N, N) pairwise attention bias for TAT.
        """
        # Cross-attention: node queries attend over LLM hidden states
        # queries: (N, 1, llm_dim), kv: (seq, 1, llm_dim)
        queries = topo_memory.unsqueeze(1)  # (N, 1, llm_dim)
        kv = llm_hidden.unsqueeze(1)  # (seq, 1, llm_dim)
        attn_out, _ = self.cross_attn(queries, kv, kv)  # (N, 1, llm_dim)
        attn_out = attn_out.squeeze(1)  # (N, llm_dim)

        llm_out = self.out_proj(attn_out)  # (N, topo_dim)

        # Compute semantic bias: pairwise similarity in projected space
        projected = self.semantic_bias_proj(llm_out)  # (N, topo_dim)
        semantic_bias = projected @ projected.T  # (N, N)

        return llm_out, semantic_bias


class TopoBridge(nn.Module):
    """Composes encoder + LLM backend + decoder for topo→LLM→topo round trip."""

    def __init__(
        self,
        backend: BaseLLMBackend,
        topo_dim: int = 32,
        llm_dim: int = 2048,
        num_prefix: int = 8,
        gate_threshold: float = 0.1,  # kept for backward compat signature
    ):
        super().__init__()
        self.encoder = TopoBridgeEncoder(topo_dim, llm_dim, num_prefix)
        self.decoder = TopoBridgeDecoder(topo_dim, llm_dim)
        self.backend = backend
        self.topo_dim = topo_dim

    def forward(
        self,
        node_embeddings: torch.Tensor,
        semantic_weight: torch.Tensor,
        task_text: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full TopoBridge forward pass — always runs (no gate skip).

        Args:
            node_embeddings: (N, topo_dim) from TAT/GNN output.
            semantic_weight: scalar [0,1] from ControlHead (unused here but kept for interface).
            task_text: Optional task description for the LLM/DSM.

        Returns:
            llm_out: (N, topo_dim) node-aligned output.
            semantic_bias: (N, N) pairwise bias for TAT attention.
        """
        # Encode
        topo_memory, prefix_tokens = self.encoder(node_embeddings)

        # LLM forward
        llm_hidden = self.backend.forward(prefix_tokens, topo_memory, task_text)

        # Decode back to topo space + semantic bias
        llm_out, semantic_bias = self.decoder(topo_memory, llm_hidden)

        return llm_out, semantic_bias

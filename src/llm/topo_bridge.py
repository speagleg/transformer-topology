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
    """Extracts node-aligned embeddings from LLM hidden states back to topo space.

    Uses cross-attention: topo_memory queries attend over LLM hidden states,
    then projects back to topo_dim.
    """

    def __init__(self, topo_dim: int = 32, llm_dim: int = 2048):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            llm_dim, num_heads=8, batch_first=False,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(llm_dim, topo_dim),
            nn.LayerNorm(topo_dim),
        )

    def forward(
        self, topo_memory: torch.Tensor, llm_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Decode LLM hidden states back to node-aligned topo embeddings.

        Args:
            topo_memory: (N, llm_dim) from encoder — used as queries.
            llm_hidden: (seq_len, llm_dim) LLM hidden states — keys/values.

        Returns:
            llm_out: (N, topo_dim) node-aligned output embeddings.
        """
        # Cross-attention: node queries attend over LLM hidden states
        # queries: (N, 1, llm_dim), kv: (seq, 1, llm_dim)
        queries = topo_memory.unsqueeze(1)  # (N, 1, llm_dim)
        kv = llm_hidden.unsqueeze(1)  # (seq, 1, llm_dim)
        attn_out, _ = self.cross_attn(queries, kv, kv)  # (N, 1, llm_dim)
        attn_out = attn_out.squeeze(1)  # (N, llm_dim)

        return self.out_proj(attn_out)  # (N, topo_dim)


class TopoBridge(nn.Module):
    """Composes encoder + LLM backend + decoder for full topo→LLM→topo round trip.

    When llm_gate < threshold, skips the LLM entirely and returns zeros.
    """

    def __init__(
        self,
        backend: BaseLLMBackend,
        topo_dim: int = 32,
        llm_dim: int = 2048,
        num_prefix: int = 8,
        gate_threshold: float = 0.1,
    ):
        super().__init__()
        self.encoder = TopoBridgeEncoder(topo_dim, llm_dim, num_prefix)
        self.decoder = TopoBridgeDecoder(topo_dim, llm_dim)
        self.backend = backend
        self.topo_dim = topo_dim
        self.gate_threshold = gate_threshold

    def forward(
        self,
        node_embeddings: torch.Tensor,
        llm_gate: torch.Tensor,
        task_text: str | None = None,
    ) -> torch.Tensor:
        """Full TopoBridge forward pass with gate-based skip.

        Args:
            node_embeddings: (N, topo_dim) from TAT output.
            llm_gate: scalar [0,1] from ControlHead.
            task_text: Optional task description for the LLM.

        Returns:
            llm_out: (N, topo_dim) — node-aligned LLM output, or zeros if gate < threshold.
        """
        n_nodes = node_embeddings.shape[0]
        device = node_embeddings.device

        if llm_gate.item() < self.gate_threshold:
            return torch.zeros(n_nodes, self.topo_dim, device=device)

        # Encode
        topo_memory, prefix_tokens = self.encoder(node_embeddings)

        # LLM forward
        llm_hidden = self.backend.forward(prefix_tokens, topo_memory, task_text)

        # Decode back to topo space
        llm_out = self.decoder(topo_memory, llm_hidden)

        # Scale by gate value so gradients flow through the gate
        return llm_out * llm_gate

"""TopoBridge: encoder/decoder for translating between topo and LLM spaces."""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward_batched(
        self, node_embeddings_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Batched encode: pad variable-size node embeddings and run in one pass.

        Args:
            node_embeddings_list: List of B tensors, each (N_i, topo_dim).

        Returns:
            topo_memory: (max_N, B, llm_dim) padded projected nodes.
            prefix_tokens: (K, B, llm_dim) prefix tokens per graph.
            memory_mask: (B, max_N) bool mask, True = padding position.
            sizes: list of N_i per graph (for unpadding).
        """
        B = len(node_embeddings_list)
        sizes = [ne.shape[0] for ne in node_embeddings_list]
        max_N = max(sizes)
        device = node_embeddings_list[0].device

        # Pad and stack node embeddings: (B, max_N, topo_dim)
        padded = torch.zeros(B, max_N, self.topo_dim, device=device)
        memory_mask = torch.ones(B, max_N, dtype=torch.bool, device=device)
        for i, ne in enumerate(node_embeddings_list):
            padded[i, :sizes[i]] = ne
            memory_mask[i, :sizes[i]] = False  # False = attend

        # Project: (B, max_N, topo_dim) → (B, max_N, llm_dim)
        topo_memory_bnd = self.node_proj(padded)  # (B, max_N, llm_dim)
        # Reshape to (max_N, B, llm_dim) for batch_first=False MHA
        topo_memory = topo_memory_bnd.transpose(0, 1)  # (max_N, B, llm_dim)

        # Generate prefix tokens via batched cross-attention
        # queries: (K, B, llm_dim), kv: (max_N, B, llm_dim)
        queries = self.prefix_queries.unsqueeze(1).expand(-1, B, -1)  # (K, B, llm_dim)
        prefix_tokens, _ = self.prefix_attn(
            queries, topo_memory, topo_memory,
            key_padding_mask=memory_mask,
        )  # (K, B, llm_dim)

        return topo_memory, prefix_tokens, memory_mask, sizes


class TopoBridgeDecoder(nn.Module):
    """Extracts node-aligned embeddings, semantic bias, features, and graph embedding."""

    def __init__(self, topo_dim: int = 32, llm_dim: int = 2048):
        super().__init__()
        self.topo_dim = topo_dim
        self.cross_attn = nn.MultiheadAttention(
            llm_dim, num_heads=8, batch_first=False,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(llm_dim, topo_dim),
            nn.LayerNorm(topo_dim),
        )
        # Semantic bias: projects node embeddings for pairwise attention bias
        self.semantic_bias_proj = nn.Linear(topo_dim, topo_dim)
        # Initialize so semantic_bias starts moderate (not overwhelming TAT).
        nn.init.normal_(self.semantic_bias_proj.weight, std=0.1)
        nn.init.zeros_(self.semantic_bias_proj.bias)
        # Graph-level embedding: pool LLM hidden states → topo_dim
        self.graph_pool = nn.Linear(llm_dim, topo_dim)

    def forward(
        self, topo_memory: torch.Tensor, llm_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode LLM hidden states back to node-aligned topo embeddings + bias + features + graph emb.

        Args:
            topo_memory: (N, llm_dim) from encoder — used as queries.
            llm_hidden: (seq_len, llm_dim) LLM hidden states — keys/values.

        Returns:
            semantic_out: (N, topo_dim) node-aligned output embeddings.
            semantic_bias: (N, N) pairwise attention bias for TAT.
            semantic_features: (N, topo_dim) per-node semantic features (for contrastive loss).
            graph_embedding: (topo_dim,) graph-level embedding.
        """
        # Cross-attention: node queries attend over LLM hidden states
        # queries: (N, 1, llm_dim), kv: (seq, 1, llm_dim)
        queries = topo_memory.unsqueeze(1)  # (N, 1, llm_dim)
        kv = llm_hidden.unsqueeze(1)  # (seq, 1, llm_dim)
        attn_out, _ = self.cross_attn(queries, kv, kv)  # (N, 1, llm_dim)
        attn_out = attn_out.squeeze(1)  # (N, llm_dim)

        semantic_out = self.out_proj(attn_out)  # (N, topo_dim)

        # Compute semantic bias: pairwise similarity in projected space
        projected = self.semantic_bias_proj(semantic_out)  # (N, topo_dim)
        semantic_bias = projected @ projected.T  # (N, N)

        # Semantic features = the projected node embeddings (for contrastive loss)
        semantic_features = semantic_out  # (N, topo_dim)

        # Graph-level embedding: mean-pool LLM hidden states → topo_dim
        graph_embedding = self.graph_pool(llm_hidden.mean(dim=0))  # (topo_dim,)

        return semantic_out, semantic_bias, semantic_features, graph_embedding

    def forward_batched(
        self,
        topo_memory: torch.Tensor,
        llm_hidden: torch.Tensor,
        memory_mask: torch.Tensor,
        sizes: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Batched decode: extract per-graph node embeddings + semantic bias + features + graph emb.

        Args:
            topo_memory: (max_N, B, llm_dim) padded projected nodes.
            llm_hidden: (K, B, llm_dim) DSM output.
            memory_mask: (B, max_N) bool mask, True = padding.
            sizes: list of N_i per graph.

        Returns:
            List of (semantic_out_i, semantic_bias_i, semantic_features_i, graph_embedding_i).
        """
        B = topo_memory.shape[1]

        # Cross-attention: node queries attend over DSM hidden states
        attn_out, _ = self.cross_attn(topo_memory, llm_hidden, llm_hidden)
        # (max_N, B, llm_dim)

        # Reshape to (B, max_N, llm_dim) for per-sample processing
        attn_out = attn_out.transpose(0, 1)  # (B, max_N, llm_dim)

        # Project to topo space: (B, max_N, topo_dim)
        semantic_out_batched = self.out_proj(attn_out)

        # Per-graph graph-level embedding from LLM hidden states
        # llm_hidden: (K, B, llm_dim) → mean over K → (B, llm_dim)
        llm_pooled = llm_hidden.mean(dim=0)  # (B, llm_dim)

        # Unpad and compute per-graph outputs
        results = []
        for i in range(B):
            n_i = sizes[i]
            semantic_out_i = semantic_out_batched[i, :n_i]  # (N_i, topo_dim)
            projected = self.semantic_bias_proj(semantic_out_i)  # (N_i, topo_dim)
            semantic_bias_i = projected @ projected.T  # (N_i, N_i)
            semantic_features_i = semantic_out_i  # (N_i, topo_dim)
            graph_embedding_i = self.graph_pool(llm_pooled[i])  # (topo_dim,)
            results.append((semantic_out_i, semantic_bias_i, semantic_features_i, graph_embedding_i))

        return results


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
        node_texts: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full TopoBridge forward pass — always runs (no gate skip).

        Args:
            node_embeddings: (N, topo_dim) from TAT/GNN output.
            semantic_weight: scalar [0,1] from ControlHead (unused here but kept for interface).
            task_text: Optional task description for the LLM/DSM.
            node_texts: Optional list of concept strings per node for KG tasks.

        Returns:
            semantic_out: (N, topo_dim) node-aligned output.
            semantic_bias: (N, N) pairwise bias for TAT attention.
            semantic_features: (N, topo_dim) per-node features (for contrastive loss).
            graph_embedding: (topo_dim,) graph-level embedding.
        """
        # Encode
        topo_memory, prefix_tokens = self.encoder(node_embeddings)

        # LLM forward (DSM backend ignores node_texts; Qwen backend uses them)
        llm_hidden = self.backend.forward(prefix_tokens, topo_memory, task_text)

        # Decode back to topo space + semantic bias + features + graph embedding
        return self.decoder(topo_memory, llm_hidden)

    def forward_batched(
        self,
        node_embeddings_list: list[torch.Tensor],
        semantic_weights: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Batched TopoBridge: process multiple graphs through DSM in one pass.

        This is the key optimization for GPU utilization. Instead of running
        B individual DSM forward passes (each on 8 prefix tokens), we batch
        them into a single pass with B*8 tokens.

        Args:
            node_embeddings_list: List of B tensors, each (N_i, topo_dim).
            semantic_weights: List of B scalar tensors (unused here, kept for API).

        Returns:
            List of (semantic_out_i, semantic_bias_i, semantic_features_i, graph_embedding_i).
        """
        # Batched encode
        topo_memory, prefix_tokens, memory_mask, sizes = \
            self.encoder.forward_batched(node_embeddings_list)

        # Batched DSM forward
        llm_hidden = self.backend.forward(
            prefix_tokens, topo_memory,
            memory_key_padding_mask=memory_mask,
        )  # (K, B, dsm_dim)

        # Batched decode
        return self.decoder.forward_batched(
            topo_memory, llm_hidden, memory_mask, sizes,
        )

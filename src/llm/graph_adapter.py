"""GraphFormer adapter: encode graph topology as LLM-readable tokens, decode back.

Encoder: GNN node embeddings -> K "graph tokens" for frozen LLM.
Decoder: LLM hidden states -> per-node semantic features + bias + graph embedding.
"""
import torch
import torch.nn as nn


class GraphFormerEncoder(nn.Module):
    """Encode graph topology as K soft tokens for LLM consumption."""

    def __init__(self, topo_dim=32, llm_dim=2048, num_tokens=16, num_layers=2,
                 num_tasks=14):
        super().__init__()
        self.input_proj = nn.Linear(topo_dim, llm_dim)
        self.graph_queries = nn.Parameter(torch.randn(num_tokens, llm_dim) * 0.02)
        self.task_embedding = nn.Embedding(num_tasks, llm_dim)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                llm_dim, nhead=8, dim_feedforward=4 * llm_dim,
                batch_first=False, norm_first=True,
            )
            for _ in range(num_layers)
        ])

    def forward(self, node_embeddings, task_id):
        memory = self.input_proj(node_embeddings)  # (N, llm_dim)
        task_emb = self.task_embedding(task_id)  # (llm_dim,)
        queries = self.graph_queries + task_emb.unsqueeze(0)  # (K, llm_dim)
        for layer in self.layers:
            queries = layer(queries, memory)
        return queries  # (K, llm_dim)


class GraphFormerDecoder(nn.Module):
    """Decode LLM hidden states back to graph-space features."""

    def __init__(self, llm_dim=2048, topo_dim=32, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                llm_dim, nhead=8, dim_feedforward=4 * llm_dim,
                batch_first=False, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.feature_proj = nn.Linear(llm_dim, topo_dim)
        self.bias_proj = nn.Linear(topo_dim, topo_dim)
        self.graph_pool = nn.Linear(llm_dim, topo_dim)

    def forward(self, node_queries, llm_hidden_states):
        for layer in self.layers:
            node_queries = layer(node_queries, llm_hidden_states)
        features = self.feature_proj(node_queries)  # (N, topo_dim)
        bias = features @ self.bias_proj(features).T  # (N, N)
        graph_emb = self.graph_pool(llm_hidden_states.mean(0))  # (topo_dim,)
        return features, bias, graph_emb

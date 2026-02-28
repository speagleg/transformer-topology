"""Approach C adapter: per-node text features via Qwen embed_tokens.

Replaces the GraphFormer encode/decode pipeline (N→16→N bottleneck) with
direct per-node embedding lookup. Text features bypass the blend and go
straight to the classifier input via extract_text_features().

The legacy forward() returns a passthrough 4-tuple for backward compat
(no blend applied).
"""
import torch
import torch.nn as nn

from src.llm.qwen_text_features import QwenTextFeatureExtractor


class QwenBridgeAdapter(nn.Module):
    """Approach C adapter: per-node text features via Qwen embed_tokens."""

    def __init__(self, llm_config: dict, topo_dim: int):
        super().__init__()
        self.text_extractor = QwenTextFeatureExtractor(
            llm_dim=llm_config.get('llm_dim', 2048),
            text_feat_dim=topo_dim,
            use_mock=llm_config.get('use_mock', False),
            qwen_model=llm_config.get('qwen_model', 'Qwen/Qwen2.5-3B-Instruct'),
            max_tokens_per_concept=llm_config.get('max_tokens_per_concept', 16),
        )

    def extract_text_features(self, node_texts, device):
        """Extract per-node text features. Returns (N, topo_dim) or None."""
        return self.text_extractor(node_texts, device)

    def forward(self, node_embeddings, semantic_weight, task_text=None, node_texts=None):
        """Legacy 4-tuple passthrough for backward compat.

        Returns (node_embeddings, zeros, node_embeddings, zeros) — no blend.
        Text features are consumed via extract_text_features() instead.
        """
        N = node_embeddings.shape[0]
        dev = node_embeddings.device
        zeros_bias = torch.zeros(N, N, device=dev)
        zeros_emb = torch.zeros(node_embeddings.shape[-1], device=dev)
        return node_embeddings, zeros_bias, node_embeddings, zeros_emb

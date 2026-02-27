"""Thin adapter wrapping QwenGraphBackend to present the TopoBridge 4-tuple interface.

Used by HierarchicalMultiHopModel when backend='qwen'. Maps task names to
integer task_ids via TASK_REGISTRY and delegates to QwenGraphBackend.forward_graph().
"""
import torch
import torch.nn as nn

from src.benchmarks.benchmark_dataset import TASK_REGISTRY
from src.llm.qwen_backend import QwenGraphBackend


class QwenBridgeAdapter(nn.Module):
    """Adapter from QwenGraphBackend 3-tuple to TopoBridge 4-tuple interface."""

    def __init__(self, llm_config: dict, topo_dim: int):
        super().__init__()
        self.backend = QwenGraphBackend({
            'topo_dim': topo_dim,
            'llm_dim': llm_config.get('llm_dim', 2048),
            'num_tokens': llm_config.get('num_tokens', 16),
            'adapter_layers': llm_config.get('adapter_layers', 2),
            'num_tasks': llm_config.get('num_tasks', 14),
            'extract_layer': llm_config.get('extract_layer', 16),
            'use_mock': llm_config.get('use_mock', False),
            'qwen_model': llm_config.get('qwen_model', 'Qwen/Qwen2.5-3B-Instruct-AWQ'),
        })
        self._task_to_id = {name: i for i, name in enumerate(TASK_REGISTRY.keys())}

    def _extract_task_name(self, task_text):
        """Extract task name from task_prompt string or bare task name."""
        if task_text is None:
            return None
        # task_prompt format: "... | task=<name>"
        if 'task=' in task_text:
            return task_text.split('task=')[-1].strip()
        return task_text

    def forward(self, node_embeddings, semantic_weight, task_text=None, node_texts=None):
        """Forward pass matching TopoBridge signature.

        Args:
            node_embeddings: (N, topo_dim) from GNN
            semantic_weight: scalar gate value (unused here, applied by caller)
            task_text: task name or full task_prompt string for task embedding lookup
            node_texts: optional list of concept strings per node for KG tasks

        Returns:
            (semantic_features, semantic_bias, semantic_features, graph_embedding)
        """
        task_name = self._extract_task_name(task_text)
        task_id = torch.tensor(
            self._task_to_id.get(task_name, 0), device=node_embeddings.device,
        )
        sem_feat, sem_bias, graph_emb = self.backend.forward_graph(
            node_embeddings, task_id, node_texts=node_texts,
        )
        return sem_feat, sem_bias, sem_feat, graph_emb

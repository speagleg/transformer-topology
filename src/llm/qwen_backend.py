"""Frozen Qwen2.5-3B backend with GraphFormer adapter.

The LLM is fully frozen and 4-bit quantized. Only the GraphFormer
encoder/decoder are trainable (~20M params).
"""
import torch
import torch.nn as nn

from src.llm.graph_adapter import GraphFormerEncoder, GraphFormerDecoder


class QwenGraphBackend(nn.Module):
    """Frozen LLM + learned adapter for graph->semantic->graph translation."""

    def __init__(self, config: dict):
        super().__init__()
        topo_dim = config.get("topo_dim", 32)
        llm_dim = config.get("llm_dim", 2048)
        num_tokens = config.get("num_tokens", 16)
        adapter_layers = config.get("adapter_layers", 2)
        num_tasks = config.get("num_tasks", 14)
        use_mock = config.get("use_mock", False)
        self.llm_dim = llm_dim
        self.extract_layer = config.get("extract_layer", 16)

        self.encoder = GraphFormerEncoder(
            topo_dim=topo_dim, llm_dim=llm_dim, num_tokens=num_tokens,
            num_layers=adapter_layers, num_tasks=num_tasks,
        )
        self.decoder = GraphFormerDecoder(
            llm_dim=llm_dim, topo_dim=topo_dim, num_layers=adapter_layers,
        )

        if use_mock:
            self.llm = nn.Sequential(
                nn.Linear(llm_dim, llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim),
            )
            self.llm.eval()
            for p in self.llm.parameters():
                p.requires_grad = False
            self._is_mock = True
        else:
            self._is_mock = False
            self.llm = None

    def _load_qwen(self):
        """Load Qwen2.5-3B-AWQ (4-bit quantized)."""
        from transformers import AutoModelForCausalLM
        model_name = "Qwen/Qwen2.5-3B-AWQ"
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cuda", torch_dtype=torch.float16,
        )
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward_graph(self, node_embeddings, task_id):
        """Full graph->LLM->graph pipeline.

        Args:
            node_embeddings: (N, topo_dim) from GNN
            task_id: scalar tensor (task index)

        Returns:
            (semantic_features, semantic_bias, graph_embedding)
        """
        graph_tokens = self.encoder(node_embeddings, task_id)

        if self._is_mock:
            hidden = self.llm(graph_tokens)
        else:
            if self.llm is None:
                self._load_qwen()
            with torch.no_grad():
                out = self.llm(
                    inputs_embeds=graph_tokens.unsqueeze(0),
                    output_hidden_states=True,
                )
                hidden = out.hidden_states[self.extract_layer].squeeze(0)

        node_proj = self.encoder.input_proj(node_embeddings)
        features, bias, graph_emb = self.decoder(node_proj, hidden)
        return features, bias, graph_emb

    def trainable_parameters(self):
        """Return only adapter parameters (not frozen LLM)."""
        yield from self.encoder.parameters()
        yield from self.decoder.parameters()

    def parameters(self, recurse=True):
        """Return trainable parameters for optimizer."""
        return self.trainable_parameters()

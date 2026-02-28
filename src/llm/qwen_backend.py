"""Qwen2.5-3B backend with GraphFormer adapter.

The LLM is mostly frozen with optional fine-tuning of the last N layers.
GraphFormer encoder/decoder are always trainable (~20M params).
"""
import torch
import torch.nn as nn

from src.llm.graph_adapter import GraphFormerEncoder, GraphFormerDecoder


class QwenGraphBackend(nn.Module):
    """LLM + learned adapter for graph->semantic->graph translation."""

    def __init__(self, config: dict):
        super().__init__()
        topo_dim = config.get("topo_dim", 32)
        llm_dim = config.get("llm_dim", 2048)
        num_tokens = config.get("num_tokens", 16)
        adapter_layers = config.get("adapter_layers", 2)
        num_tasks = config.get("num_tasks", 14)
        use_mock = config.get("use_mock", False)
        self.llm_dim = llm_dim
        self.extract_layer = config.get("extract_layer", -1)
        self.num_trainable_layers = config.get("num_trainable_layers", 0)
        self.model_name = config.get("qwen_model", "Qwen/Qwen2.5-3B-Instruct-AWQ")

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
        """Load Qwen model, optionally unfreezing last N layers."""
        from transformers import AutoModelForCausalLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="cuda", torch_dtype=torch.float16,
        )
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False
        # Selectively unfreeze last N transformer layers
        if self.num_trainable_layers > 0:
            layers = self.llm.model.layers
            for layer in layers[-self.num_trainable_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
            # Gradient checkpointing: trade compute for VRAM on unfrozen layers
            self.llm.gradient_checkpointing_enable()
            print(f"  Qwen: unfroze last {self.num_trainable_layers}/{len(layers)} layers "
                  f"(gradient checkpointing enabled)")

    def forward_graph(self, node_embeddings, task_id, node_texts=None):
        """Full graph->LLM->graph pipeline.

        Args:
            node_embeddings: (N, topo_dim) from GNN
            task_id: scalar tensor (task index)
            node_texts: optional list of concept strings for each node

        Returns:
            (semantic_features, semantic_bias, graph_embedding)
        """
        graph_tokens = self.encoder(node_embeddings, task_id)

        if self._is_mock:
            graph_hidden = self.llm(graph_tokens)
            text_hidden = None
        else:
            if self.llm is None:
                self._load_qwen()

            # Build input: graph tokens + optional text tokens (capped at 128 tokens)
            if node_texts is not None:
                text_prompt = "Graph nodes: " + ", ".join(
                    f"{i}={t}" for i, t in enumerate(node_texts)
                ) + "."
                if not hasattr(self, '_tokenizer'):
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                text_ids = self._tokenizer(
                    text_prompt, return_tensors="pt", max_length=128, truncation=True,
                ).input_ids
                text_ids = text_ids.to(node_embeddings.device)
                text_emb = self.llm.model.embed_tokens(text_ids).squeeze(0)
                combined = torch.cat([graph_tokens, text_emb], dim=0)
            else:
                combined = graph_tokens

            # Cast to Qwen's dtype (fp16) for forward pass
            combined_half = combined.to(self.llm.dtype)
            with torch.no_grad():
                out = self.llm(
                    inputs_embeds=combined_half.unsqueeze(0),
                    output_hidden_states=True,
                )
            hidden = out.hidden_states[self.extract_layer].squeeze(0)
            hidden = hidden.to(graph_tokens.dtype)
            # Split into graph and text hidden states
            num_graph = graph_tokens.shape[0]
            graph_hidden = hidden[:num_graph]
            text_hidden = hidden[num_graph:] if hidden.shape[0] > num_graph else None

        node_proj = self.encoder.input_proj(node_embeddings)
        features, bias, graph_emb = self.decoder(
            node_proj, graph_hidden, text_hidden_states=text_hidden,
        )
        return features, bias, graph_emb

    def trainable_parameters(self):
        """Return adapter parameters + any unfrozen LLM layers."""
        yield from self.encoder.parameters()
        yield from self.decoder.parameters()

    def llm_trainable_parameters(self):
        """Return only unfrozen LLM layer parameters (for separate optimizer group)."""
        if self.llm is not None and self.num_trainable_layers > 0:
            layers = self.llm.model.layers
            for layer in layers[-self.num_trainable_layers:]:
                yield from layer.parameters()

    def parameters(self, recurse=True):
        """Return trainable parameters for optimizer."""
        yield from self.trainable_parameters()
        yield from self.llm_trainable_parameters()

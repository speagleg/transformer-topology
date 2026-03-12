"""Qwen 2.5 contextual encoder: frozen Qwen model (truncated) + learned projection.

Caches raw LLM-dim vectors per concept. Projection applied at forward time for gradient flow.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn


class QwenContextualEncoder(nn.Module):
    """Encodes concept strings via frozen Qwen layers + learned projection.

    In mock mode, uses deterministic hash-based embeddings for testing.
    In real mode, loads truncated Qwen model (first N layers, frozen)
    and uses its own forward method for compatibility across transformers versions.
    """

    def __init__(
        self,
        llm_dim: int = 2048,
        output_dim: int = 32,
        use_mock: bool = False,
        qwen_model: str = "Qwen/Qwen2.5-3B-Instruct",
        max_tokens_per_concept: int = 16,
        num_layers: int = 4,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.output_dim = output_dim
        self.use_mock = use_mock
        self.max_tokens = max_tokens_per_concept
        self.num_layers = num_layers

        # Learned projection (gradients flow here)
        self.proj = nn.Linear(llm_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

        # Caches: raw llm_dim vectors (before projection)
        self._cpu_cache: dict[str, torch.Tensor] = {}
        self._gpu_cache: torch.Tensor | None = None
        self._gpu_index: dict[str, int] = {}

        # Load real Qwen components if not mock
        self._tokenizer = None
        self._qwen_model = None  # truncated Qwen model (not LM head)
        if not use_mock:
            self._load_qwen(qwen_model)

    def _load_qwen(self, model_name: str) -> None:
        """Load frozen, truncated Qwen model (first N layers only)."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError("transformers required for real Qwen mode")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True,
        )

        # Keep only the base model (not LM head), truncate layers
        self._qwen_model = full_model.model
        # Remove layers beyond num_layers to save VRAM
        self._qwen_model.layers = self._qwen_model.layers[:self.num_layers]

        # Freeze everything
        for p in self._qwen_model.parameters():
            p.requires_grad = False

        # Free the LM head and reference
        del full_model

    def _mock_embed(self, text: str) -> torch.Tensor:
        """Deterministic hash-based embedding for testing."""
        h = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        gen = torch.Generator().manual_seed(h)
        return torch.randn(self.llm_dim, generator=gen)

    @torch.no_grad()
    def _encode_texts(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Encode texts to raw LLM-dim vectors (no projection). Uses cache."""
        results: list[tuple[int, torch.Tensor | None]] = []
        uncached = []
        uncached_idx = []

        for i, t in enumerate(texts):
            if t in self._cpu_cache:
                results.append((i, self._cpu_cache[t]))
            else:
                uncached.append(t)
                uncached_idx.append(i)
                results.append((i, None))

        if uncached:
            if self.use_mock:
                for text, idx in zip(uncached, uncached_idx):
                    vec = self._mock_embed(text)
                    self._cpu_cache[text] = vec
                    results[idx] = (idx, vec)
            else:
                vecs = self._qwen_encode(uncached, device)
                for text, idx, vec in zip(uncached, uncached_idx, vecs):
                    cpu_vec = vec.cpu().float()
                    self._cpu_cache[text] = cpu_vec
                    results[idx] = (idx, cpu_vec)

        results.sort(key=lambda x: x[0])
        return torch.stack([r[1] for r in results]).to(device)

    @torch.no_grad()
    def _qwen_encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Run texts through frozen truncated Qwen model."""
        tokens = self._tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_tokens,
        ).to(device)

        # Use the model's own forward — handles rotary embeddings, masks, etc.
        out = self._qwen_model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
        )
        hidden = out.last_hidden_state  # (batch, seq, llm_dim)

        # Mean pool over non-padding tokens
        mask = tokens.attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.float()

    def forward(
        self, texts: list[str] | None, device: torch.device,
    ) -> torch.Tensor | None:
        """Encode concept texts to (N, output_dim) with gradient through projection.

        Args:
            texts: List of concept strings, one per node. None returns None.
            device: Target device.

        Returns:
            (N, output_dim) tensor with gradients through proj/norm, or None.
        """
        if texts is None or len(texts) == 0:
            return None

        # Get raw LLM-dim vectors (cached, no grad)
        raw = self._encode_texts(texts, device)  # (N, llm_dim)

        # Project with gradients
        return self.norm(self.proj(raw))  # (N, output_dim)

    def precompute(self, concepts: list[str], device: torch.device) -> int:
        """Pre-compute and cache raw embeddings for a list of concepts."""
        new_concepts = [c for c in concepts if c not in self._cpu_cache]
        if not new_concepts:
            return 0
        self._encode_texts(new_concepts, device)
        return len(new_concepts)

    def save_cache(self, path: str) -> None:
        torch.save(self._cpu_cache, path)

    def load_cache(self, path: str) -> int:
        data = torch.load(path, map_location="cpu", weights_only=True)
        self._cpu_cache.update(data)
        return len(data)

    def build_gpu_cache(self, device: torch.device) -> None:
        """Build GPU tensor for fast batch lookup."""
        if not self._cpu_cache:
            return
        concepts = sorted(self._cpu_cache.keys())
        self._gpu_index = {c: i for i, c in enumerate(concepts)}
        self._gpu_cache = torch.stack(
            [self._cpu_cache[c] for c in concepts]
        ).to(device)

    def clear_cache(self) -> None:
        self._cpu_cache.clear()
        self._gpu_cache = None
        self._gpu_index.clear()

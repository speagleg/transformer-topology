"""Per-node text feature extractor using Qwen embed_tokens.

Approach C: Instead of running full Qwen forward passes through GraphFormer
encode/decode (catastrophic N→16→N bottleneck), we use Qwen's embedding table
as a frozen per-node feature extractor. Each concept string gets tokenized,
looked up in embed_tokens, mean-pooled, and projected to topo_dim.

VRAM: ~0.6GB (embed_tokens only) vs ~6GB (full Qwen model).
"""

import torch
import torch.nn as nn


class QwenTextFeatureExtractor(nn.Module):
    """Extract per-node text features from concept strings via Qwen embed_tokens.

    For each concept string:
      1. Tokenize (up to max_tokens_per_concept subwords)
      2. Look up in embed_tokens (frozen) → (num_tokens, llm_dim)
      3. Mean-pool over tokens → (llm_dim,)
      4. Project via learned Linear + LayerNorm → (text_feat_dim,)

    Raw embeddings are cached by concept string (frozen = deterministic).
    """

    def __init__(self, llm_dim=2048, text_feat_dim=32, use_mock=False,
                 qwen_model='Qwen/Qwen2.5-3B-Instruct',
                 max_tokens_per_concept=16):
        super().__init__()
        self.llm_dim = llm_dim
        self.text_feat_dim = text_feat_dim
        self.use_mock = use_mock
        self.max_tokens = max_tokens_per_concept

        # Learned projection (has gradients)
        self.proj = nn.Sequential(
            nn.Linear(llm_dim, text_feat_dim),
            nn.LayerNorm(text_feat_dim),
        )

        # Embedding table (frozen, loaded lazily)
        self._embed_tokens = None
        self._tokenizer = None
        self._qwen_model_name = qwen_model

        if use_mock:
            self._embed_tokens = nn.Embedding(10000, llm_dim)
            self._embed_tokens.requires_grad_(False)

        # Cache: concept string → (llm_dim,) tensor on CPU
        self._cache: dict[str, torch.Tensor] = {}

    def _load_embed_tokens(self):
        """Lazy-load only embed_tokens from Qwen (saves ~5.4GB vs full model)."""
        if self._embed_tokens is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"  Loading embed_tokens from {self._qwen_model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            self._qwen_model_name, torch_dtype=torch.float16,
        )
        self._embed_tokens = model.model.embed_tokens
        self._embed_tokens.requires_grad_(False)
        self._tokenizer = AutoTokenizer.from_pretrained(self._qwen_model_name)
        # Free the rest of the model
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  embed_tokens loaded: {sum(p.numel() for p in self._embed_tokens.parameters()):,} params")

    def _get_tokenizer(self):
        """Get or lazy-load tokenizer."""
        if self._tokenizer is not None:
            return self._tokenizer
        if self.use_mock:
            return None
        self._load_embed_tokens()
        return self._tokenizer

    def _get_raw_embedding(self, concept: str, device: torch.device) -> torch.Tensor:
        """Get (llm_dim,) raw embedding for a concept, using cache."""
        if concept in self._cache:
            return self._cache[concept].to(device)

        if self.use_mock:
            # Hash-based deterministic index
            idx = hash(concept) % self._embed_tokens.num_embeddings
            with torch.no_grad():
                raw = self._embed_tokens(torch.tensor([idx])).squeeze(0)
        else:
            self._load_embed_tokens()
            tokenizer = self._get_tokenizer()
            tokens = tokenizer.encode(concept, add_special_tokens=False)
            tokens = tokens[:self.max_tokens]
            if not tokens:
                tokens = [0]  # fallback to padding token
            token_ids = torch.tensor(tokens, device=device)
            embed_device = next(self._embed_tokens.parameters()).device
            with torch.no_grad():
                token_embs = self._embed_tokens(token_ids.to(embed_device))
                raw = token_embs.mean(dim=0).to(device)

        # Cache on CPU to save GPU memory
        self._cache[concept] = raw.detach().cpu()
        return raw.to(device)

    def forward(self, node_texts: list[str] | None, device: torch.device) -> torch.Tensor | None:
        """Extract text features for all nodes.

        Args:
            node_texts: List of concept strings, one per node. None → returns None.
            device: Target device for output tensor.

        Returns:
            (N, text_feat_dim) tensor or None if no texts provided.
        """
        if node_texts is None or len(node_texts) == 0:
            return None

        # Get raw embeddings for each concept (frozen, cached)
        raw_embs = []
        for text in node_texts:
            raw = self._get_raw_embedding(text, device)
            raw_embs.append(raw)

        # Stack and project (proj has gradients)
        stacked = torch.stack(raw_embs)  # (N, llm_dim)
        return self.proj(stacked.to(device))  # (N, text_feat_dim)

    def clear_cache(self):
        """Clear the embedding cache (e.g., between tasks)."""
        self._cache.clear()

# v10 Dual-Track Fusion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the identical-iteration executive loop with an asymmetric dual-track loop (structural → cross-modal fusion) and redesign KG tasks to require genuine structure+semantics fusion.

**Architecture:** Iteration 1 does pure GNN+wave+TAT structural encoding. Iteration 2 adds text-conditioned GNN messages + cross-attention to Qwen contextual embeddings + TAT. A learned `fusion_weight` gates how much iteration 2 overrides iteration 1. An attention-based readout replaces fixed query/target extraction.

**Tech Stack:** PyTorch, PyG (torch_geometric), transformers (Qwen 2.5-3B frozen layers), NetworkX, ConceptNet

**Spec:** `docs/plans/2026-03-11-v10-dual-track-fusion-design.md`

---

## File Map

### New Files
| File | Responsibility |
|------|---------------|
| `src/llm/qwen_contextual_encoder.py` | 4-layer frozen Qwen encoder + learned projection + caching |
| `src/reasoning_loop/cross_attention.py` | Multi-head cross-attention block for iteration 2 |
| `src/reasoning_loop/attention_readout.py` | Task-conditioned attention readout over all nodes |
| `src/reasoning_loop/text_conditioned_gnn.py` | Text-similarity-modulated message passing layer |
| `config/v10_dual_track.yaml` | Training config for v10 |
| `tests/test_llm/test_qwen_contextual_encoder.py` | Tests for Qwen encoder |
| `tests/test_reasoning_loop/test_cross_attention.py` | Tests for cross-attention block |
| `tests/test_reasoning_loop/test_attention_readout.py` | Tests for attention readout |
| `tests/test_reasoning_loop/test_text_conditioned_gnn.py` | Tests for text-conditioned GNN |
| `tests/test_benchmarks/test_v10_tasks.py` | Tests for new KG task generators |
| `tests/test_integration/test_v10_forward.py` | End-to-end forward pass test |

### Modified Files
| File | Changes |
|------|---------|
| `src/gnn_executive/control_head.py` | Add `fusion_weight` head, remove metacog fields |
| `src/reasoning_loop/executive_loop.py` | Dual-track iteration logic, remove .detach() between iters |
| `src/training/batch_utils.py` | New classifier input assembly with attention readout |
| `src/benchmarks/conceptnet_tasks.py` | New task generators: transitive, consistency, causal_chain; redesign analogy |
| `src/benchmarks/benchmark_dataset.py` | Update TASK_REGISTRY with new tasks |
| `src/computation_graph/diagnostics.py` | Add fusion_weight tracking |
| `scripts/run_dsm_curriculum.py` | Phase B training loop, optimizer groups |

### Untouched (but referenced)
| File | Why |
|------|-----|
| `src/gnn_executive/spatial.py` | SpatialGNN unchanged; text-conditioned GNN is a NEW layer used only in iter 2 |
| `src/gnn_executive/executive.py` | GNNExecutive forward unchanged; iter 2 uses separate text-conditioned path |
| `src/wave/` | Wave dynamics unchanged, only runs in iteration 1 |
| `src/tat/` | TAT unchanged, called in both iterations with different inputs |

---

## Chunk 1: New Modules (Tasks 1-4)

### Task 1: Qwen Contextual Encoder

**Files:**
- Create: `src/llm/qwen_contextual_encoder.py`
- Test: `tests/test_llm/test_qwen_contextual_encoder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm/test_qwen_contextual_encoder.py
import pytest
import torch
from src.llm.qwen_contextual_encoder import QwenContextualEncoder


class TestQwenContextualEncoder:
    """Tests for QwenContextualEncoder with mock mode."""

    def test_init_mock(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        assert enc.output_dim == 32
        assert enc.llm_dim == 64

    def test_forward_returns_correct_shape(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        texts = ["dog", "cat", "animal"]
        device = torch.device("cpu")
        result = enc(texts, device)
        assert result.shape == (3, 32)

    def test_forward_none_texts_returns_none(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        result = enc(None, torch.device("cpu"))
        assert result is None

    def test_forward_empty_texts_returns_none(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        result = enc([], torch.device("cpu"))
        assert result is None

    def test_projection_has_gradients(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        texts = ["dog", "cat"]
        result = enc(texts, torch.device("cpu"))
        loss = result.sum()
        loss.backward()
        assert enc.proj.weight.grad is not None

    def test_cache_precompute_and_lookup(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        device = torch.device("cpu")
        # First call computes
        r1 = enc(["dog", "cat"], device)
        # Second call uses cache
        r2 = enc(["dog", "cat"], device)
        # Raw cache values same (projection may differ due to grad state)
        assert "dog" in enc._cpu_cache
        assert "cat" in enc._cpu_cache

    def test_cache_stores_raw_llm_dim(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        enc(["dog"], torch.device("cpu"))
        assert enc._cpu_cache["dog"].shape == (64,)

    def test_save_load_cache(self, tmp_path):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        enc(["dog", "cat"], torch.device("cpu"))
        path = tmp_path / "cache.pt"
        enc.save_cache(str(path))

        enc2 = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        loaded = enc2.load_cache(str(path))
        assert loaded == 2
        assert "dog" in enc2._cpu_cache

    def test_build_gpu_cache(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        enc(["dog", "cat", "fish"], torch.device("cpu"))
        enc.build_gpu_cache(torch.device("cpu"))
        assert enc._gpu_cache is not None
        assert enc._gpu_cache.shape[0] == 3  # 3 concepts

    def test_different_texts_different_embeddings(self):
        enc = QwenContextualEncoder(llm_dim=64, output_dim=32, use_mock=True)
        r1 = enc(["dog"], torch.device("cpu"))
        r2 = enc(["democracy"], torch.device("cpu"))
        # Mock uses hash-based deterministic random, so different texts -> different embeddings
        assert not torch.allclose(r1, r2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_llm/test_qwen_contextual_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement QwenContextualEncoder**

```python
# src/llm/qwen_contextual_encoder.py
"""Qwen 2.5 contextual encoder: frozen embed_tokens + 4 transformer layers + learned projection.

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
    In real mode, loads Qwen embed_tokens + first N transformer layers (frozen).
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
        self._embed_tokens = None
        self._layers = None
        if not use_mock:
            self._load_qwen(qwen_model)

    def _load_qwen(self, model_name: str) -> None:
        """Load frozen Qwen embed_tokens + first N transformer layers."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError("transformers required for real Qwen mode")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True,
        )
        self._embed_tokens = model.model.embed_tokens
        self._layers = nn.ModuleList(list(model.model.layers[:self.num_layers]))
        self._layer_norm = model.model.norm  # final RMSNorm

        # Freeze all Qwen parameters
        for p in self._embed_tokens.parameters():
            p.requires_grad = False
        for layer in self._layers:
            for p in layer.parameters():
                p.requires_grad = False
        for p in self._layer_norm.parameters():
            p.requires_grad = False

        # Free the rest of the model
        del model

    def _mock_embed(self, text: str) -> torch.Tensor:
        """Deterministic hash-based embedding for testing."""
        h = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        gen = torch.Generator().manual_seed(h)
        return torch.randn(self.llm_dim, generator=gen)

    @torch.no_grad()
    def _encode_texts(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Encode texts to raw LLM-dim vectors (no projection). Uses cache."""
        results = []
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
        """Run texts through frozen Qwen layers."""
        tokens = self._tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_tokens,
        ).to(device)

        hidden = self._embed_tokens(tokens.input_ids)
        mask = tokens.attention_mask

        for layer in self._layers:
            out = layer(hidden, attention_mask=mask)
            hidden = out[0]

        hidden = self._layer_norm(hidden)

        # Mean pool over non-padding tokens
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_llm/test_qwen_contextual_encoder.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/qwen_contextual_encoder.py tests/test_llm/test_qwen_contextual_encoder.py
git commit -m "feat: QwenContextualEncoder with frozen 4-layer Qwen + learned projection"
```

---

### Task 2: Cross-Attention Block

**Files:**
- Create: `src/reasoning_loop/cross_attention.py`
- Test: `tests/test_reasoning_loop/test_cross_attention.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reasoning_loop/test_cross_attention.py
import pytest
import torch
from src.reasoning_loop.cross_attention import CrossAttentionBlock


class TestCrossAttentionBlock:

    def test_init(self):
        block = CrossAttentionBlock(embed_dim=32, num_heads=4)
        assert block.embed_dim == 32

    def test_forward_shape(self):
        block = CrossAttentionBlock(embed_dim=32, num_heads=4)
        h_struct = torch.randn(10, 32)  # N=10 nodes
        h_text = torch.randn(10, 32)
        out = block(h_struct, h_text)
        assert out.shape == (10, 32)

    def test_forward_different_n(self):
        """Text can have different N if nodes were filtered."""
        block = CrossAttentionBlock(embed_dim=32, num_heads=4)
        h_struct = torch.randn(10, 32)
        h_text = torch.randn(8, 32)  # fewer text nodes
        out = block(h_struct, h_text)
        assert out.shape == (10, 32)

    def test_residual_connection(self):
        """Output should be close to input when attention weights are small."""
        block = CrossAttentionBlock(embed_dim=32, num_heads=4)
        # Zero out attention weights
        with torch.no_grad():
            for p in block.cross_attn.parameters():
                p.zero_()
        h_struct = torch.randn(5, 32)
        h_text = torch.randn(5, 32)
        out = block(h_struct, h_text)
        # With zeroed attn, output ≈ LayerNorm(h_struct + 0) ≈ LayerNorm(h_struct)
        expected = block.norm(h_struct)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_gradients_flow_to_both_inputs(self):
        block = CrossAttentionBlock(embed_dim=32, num_heads=4)
        h_struct = torch.randn(5, 32, requires_grad=True)
        h_text = torch.randn(5, 32, requires_grad=True)
        out = block(h_struct, h_text)
        out.sum().backward()
        assert h_struct.grad is not None
        assert h_text.grad is not None

    def test_gradients_flow_to_params(self):
        block = CrossAttentionBlock(embed_dim=32, num_heads=4)
        h_struct = torch.randn(5, 32)
        h_text = torch.randn(5, 32)
        out = block(h_struct, h_text)
        out.sum().backward()
        has_grad = any(p.grad is not None for p in block.parameters())
        assert has_grad
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_reasoning_loop/test_cross_attention.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement CrossAttentionBlock**

```python
# src/reasoning_loop/cross_attention.py
"""Cross-attention block for structure→text fusion in iteration 2."""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention: structural embeddings attend to text embeddings.

    h_fused = LayerNorm(h_struct + MultiHeadAttn(Q=h_struct, K=h_text, V=h_text))
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h_struct: torch.Tensor,   # (N, embed_dim) structural embeddings
        h_text: torch.Tensor,     # (M, embed_dim) text embeddings
    ) -> torch.Tensor:            # (N, embed_dim) fused embeddings
        """Cross-attend structural queries to text keys/values."""
        # MultiheadAttention expects (batch, seq, dim) with batch_first=True
        # We treat this as batch=1, seq=N or M
        q = h_struct.unsqueeze(0)  # (1, N, d)
        k = h_text.unsqueeze(0)    # (1, M, d)
        v = h_text.unsqueeze(0)    # (1, M, d)

        attn_out, _ = self.cross_attn(q, k, v)  # (1, N, d)
        attn_out = attn_out.squeeze(0)  # (N, d)

        return self.norm(h_struct + attn_out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_reasoning_loop/test_cross_attention.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/reasoning_loop/cross_attention.py tests/test_reasoning_loop/test_cross_attention.py
git commit -m "feat: CrossAttentionBlock for structure→text fusion"
```

---

### Task 3: Attention-Based Readout

**Files:**
- Create: `src/reasoning_loop/attention_readout.py`
- Test: `tests/test_reasoning_loop/test_attention_readout.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reasoning_loop/test_attention_readout.py
import pytest
import torch
from src.reasoning_loop.attention_readout import AttentionReadout


class TestAttentionReadout:

    def test_init(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        assert readout.embed_dim == 32

    def test_forward_shape(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        query_idx = 0
        target_idx = 3
        task_id = 5
        context = readout(h_out, query_idx, target_idx, task_id)
        assert context.shape == (32,)

    def test_forward_without_task_id(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        context = readout(h_out, 0, 3, task_id=None)
        assert context.shape == (32,)

    def test_different_tasks_different_queries(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        c1 = readout(h_out, 0, 3, task_id=0)
        c2 = readout(h_out, 0, 3, task_id=1)
        assert not torch.allclose(c1, c2)

    def test_gradients_flow(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32, requires_grad=True)
        context = readout(h_out, 0, 3, task_id=5)
        context.sum().backward()
        assert h_out.grad is not None

    def test_attention_focuses_on_query_target_region(self):
        """Readout should attend more to query/target than distant nodes."""
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        # Make query/target embeddings distinctive
        h_out = torch.zeros(10, 32)
        h_out[2] = torch.ones(32)  # query
        h_out[5] = torch.ones(32) * 2  # target
        context = readout(h_out, 2, 5, task_id=0)
        # Context should be nonzero (influenced by query/target)
        assert context.abs().sum() > 0

    def test_classifier_input_shape(self):
        """Test the full classifier input assembly."""
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        topo_features = torch.randn(4)
        fusion_weight = torch.tensor(0.5)

        combined = readout.build_classifier_input(
            h_out, 0, 3, task_id=5,
            topo_features=topo_features,
            fusion_weight=fusion_weight,
        )
        # Expected: h_out[query](32) + h_out[target](32) + context(32) + topo(4) + fusion(1) = 101
        assert combined.shape == (101,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_reasoning_loop/test_attention_readout.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement AttentionReadout**

```python
# src/reasoning_loop/attention_readout.py
"""Task-conditioned attention readout over all node embeddings."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class AttentionReadout(nn.Module):
    """Learned attention readout conditioned on task + query/target nodes.

    Produces a global context vector by attending over all node embeddings,
    using a query constructed from the task embedding + local node embeddings.
    """

    def __init__(self, embed_dim: int, num_tasks: int = 19):
        super().__init__()
        self.embed_dim = embed_dim
        self.task_embedding = nn.Embedding(num_tasks, embed_dim)
        # Query projection: [task_emb, h_query, h_target] -> readout_query
        self.query_proj = nn.Linear(3 * embed_dim, embed_dim)
        self._scale = math.sqrt(embed_dim)

    def forward(
        self,
        h_out: torch.Tensor,     # (N, embed_dim) all node embeddings
        query_idx: int,
        target_idx: int,
        task_id: int | None = None,
    ) -> torch.Tensor:           # (embed_dim,) global context vector
        """Compute attention-pooled context over all nodes."""
        device = h_out.device

        if task_id is not None:
            task_emb = self.task_embedding(
                torch.tensor(task_id, device=device)
            )  # (embed_dim,)
        else:
            task_emb = torch.zeros(self.embed_dim, device=device)

        h_query = h_out[query_idx]   # (embed_dim,)
        h_target = h_out[target_idx] # (embed_dim,)

        # Build readout query
        q_input = torch.cat([task_emb, h_query, h_target])  # (3*embed_dim,)
        readout_q = self.query_proj(q_input)  # (embed_dim,)

        # Attention over all nodes
        scores = (h_out @ readout_q) / self._scale  # (N,)
        weights = torch.softmax(scores, dim=0)       # (N,)
        context = (weights.unsqueeze(-1) * h_out).sum(dim=0)  # (embed_dim,)

        return context

    def build_classifier_input(
        self,
        h_out: torch.Tensor,
        query_idx: int,
        target_idx: int,
        task_id: int | None = None,
        topo_features: torch.Tensor | None = None,
        fusion_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble full classifier input vector.

        Returns: (101,) tensor = h_query(32) + h_target(32) + context(32) + topo(4) + fusion(1)
        """
        context = self(h_out, query_idx, target_idx, task_id)

        parts = [
            h_out[query_idx],   # (embed_dim,)
            h_out[target_idx],  # (embed_dim,)
            context,            # (embed_dim,)
        ]

        if topo_features is not None:
            parts.append(topo_features)

        if fusion_weight is not None:
            parts.append(fusion_weight.unsqueeze(0) if fusion_weight.dim() == 0 else fusion_weight)

        return torch.cat(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_reasoning_loop/test_attention_readout.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/reasoning_loop/attention_readout.py tests/test_reasoning_loop/test_attention_readout.py
git commit -m "feat: AttentionReadout with task-conditioned attention over all nodes"
```

---

### Task 4: Text-Conditioned GNN Layer

**Files:**
- Create: `src/reasoning_loop/text_conditioned_gnn.py`
- Test: `tests/test_reasoning_loop/test_text_conditioned_gnn.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reasoning_loop/test_text_conditioned_gnn.py
import pytest
import torch
from src.reasoning_loop.text_conditioned_gnn import TextConditionedGNN


class TestTextConditionedGNN:

    def test_init(self):
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        assert gnn.embed_dim == 32

    def test_forward_shape(self):
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0,1,2,3], [1,2,3,4]])
        text_embs = torch.randn(10, 32)
        out = gnn(x, edge_index, text_embs)
        assert out.shape == (10, 32)

    def test_forward_without_text(self):
        """Should work without text (no modulation)."""
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0,1,2,3], [1,2,3,4]])
        out = gnn(x, edge_index, text_embs=None)
        assert out.shape == (10, 32)

    def test_text_modulation_changes_output(self):
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0,1,2,3], [1,2,3,4]])
        text_embs = torch.randn(10, 32)
        out_with = gnn(x, edge_index, text_embs)
        out_without = gnn(x, edge_index, text_embs=None)
        assert not torch.allclose(out_with, out_without)

    def test_gradients_flow_through_text(self):
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0,1,2,3], [1,2,3,4]])
        text_embs = torch.randn(10, 32, requires_grad=True)
        out = gnn(x, edge_index, text_embs)
        out.sum().backward()
        assert text_embs.grad is not None

    def test_gradients_flow_through_x(self):
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(10, 32, requires_grad=True)
        edge_index = torch.tensor([[0,1,2,3], [1,2,3,4]])
        text_embs = torch.randn(10, 32)
        out = gnn(x, edge_index, text_embs)
        out.sum().backward()
        assert x.grad is not None

    def test_single_node_graph(self):
        gnn = TextConditionedGNN(embed_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(1, 32)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        text_embs = torch.randn(1, 32)
        out = gnn(x, edge_index, text_embs)
        assert out.shape == (1, 32)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_reasoning_loop/test_text_conditioned_gnn.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement TextConditionedGNN**

```python
# src/reasoning_loop/text_conditioned_gnn.py
"""Text-conditioned GNN for iteration 2: edge messages modulated by text similarity."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class TextModulatedMessagePassing(MessagePassing):
    """Message passing where edge messages are amplified by source/target text similarity."""

    def __init__(self, in_dim: int, out_dim: int, text_dim: int):
        super().__init__(aggr='add')
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ELU(),
        )
        # Text similarity: [text_src, text_dst, text_src * text_dst] -> scalar
        self.text_sim = nn.Sequential(
            nn.Linear(3 * text_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,          # (N, in_dim)
        edge_index: torch.Tensor,  # (2, E)
        text_embs: torch.Tensor | None = None,  # (N, text_dim)
    ) -> torch.Tensor:
        if text_embs is not None:
            self._text_embs = text_embs
            self._use_text = True
        else:
            self._use_text = False
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor,
                edge_index_i: torch.Tensor, edge_index_j: torch.Tensor) -> torch.Tensor:
        msg = self.msg_mlp(x_j)

        if self._use_text:
            t_src = self._text_embs[edge_index_j]  # (E, text_dim)
            t_dst = self._text_embs[edge_index_i]  # (E, text_dim)
            sim_input = torch.cat([t_src, t_dst, t_src * t_dst], dim=-1)
            sim = self.text_sim(sim_input)  # (E, 1)
            msg = msg * (1 + sim)

        return msg


class TextConditionedGNN(nn.Module):
    """Multi-layer GNN with text-modulated message passing.

    Used only in iteration 2 of the dual-track executive loop.
    Uses hidden_dim == embed_dim to enable residual connections on every layer.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        # Force hidden_dim == embed_dim for residual connections
        hidden_dim = embed_dim
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(
                TextModulatedMessagePassing(embed_dim, embed_dim, embed_dim)
            )

        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,           # (N, embed_dim)
        edge_index: torch.Tensor,   # (2, E)
        text_embs: torch.Tensor | None = None,  # (N, embed_dim)
    ) -> torch.Tensor:              # (N, embed_dim)
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, edge_index, text_embs)
            h_new = norm(h_new)
            if h.shape == h_new.shape:
                h_new = h_new + h  # residual
            h = h_new
        return h
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_reasoning_loop/test_text_conditioned_gnn.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/reasoning_loop/text_conditioned_gnn.py tests/test_reasoning_loop/test_text_conditioned_gnn.py
git commit -m "feat: TextConditionedGNN with text-similarity edge modulation"
```

---

## Chunk 2: Modified Core Modules (Tasks 5-7)

### Task 5: ControlHead — Add fusion_weight

**Files:**
- Modify: `src/gnn_executive/control_head.py`
- Test: `tests/test_gnn_executive/test_control_head.py` (add tests)

- [ ] **Step 1: Write failing tests**

Add to existing test file:

```python
# Append to tests/test_gnn_executive/test_control_head.py

class TestControlHeadFusionWeight:

    def test_control_signal_has_fusion_weight(self):
        from src.gnn_executive.control_head import ControlHead
        head = ControlHead(embedding_dim=32, num_freqs=16, num_filters=4)
        x = torch.randn(10, 32)
        signal = head(x)
        assert hasattr(signal, 'fusion_weight')

    def test_fusion_weight_is_sigmoid(self):
        from src.gnn_executive.control_head import ControlHead
        head = ControlHead(embedding_dim=32, num_freqs=16, num_filters=4)
        x = torch.randn(10, 32)
        signal = head(x)
        assert signal.fusion_weight is not None
        assert 0 <= signal.fusion_weight.item() <= 1

    def test_fusion_weight_init_near_half(self):
        """fusion_weight bias=0 → sigmoid(0)=0.5"""
        from src.gnn_executive.control_head import ControlHead
        head = ControlHead(embedding_dim=32, num_freqs=16, num_filters=4)
        x = torch.randn(10, 32)
        signal = head(x)
        assert abs(signal.fusion_weight.item() - 0.5) < 0.15

    def test_fusion_weight_gradient_flows(self):
        from src.gnn_executive.control_head import ControlHead
        head = ControlHead(embedding_dim=32, num_freqs=16, num_filters=4)
        x = torch.randn(10, 32)
        signal = head(x)
        signal.fusion_weight.backward()
        assert head.fusion_weight_head.weight.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_gnn_executive/test_control_head.py::TestControlHeadFusionWeight -v`
Expected: FAIL

- [ ] **Step 3: Add fusion_weight to ControlHead and ControlSignal**

Modify `src/gnn_executive/control_head.py`:

1. Add `fusion_weight: torch.Tensor = None` field to `ControlSignal` dataclass
2. Add `self.fusion_weight_head = nn.Linear(trunk_out_dim, 1)` in `ControlHead.__init__`
3. Initialize with `nn.init.zeros_(self.fusion_weight_head.bias)` (sigmoid(0)=0.5)
4. In `ControlHead.forward()`, compute `fusion_weight = torch.sigmoid(self.fusion_weight_head(trunk_out).squeeze(-1))` and set on ControlSignal

- [ ] **Step 4: Run ALL control_head tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_gnn_executive/test_control_head.py -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add src/gnn_executive/control_head.py tests/test_gnn_executive/test_control_head.py
git commit -m "feat: add fusion_weight to ControlHead/ControlSignal"
```

---

### Task 6: Dual-Track Executive Loop

**Files:**
- Modify: `src/reasoning_loop/executive_loop.py`
- Test: `tests/test_reasoning_loop/test_executive_loop.py` (add tests)

This is the most complex change. The ExecutiveReasoningLoop needs to:
1. Accept text embeddings as input
2. Run iteration 1 (structural): existing GNN + wave + TAT
3. Run iteration 2 (fusion): TextConditionedGNN + CrossAttention + TAT
4. Blend with fusion_weight
5. Remove `.detach()` between iterations

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_reasoning_loop/test_executive_loop.py
# or create tests/test_reasoning_loop/test_dual_track.py

import torch
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop


class TestDualTrackLoop:

    def _make_cc(self, n=10, embed_dim=32):
        """Create a simple CellComplex for testing."""
        cc = CellComplex()
        for i in range(n):
            cc.add_cell(0, (i,), embedding=torch.randn(embed_dim))
        edges = [(i, i+1) for i in range(n-1)]
        for u, v in edges:
            cc.add_cell(1, (u, v), embedding=torch.randn(embed_dim))
        cc.node_texts = [f"concept_{i}" for i in range(n)]
        return cc

    def test_forward_with_text_embeddings(self):
        loop = ExecutiveReasoningLoop(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
            gnn_spectral_layers=0, max_freqs=8, tat_layers=1,
            tat_spatial_heads=4, tat_spectral_heads=0, tat_ff_dim=64,
            max_iterations=2, use_dual_track=True,
        )
        cc = self._make_cc()
        text_embs = torch.randn(10, 32)
        out, iters, diag = loop(cc, text_embeddings=text_embs)
        assert out.shape == (10, 32)
        assert iters == 2  # always 2 in dual-track mode

    def test_forward_without_text_falls_back(self):
        loop = ExecutiveReasoningLoop(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
            gnn_spectral_layers=0, max_freqs=8, tat_layers=1,
            tat_spatial_heads=4, tat_spectral_heads=0, tat_ff_dim=64,
            max_iterations=2, use_dual_track=True,
        )
        cc = self._make_cc()
        out, iters, diag = loop(cc, text_embeddings=None)
        assert out.shape == (10, 32)

    def test_diagnostics_include_fusion_weight(self):
        loop = ExecutiveReasoningLoop(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
            gnn_spectral_layers=0, max_freqs=8, tat_layers=1,
            tat_spatial_heads=4, tat_spectral_heads=0, tat_ff_dim=64,
            max_iterations=2, use_dual_track=True,
        )
        cc = self._make_cc()
        text_embs = torch.randn(10, 32)
        _, _, diag = loop(cc, text_embeddings=text_embs)
        assert 'fusion_weight' in diag

    def test_fusion_weight_zero_gives_structural_only(self):
        """When fusion_weight=0, output should equal iteration 1 output."""
        loop = ExecutiveReasoningLoop(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
            gnn_spectral_layers=0, max_freqs=8, tat_layers=1,
            tat_spatial_heads=4, tat_spectral_heads=0, tat_ff_dim=64,
            max_iterations=2, use_dual_track=True,
        )
        cc = self._make_cc()
        text_embs = torch.randn(10, 32)
        # Force fusion_weight to 0
        out, _, _ = loop(cc, text_embeddings=text_embs, force_fusion_weight=0.0)
        out_no_text, _, _ = loop(cc, text_embeddings=None, force_fusion_weight=0.0)
        # Should be identical (iteration 2 zeroed out)
        assert torch.allclose(out, out_no_text, atol=1e-5)

    def test_gradient_flows_through_both_iterations(self):
        loop = ExecutiveReasoningLoop(
            embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
            gnn_spectral_layers=0, max_freqs=8, tat_layers=1,
            tat_spatial_heads=4, tat_spectral_heads=0, tat_ff_dim=64,
            max_iterations=2, use_dual_track=True,
        )
        cc = self._make_cc()
        text_embs = torch.randn(10, 32, requires_grad=True)
        out, _, _ = loop(cc, text_embeddings=text_embs)
        out.sum().backward()
        # Text embeddings should receive gradient (through cross-attention)
        assert text_embs.grad is not None
        # GNN params should also have gradient
        gnn_has_grad = any(
            p.grad is not None
            for p in loop.gnn_executive.parameters()
        )
        assert gnn_has_grad
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_reasoning_loop/test_dual_track.py -v`
Expected: FAIL

- [ ] **Step 3: Modify ExecutiveReasoningLoop**

Key changes to `src/reasoning_loop/executive_loop.py`:

1. Add `use_dual_track: bool = False` parameter to `__init__`
2. When `use_dual_track=True`, create:
   - `self.text_gnn = TextConditionedGNN(embedding_dim, gnn_hidden, num_layers=2)`
   - `self.cross_attn = CrossAttentionBlock(embedding_dim, num_heads=4)`
3. Add `text_embeddings` and `force_fusion_weight` params to `forward()` and `forward_batched()`
4. Dual-track forward logic:
   ```python
   # Iteration 1: structural (existing code, 1 iteration)
   # IMPORTANT: Do NOT detach embeddings before TAT in dual-track mode (spec C2)
   # Use cc.set_embeddings(0, gnn_out) without .detach() so gradients flow back
   h_struct, _, control = self.gnn_executive.forward_with_control(cc, ...)
   cc.set_embeddings(0, gnn_out)  # NO .detach() in dual-track mode
   h_struct = self.tat(h_struct, cc, control_signal=control)

   # Get fusion_weight from control signal
   fw = control.fusion_weight if force_fusion_weight is None else torch.tensor(force_fusion_weight)

   # Iteration 2: cross-modal fusion (new)
   # NOTE: No wave dynamics in iteration 2 (spec requirement)
   if text_embeddings is not None and fw > 0:
       # Extract edge_index from CellComplex cell tuples, NOT from B1.nonzero()
       edge_index = torch.tensor(
           [[u, v] for (u, v) in cc.cells[1].keys()], dtype=torch.long
       ).t().contiguous().to(h_struct.device)  # (2, E)
       h_text_gnn = self.text_gnn(h_struct, edge_index, text_embeddings)
       h_cross = self.cross_attn(h_text_gnn, text_embeddings)
       h_final = self.tat(h_cross, cc, control_signal=control)
       h_out = (1 - fw) * h_struct + fw * h_final
   else:
       h_out = h_struct
   ```
5. Remove `.detach()` between iterations in dual-track mode:
   - Line ~269: `cc.set_embeddings(0, gnn_out.detach())` → `cc.set_embeddings(0, gnn_out)` when `use_dual_track=True`
   - Line ~282: same pattern — keep `.detach()` for backward-compat non-dual-track path
   - Gradient clipping `max_norm=5.0` mitigates deeper gradient chain (already in config)
6. Store `fusion_weight` in diagnostics dict
7. Keep deprecated ControlSignal fields (text_gate, structure_gate, etc.) as `None` — removal deferred to avoid breaking non-dual-track consumers

**Important**: Keep the existing non-dual-track path working for backward compatibility. The `use_dual_track=False` path should be identical to current behavior.

- [ ] **Step 4: Run ALL executive loop tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_reasoning_loop/ -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 5: Run full test suite to check no regressions**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/ -x --timeout=120`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/reasoning_loop/executive_loop.py tests/test_reasoning_loop/test_dual_track.py
git commit -m "feat: dual-track executive loop with asymmetric iterations"
```

---

### Task 7: Batch Utils — New Classifier Input Assembly

**Files:**
- Modify: `src/training/batch_utils.py`
- Test: `tests/test_training/test_batch_utils.py` (add tests)

- [ ] **Step 1: Write failing tests**

```python
# Append to or create tests/test_training/test_v10_batch.py

import torch
from unittest.mock import MagicMock
from src.reasoning_loop.attention_readout import AttentionReadout


class TestV10BatchForward:

    def test_attention_readout_builds_correct_dim(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        topo = torch.randn(4)
        fw = torch.tensor(0.5)
        combined = readout.build_classifier_input(
            h_out, 0, 3, task_id=5, topo_features=topo, fusion_weight=fw,
        )
        assert combined.shape == (101,)

    def test_attention_readout_without_optional_features(self):
        readout = AttentionReadout(embed_dim=32, num_tasks=19)
        h_out = torch.randn(10, 32)
        combined = readout.build_classifier_input(h_out, 0, 3, task_id=5)
        # 32 + 32 + 32 = 96 (no topo, no fusion)
        assert combined.shape == (96,)
```

- [ ] **Step 2: Run tests to verify they pass** (these use already-implemented AttentionReadout)

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_training/test_v10_batch.py -v`
Expected: PASS (AttentionReadout already implemented in Task 3)

- [ ] **Step 3: Modify _forward_batch in batch_utils.py**

Key changes:
1. When `model.use_dual_track` is True, use AttentionReadout path
2. Pass text_embeddings from QwenContextualEncoder to executive loop
3. Use `readout.build_classifier_input()` instead of manual query/target extraction
4. Compute topo_features (hodge 3-vec + wave energy) as before
5. Extract fusion_weight from diagnostics

The old path (non-dual-track) remains unchanged for backward compat.

- [ ] **Step 4: Run batch_utils tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_training/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/batch_utils.py tests/test_training/test_v10_batch.py
git commit -m "feat: v10 batch forward with attention readout and dual-track support"
```

---

## Chunk 3: New KG Tasks (Tasks 8-12)

### Task 8: kg_transitive Task Generator

**Files:**
- Modify: `src/benchmarks/conceptnet_tasks.py`
- Test: `tests/test_benchmarks/test_v10_tasks.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_benchmarks/test_v10_tasks.py
import pytest
import networkx as nx
from src.benchmarks.conceptnet_tasks import (
    generate_kg_transitive_task,
    TRANSITIVE_CLASSES,
)


def _make_conceptnet_graph():
    """Create a small ConceptNet-like graph for testing."""
    G = nx.DiGraph()
    # IsA chain: dog -> mammal -> animal
    G.add_edge("dog", "mammal", relation="IsA", weight=2.0)
    G.add_edge("mammal", "animal", relation="IsA", weight=2.0)
    # AtLocation: dog -> park, park -> city
    G.add_edge("dog", "park", relation="AtLocation", weight=1.5)
    G.add_edge("park", "city", relation="PartOf", weight=1.5)
    # More edges for subgraph extraction
    G.add_edge("cat", "mammal", relation="IsA", weight=2.0)
    G.add_edge("bird", "animal", relation="IsA", weight=2.0)
    G.add_edge("robin", "bird", relation="IsA", weight=2.0)
    G.add_edge("fish", "animal", relation="IsA", weight=2.0)
    G.add_edge("dog", "bone", relation="UsedFor", weight=1.0)
    G.add_edge("cat", "house", relation="AtLocation", weight=1.0)
    G.add_edge("bird", "sky", relation="AtLocation", weight=1.0)
    G.add_edge("house", "city", relation="PartOf", weight=1.0)
    G.add_edge("sky", "nature", relation="PartOf", weight=1.0)
    G.add_edge("bone", "dog", relation="UsedFor", weight=1.0)
    G.add_edge("park", "nature", relation="PartOf", weight=1.0)
    return G


class TestKgTransitive:

    def test_transitive_classes_defined(self):
        assert len(TRANSITIVE_CLASSES) == 5
        assert "none" in TRANSITIVE_CLASSES

    def test_generate_returns_5_tuple(self):
        G = _make_conceptnet_graph()
        result = generate_kg_transitive_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
        assert result is not None
        cc, query, target, answer, meta = result
        assert 0 <= answer < 5
        assert meta['task_type'] == 'kg_transitive'

    def test_answer_in_range(self):
        G = _make_conceptnet_graph()
        for _ in range(20):
            result = generate_kg_transitive_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
            if result is not None:
                _, _, _, answer, _ = result
                assert 0 <= answer < len(TRANSITIVE_CLASSES)

    def test_metadata_has_chain(self):
        G = _make_conceptnet_graph()
        result = generate_kg_transitive_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
        if result is not None:
            _, _, _, _, meta = result
            assert 'chain' in meta or 'path' in meta
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_benchmarks/test_v10_tasks.py::TestKgTransitive -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement generate_kg_transitive_task**

Add to `src/benchmarks/conceptnet_tasks.py`:

```python
TRANSITIVE_CLASSES = ["IsA", "HasA", "PartOf", "Causes", "none"]

# Define which relations are transitive
_TRANSITIVE_RELATIONS = {"IsA", "PartOf", "HasA"}

def generate_kg_transitive_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple | None:
    """Generate transitive inference task.

    Finds A→B→C chain with known relations, asks whether A→C holds transitively.
    Answer is the transitive relation class, or 'none' if not transitive.
    """
    # 1. Extract subgraph
    subgraph = _safe_extract(G, min_nodes, max_nodes)
    if subgraph is None:
        return None

    # 2. Find a 2-hop path A→B→C with labeled relations
    nodes = list(subgraph.nodes())
    random.shuffle(nodes)
    for a in nodes:
        for b in subgraph.successors(a):
            rel_ab = subgraph[a][b].get('relation', '')
            for c in subgraph.successors(b):
                if c == a:
                    continue
                rel_bc = subgraph[b][c].get('relation', '')

                # Determine if transitive
                if rel_ab == rel_bc and rel_ab in _TRANSITIVE_RELATIONS:
                    answer_str = rel_ab  # Same transitive relation
                else:
                    answer_str = "none"

                answer = TRANSITIVE_CLASSES.index(answer_str)

                # Build CellComplex from subgraph
                cc = conceptnet_subgraph_to_cc(subgraph, embedding_dim)
                node_list = list(subgraph.nodes())
                query_idx = node_list.index(a)
                target_idx = node_list.index(c)

                meta = {
                    'task_type': 'kg_transitive',
                    'chain': [a, b, c],
                    'relations': [rel_ab, rel_bc],
                    'answer_str': answer_str,
                }
                return cc, query_idx, target_idx, answer, meta

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_benchmarks/test_v10_tasks.py::TestKgTransitive -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks/conceptnet_tasks.py tests/test_benchmarks/test_v10_tasks.py
git commit -m "feat: kg_transitive task generator for multi-hop transitive inference"
```

---

### Task 9: kg_consistency Task Generator

**Files:**
- Modify: `src/benchmarks/conceptnet_tasks.py`
- Test: `tests/test_benchmarks/test_v10_tasks.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_benchmarks/test_v10_tasks.py

from src.benchmarks.conceptnet_tasks import generate_kg_consistency_task


class TestKgConsistency:

    def test_generate_returns_5_tuple(self):
        G = _make_conceptnet_graph()
        result = generate_kg_consistency_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
        assert result is not None
        cc, query, target, answer, meta = result
        assert answer in (0, 1)  # binary: consistent or contradiction
        assert meta['task_type'] == 'kg_consistency'

    def test_balanced_classes(self):
        G = _make_conceptnet_graph()
        answers = []
        for _ in range(50):
            result = generate_kg_consistency_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
            if result is not None:
                answers.append(result[3])
        if len(answers) > 10:
            # Should have both classes
            assert 0 in answers and 1 in answers

    def test_metadata_has_corruption_info(self):
        G = _make_conceptnet_graph()
        result = generate_kg_consistency_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
        if result is not None:
            _, _, _, _, meta = result
            assert 'corrupted' in meta
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_benchmarks/test_v10_tasks.py::TestKgConsistency -v`
Expected: FAIL

- [ ] **Step 3: Implement generate_kg_consistency_task**

Add to `src/benchmarks/conceptnet_tasks.py`:

```python
def generate_kg_consistency_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple | None:
    """Generate logical consistency checking task.

    50% of samples are consistent (answer=1), 50% have a semantic contradiction (answer=0).
    Corruption: replace one edge's target with a concept from a different IsA branch.
    """
    subgraph = _safe_extract(G, min_nodes, max_nodes)
    if subgraph is None:
        return None

    corrupted = random.random() < 0.5

    if corrupted:
        # Find an IsA edge and swap target with incompatible concept
        isa_edges = [(u, v) for u, v, d in subgraph.edges(data=True)
                     if d.get('relation') == 'IsA']
        if not isa_edges:
            corrupted = False
        else:
            u, v = random.choice(isa_edges)
            # Find v's siblings (other children of v's parent) that are in different branches
            parents_of_v = [p for p in G.predecessors(v)
                           if G[p][v].get('relation') == 'IsA'] if v in G else []
            candidates = []
            for parent in parents_of_v:
                siblings = [s for s in G.successors(parent)
                           if G[parent][s].get('relation') == 'IsA'
                           and s != v and s in subgraph]
                candidates.extend(siblings)

            if not candidates:
                # Fallback: pick any concept not in subgraph's IsA ancestors of u
                all_nodes = list(subgraph.nodes())
                candidates = [n for n in all_nodes if n != u and n != v]

            if candidates:
                replacement = random.choice(candidates)
                # Swap v with replacement in the subgraph
                subgraph = subgraph.copy()
                subgraph.remove_edge(u, v)
                subgraph.add_edge(u, replacement,
                                 relation='IsA', weight=1.0)
            else:
                corrupted = False

    answer = 0 if corrupted else 1
    cc = conceptnet_subgraph_to_cc(subgraph, embedding_dim)
    node_list = list(subgraph.nodes())

    if corrupted and u in node_list and replacement in node_list:
        # Point query/target at the corrupted edge for meaningful signal
        query_idx = node_list.index(u)
        target_idx = node_list.index(replacement)
    else:
        query_idx = random.randint(0, len(node_list) - 1)
        target_idx = random.randint(0, len(node_list) - 1)
        if target_idx == query_idx:
            target_idx = (query_idx + 1) % len(node_list)

    meta = {
        'task_type': 'kg_consistency',
        'corrupted': corrupted,
        'num_nodes': len(node_list),
    }
    return cc, query_idx, target_idx, answer, meta
```

- [ ] **Step 4: Run tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_benchmarks/test_v10_tasks.py::TestKgConsistency -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks/conceptnet_tasks.py tests/test_benchmarks/test_v10_tasks.py
git commit -m "feat: kg_consistency task for logical contradiction detection"
```

---

### Task 10: kg_analogy Redesign

**Files:**
- Modify: `src/benchmarks/conceptnet_tasks.py`
- Test: `tests/test_benchmarks/test_v10_tasks.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_benchmarks/test_v10_tasks.py

from src.benchmarks.conceptnet_tasks import generate_kg_analogy_task_v10


class TestKgAnalogyV10:

    def test_generate_returns_5_tuple(self):
        G = _make_conceptnet_graph()
        result = generate_kg_analogy_task_v10(G, embedding_dim=32, min_nodes=3, max_nodes=8)
        assert result is not None
        cc, query, target, answer, meta = result
        assert 0 <= answer <= 2
        assert meta['task_type'] == 'kg_analogy'

    def test_metadata_has_subgraph_info(self):
        G = _make_conceptnet_graph()
        result = generate_kg_analogy_task_v10(G, embedding_dim=32, min_nodes=3, max_nodes=8)
        if result is not None:
            _, _, _, _, meta = result
            assert 'subgraph_a_size' in meta
            assert 'subgraph_b_size' in meta
```

- [ ] **Step 2-5**: Implement with `nx.is_isomorphic()` edge-type matching, test, commit.

```bash
git commit -m "feat: redesigned kg_analogy with structural isomorphism matching"
```

---

### Task 11: kg_causal_chain Task Generator

**Files:**
- Modify: `src/benchmarks/conceptnet_tasks.py`
- Test: `tests/test_benchmarks/test_v10_tasks.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_benchmarks/test_v10_tasks.py

from src.benchmarks.conceptnet_tasks import (
    generate_kg_causal_chain_task,
    CAUSAL_CHAIN_CLASSES,
)


class TestKgCausalChain:

    def test_causal_classes_defined(self):
        assert len(CAUSAL_CHAIN_CLASSES) == 3
        assert "coherent" in CAUSAL_CHAIN_CLASSES
        assert "broken" in CAUSAL_CHAIN_CLASSES
        assert "incoherent" in CAUSAL_CHAIN_CLASSES

    def test_generate_returns_5_tuple(self):
        G = _make_conceptnet_graph()
        result = generate_kg_causal_chain_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
        assert result is not None
        cc, query, target, answer, meta = result
        assert 0 <= answer < 3
        assert meta['task_type'] == 'kg_causal_chain'

    def test_metadata_has_chain(self):
        G = _make_conceptnet_graph()
        result = generate_kg_causal_chain_task(G, embedding_dim=32, min_nodes=5, max_nodes=15)
        if result is not None:
            _, _, _, _, meta = result
            assert 'chain' in meta
            assert 'class_name' in meta
```

- [ ] **Step 2-5**: Implement using ConceptNet Causes/UsedFor/CapableOf edge chains, test, commit.

```bash
git commit -m "feat: kg_causal_chain task for causal coherence reasoning"
```

---

### Task 12: Update TASK_REGISTRY

**Files:**
- Modify: `src/benchmarks/benchmark_dataset.py`
- Test: existing tests should cover

- [ ] **Step 1: Update TASK_REGISTRY**

Replace old KG entries:
```python
# Remove: kg_concept, kg_pathvalid, kg_cluster
# Keep: kg_relation
# Add: kg_transitive (5 classes), kg_consistency (2 classes),
#       kg_analogy (3 classes, updated), kg_causal_chain (3 classes)
```

- [ ] **Step 2: Update get_max_classes() if it exists**

Ensure the new tasks return correct class counts.

- [ ] **Step 3: Run benchmark dataset tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_benchmarks/ -v`
Expected: PASS (may need to update some test constants)

- [ ] **Step 4: Commit**

```bash
git add src/benchmarks/benchmark_dataset.py
git commit -m "feat: update TASK_REGISTRY with v10 KG tasks"
```

---

## Chunk 4: Integration & Config (Tasks 13-15)

### Task 13: v10 Config YAML

**Files:**
- Create: `config/v10_dual_track.yaml`

- [ ] **Step 1: Create config**

```yaml
# config/v10_dual_track.yaml
# v10: Dual-Track Fusion — asymmetric executive loop + redesigned KG tasks

model:
  embedding_dim: 32
  gnn_hidden: 64
  gnn_spatial_layers: 2
  gnn_spectral_layers: 2
  max_freqs: 16
  tat_layers: 2
  tat_spatial_heads: 4
  tat_spectral_heads: 4
  tat_ff_dim: 128
  max_iterations: 2
  convergence_threshold: 0.05
  use_higher_order: true
  use_topological_pe: true
  use_structural_features: true
  use_wave_dynamics: true
  use_topo_feedback: true
  use_embedding_topo_feedback: true
  use_multi_head_classifier: true
  use_dual_track: true      # v10: asymmetric iterations

wave:
  wave_mode: ensemble
  filter_types: [chebyshev, wave_cosine, heat]
  include_identity: true
  wave_strength_gate: true
  use_neural_ode: true

llm:
  backend: qwen_contextual
  qwen_model: Qwen/Qwen2.5-3B-Instruct
  llm_dim: 2048
  output_dim: 32
  num_qwen_layers: 4
  max_tokens_per_concept: 16
  num_tasks: 19  # 14 original + 5 KG - 3 removed + 3 new KG = 19
  use_mock: false

training:
  device: auto
  learning_rate: 0.0003
  cross_attn_learning_rate: 0.001
  readout_learning_rate: 0.001
  weight_decay: 0.01
  label_smoothing: 0.02
  loss_fn: focal
  focal_gamma: 2.0
  curl_penalty_weight: 0.01
  curl_penalty_threshold: 0.5
  scheduler: cosine_warm_restarts
  scheduler_eta_min: 0.00001
  batch_size: 8
  use_batched: true
  epochs_phase_a: 5
  patience: 15
  max_norm: 5.0
  accumulation_steps: 4
  replay_ratio: 0.2

topology_observer:
  enabled: true
  analyze_every: 5
  active_mode: true

benchmark:
  train_samples: 5000
  val_samples: 500
  train_n_nodes: 20
  train_n_nodes_min: 16
  train_n_nodes_max: 32
  checkpoint_dir: data/v10_dual_track/checkpoints
  pregen_dir: data/v10_datasets
  conceptnet_path: data/conceptnet/conceptnet_en.pkl

phase_a:
  tasks: [bfs, hodge_class, diverse]
  freeze_fusion: true

phase_b:
  tasks: [kg_relation, kg_transitive, kg_consistency, kg_analogy, kg_causal_chain]
  epochs:
    kg_relation: 30
    kg_transitive: 30
    kg_consistency: 30
    kg_analogy: 30
    kg_causal_chain: 30
  kg_train_samples: 25000
  kg_val_samples: 2000
  replay_structural_ratio: 0.2
  resume_from: data/v6_track2/checkpoints/phase_c_labeled_reasoning_best.pt
```

- [ ] **Step 2: Commit**

```bash
git add config/v10_dual_track.yaml
git commit -m "feat: v10 dual-track config"
```

---

### Task 14: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration/test_v10_forward.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration/test_v10_forward.py
"""End-to-end test: build v10 model, run forward pass, verify shapes and gradients."""
import pytest
import torch
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop
from src.reasoning_loop.attention_readout import AttentionReadout
from src.llm.qwen_contextual_encoder import QwenContextualEncoder


class TestV10EndToEnd:

    def _make_cc(self, n=10, embed_dim=32):
        cc = CellComplex()
        for i in range(n):
            cc.add_cell(0, (i,), embedding=torch.randn(embed_dim))
        for i in range(n - 1):
            cc.add_cell(1, (i, i + 1), embedding=torch.randn(embed_dim))
        cc.node_texts = [f"concept_{i}" for i in range(n)]
        return cc

    def test_full_forward_pass(self):
        """Build all v10 components, run one sample, verify output."""
        embed_dim = 32

        # 1. Qwen encoder (mock)
        encoder = QwenContextualEncoder(llm_dim=64, output_dim=embed_dim, use_mock=True)

        # 2. Executive loop (dual track)
        loop = ExecutiveReasoningLoop(
            embedding_dim=embed_dim, gnn_hidden=64, gnn_spatial_layers=2,
            gnn_spectral_layers=0, max_freqs=8, tat_layers=1,
            tat_spatial_heads=4, tat_spectral_heads=0, tat_ff_dim=64,
            max_iterations=2, use_dual_track=True,
        )

        # 3. Readout
        readout = AttentionReadout(embed_dim=embed_dim, num_tasks=19)

        # 4. Classifier
        classifier = torch.nn.Sequential(
            torch.nn.Linear(101, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 10),
        )

        # Run
        cc = self._make_cc()
        device = torch.device("cpu")

        text_embs = encoder(cc.node_texts, device)
        assert text_embs.shape == (10, embed_dim)

        h_out, iters, diag = loop(cc, text_embeddings=text_embs)
        assert h_out.shape == (10, embed_dim)
        assert 'fusion_weight' in diag

        topo = torch.randn(4)
        fw = diag['fusion_weight'] if isinstance(diag['fusion_weight'], torch.Tensor) else torch.tensor(diag['fusion_weight'])
        combined = readout.build_classifier_input(
            h_out, 0, 3, task_id=0, topo_features=topo, fusion_weight=fw,
        )
        assert combined.shape == (101,)

        logits = classifier(combined)
        assert logits.shape == (10,)

    def test_gradient_flows_end_to_end(self):
        embed_dim = 32
        encoder = QwenContextualEncoder(llm_dim=64, output_dim=embed_dim, use_mock=True)
        loop = ExecutiveReasoningLoop(
            embedding_dim=embed_dim, gnn_hidden=64, gnn_spatial_layers=2,
            gnn_spectral_layers=0, max_freqs=8, tat_layers=1,
            tat_spatial_heads=4, tat_spectral_heads=0, tat_ff_dim=64,
            max_iterations=2, use_dual_track=True,
        )
        readout = AttentionReadout(embed_dim=embed_dim, num_tasks=19)
        classifier = torch.nn.Sequential(
            torch.nn.Linear(101, 64), torch.nn.GELU(), torch.nn.Linear(64, 5),
        )

        cc = self._make_cc()
        text_embs = encoder(cc.node_texts, torch.device("cpu"))
        h_out, _, diag = loop(cc, text_embeddings=text_embs)

        topo = torch.randn(4)
        fw = diag.get('fusion_weight', torch.tensor(0.5))
        if not isinstance(fw, torch.Tensor):
            fw = torch.tensor(fw)
        combined = readout.build_classifier_input(
            h_out, 0, 3, task_id=0, topo_features=topo, fusion_weight=fw,
        )
        logits = classifier(combined)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([2]))
        loss.backward()

        # Verify gradients reach all components
        assert encoder.proj.weight.grad is not None, "No gradient to Qwen projection"
        gnn_grads = sum(1 for p in loop.parameters() if p.grad is not None)
        assert gnn_grads > 0, "No gradients to executive loop"
        readout_grads = sum(1 for p in readout.parameters() if p.grad is not None)
        assert readout_grads > 0, "No gradients to readout"
```

- [ ] **Step 2: Run integration test**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_integration/test_v10_forward.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration/test_v10_forward.py
git commit -m "test: v10 end-to-end integration tests"
```

---

### Task 15: Training Script Updates

**Files:**
- Modify: `scripts/run_dsm_curriculum.py`

- [ ] **Step 1: Add v10 training path**

Key changes to `run_dsm_curriculum.py`:
1. Detect `use_dual_track` from config
2. Build QwenContextualEncoder instead of QwenTextFeatureExtractor
3. Phase A: freeze cross-attn/readout, force fusion_weight=0
4. Phase B: train all, pass text_embeddings through
5. Optimizer groups: GNN/TAT (0.0003), cross-attn/Qwen proj (0.001), readout/classifier (0.001)
6. Add curl penalty to loss computation
7. Log fusion_weight per epoch

- [ ] **Step 2: Run existing curriculum tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/test_scripts/ -v`
Expected: PASS (backward compat)

- [ ] **Step 3: Commit**

```bash
git add scripts/run_dsm_curriculum.py
git commit -m "feat: v10 dual-track training path in run_dsm_curriculum"
```

---

### Task 16: Full Test Suite Verification

- [ ] **Step 1: Run entire test suite**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && .venv/bin/python -m pytest tests/ --timeout=120 -q`
Expected: All tests PASS (700+ existing + ~40 new)

- [ ] **Step 2: Fix any failures**

- [ ] **Step 3: Final commit**

```bash
git commit -m "chore: v10 test suite green"
```

---

## Summary

| Task | Component | Est. Complexity |
|------|-----------|----------------|
| 1 | QwenContextualEncoder | Medium (new module, caching) |
| 2 | CrossAttentionBlock | Simple (wraps nn.MultiheadAttention) |
| 3 | AttentionReadout | Simple (attention + assembly) |
| 4 | TextConditionedGNN | Medium (custom MessagePassing) |
| 5 | ControlHead fusion_weight | Simple (add one head) |
| 6 | Dual-Track Executive Loop | **Complex** (core refactor) |
| 7 | Batch Utils update | Medium (new forward path) |
| 8 | kg_transitive | Medium (ConceptNet chain logic) |
| 9 | kg_consistency | Medium (semantic corruption) |
| 10 | kg_analogy redesign | **Complex** (isomorphism matching) |
| 11 | kg_causal_chain | Medium (causal chain extraction) |
| 12 | TASK_REGISTRY update | Simple |
| 13 | Config YAML | Simple |
| 14 | Integration tests | Medium |
| 15 | Training script | Medium |
| 16 | Full test verification | Simple |

**Critical path**: Tasks 1-4 (new modules) → Task 5-6 (core integration) → Task 7 (batch) → Task 14 (verify) → Tasks 8-12 (tasks) → Task 15 (training)

**Parallelizable**: Tasks 1-4 are independent. Tasks 8-11 are independent.

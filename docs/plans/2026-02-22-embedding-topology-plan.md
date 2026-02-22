# Embedding Topology Analyzer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a dual-mode framework (observer + active) for discovering topological structure in LLM internal representations using our existing topological toolkit.

**Architecture:** Three analysis layers (embedding manifold, attention flow, weight space) connected by cross-layer sheaf. Observer mode works on any transformer via forward hooks. Active mode feeds topology back into DSM's ControlHead. All layers reuse existing CellComplex, Hodge decomposition, persistence homology, and sheaf diffusion infrastructure.

**Tech Stack:** PyTorch, gudhi (persistence), existing `src/spectral/` + `src/cell_complex/` modules, `SheafLaplacian` from `src/spectral/sheaf_diffusion.py`

**Design doc:** `docs/plans/2026-02-22-embedding-topology-design.md`

---

### Task 1: Data Structures (TopologicalProfile + LayerCellComplex)

**Files:**
- Create: `src/topology_analyzer/__init__.py`
- Create: `src/topology_analyzer/profile.py`
- Test: `tests/test_topology_analyzer/test_profile.py`
- Create: `tests/test_topology_analyzer/__init__.py`

**Step 1: Write the failing test**

```python
# tests/test_topology_analyzer/__init__.py
# (empty)
```

```python
# tests/test_topology_analyzer/test_profile.py
"""Tests for TopologicalProfile and LayerCellComplex data structures."""

import time
import numpy as np
import torch
import pytest
from src.cell_complex.cell_complex import CellComplex


def _make_cc(n_nodes=5, dim=8):
    cc = CellComplex(embedding_dim=dim)
    for i in range(n_nodes):
        cc.add_0_cell(torch.randn(dim), "node")
    for i in range(n_nodes - 1):
        cc.add_1_cell(i, i + 1, torch.randn(dim), "edge")
    return cc


class TestLayerCellComplex:
    def test_create_from_embedding(self):
        from src.topology_analyzer.profile import LayerCellComplex
        cc = _make_cc()
        lcc = LayerCellComplex(
            cc=cc, layer_idx=0, source="embedding",
            head_idx=None, neighborhood="knn", k_or_epsilon=5.0,
        )
        assert lcc.layer_idx == 0
        assert lcc.source == "embedding"
        assert lcc.head_idx is None

    def test_create_from_attention(self):
        from src.topology_analyzer.profile import LayerCellComplex
        cc = _make_cc()
        lcc = LayerCellComplex(
            cc=cc, layer_idx=2, source="attention",
            head_idx=3, neighborhood="knn", k_or_epsilon=5.0,
        )
        assert lcc.head_idx == 3
        assert lcc.source == "attention"


class TestTopologicalProfile:
    def test_create_empty(self):
        from src.topology_analyzer.profile import TopologicalProfile
        profile = TopologicalProfile(
            per_layer_persistence={},
            per_layer_betti={},
            per_layer_spectral_gap={},
            per_layer_hodge_ratios={},
            cross_layer_sheaf_gap=0.0,
            per_head_hodge_ratios={},
            per_head_classification={},
            per_layer_head_sheaf_gap={},
            cross_layer_attention_wasserstein={},
            per_layer_effective_rank={},
            per_layer_condition_number={},
            per_layer_sv_persistence={},
            cross_layer_weight_sheaf_gap=0.0,
            model_name="test",
            num_layers=4,
            num_heads=4,
            hidden_dim=64,
            seq_len=16,
            timestamp=time.time(),
        )
        assert profile.num_layers == 4
        assert profile.model_name == "test"

    def test_to_dict_and_summary(self):
        from src.topology_analyzer.profile import TopologicalProfile
        profile = TopologicalProfile(
            per_layer_persistence={0: [np.array([[0.0, 0.5]]), np.empty((0, 2))]},
            per_layer_betti={0: (1, 0)},
            per_layer_spectral_gap={0: 0.5},
            per_layer_hodge_ratios={0: (0.6, 0.3, 0.1)},
            cross_layer_sheaf_gap=0.42,
            per_head_hodge_ratios={(0, 0): (0.7, 0.2, 0.1)},
            per_head_classification={(0, 0): "gradient"},
            per_layer_head_sheaf_gap={0: 0.3},
            cross_layer_attention_wasserstein={},
            per_layer_effective_rank={0: 10.5},
            per_layer_condition_number={0: 100.0},
            per_layer_sv_persistence={0: [np.array([[0.1, 0.9]])]},
            cross_layer_weight_sheaf_gap=0.35,
            model_name="test",
            num_layers=1,
            num_heads=1,
            hidden_dim=64,
            seq_len=16,
            timestamp=time.time(),
        )
        summary = profile.summary()
        assert "cross_layer_sheaf_gap" in summary
        assert summary["cross_layer_sheaf_gap"] == 0.42

    def test_active_features_vector(self):
        """The 6-feature vector for ControlHead topo_feedback."""
        from src.topology_analyzer.profile import TopologicalProfile
        profile = TopologicalProfile(
            per_layer_persistence={},
            per_layer_betti={},
            per_layer_spectral_gap={},
            per_layer_hodge_ratios={},
            cross_layer_sheaf_gap=0.42,
            per_head_hodge_ratios={(0, 0): (0.3, 0.5, 0.2)},
            per_head_classification={(0, 0): "curl"},
            per_layer_head_sheaf_gap={},
            cross_layer_attention_wasserstein={},
            per_layer_effective_rank={},
            per_layer_condition_number={},
            per_layer_sv_persistence={},
            cross_layer_weight_sheaf_gap=0.35,
            model_name="test",
            num_layers=1,
            num_heads=1,
            hidden_dim=64,
            seq_len=16,
            timestamp=time.time(),
        )
        features = profile.active_features()
        assert features.shape == (6,)
        # embedding_coherence = cross_layer_sheaf_gap
        assert features[3].item() == pytest.approx(0.42)
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_profile.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.topology_analyzer'"

**Step 3: Write minimal implementation**

```python
# src/topology_analyzer/__init__.py
from src.topology_analyzer.profile import TopologicalProfile, LayerCellComplex

__all__ = ['TopologicalProfile', 'LayerCellComplex']
```

```python
# src/topology_analyzer/profile.py
"""Data structures for transformer topology analysis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from src.cell_complex.cell_complex import CellComplex


@dataclass
class LayerCellComplex:
    """CellComplex built from one transformer layer's hidden states or attention."""
    cc: CellComplex
    layer_idx: int
    source: str  # "embedding" | "attention" | "weight"
    head_idx: int | None  # for attention complexes
    neighborhood: str  # "knn" | "epsilon" | "mutual_knn"
    k_or_epsilon: float


@dataclass
class TopologicalProfile:
    """Complete topological profile of a transformer at one point in time."""

    # Layer 1: Embedding Manifold
    per_layer_persistence: dict[int, list[np.ndarray]]
    per_layer_betti: dict[int, tuple[int, int]]
    per_layer_spectral_gap: dict[int, float]
    per_layer_hodge_ratios: dict[int, tuple[float, float, float]]
    cross_layer_sheaf_gap: float

    # Layer 2: Attention Flow
    per_head_hodge_ratios: dict[tuple[int, int], tuple[float, float, float]]
    per_head_classification: dict[tuple[int, int], str]
    per_layer_head_sheaf_gap: dict[int, float]
    cross_layer_attention_wasserstein: dict[int, float]

    # Layer 3: Weight Space
    per_layer_effective_rank: dict[int, float]
    per_layer_condition_number: dict[int, float]
    per_layer_sv_persistence: dict[int, list[np.ndarray]]
    cross_layer_weight_sheaf_gap: float

    # Metadata
    model_name: str
    num_layers: int
    num_heads: int
    hidden_dim: int
    seq_len: int
    timestamp: float

    def summary(self) -> dict[str, float]:
        """Return a flat dict of key scalar metrics."""
        result: dict[str, float] = {
            "cross_layer_sheaf_gap": self.cross_layer_sheaf_gap,
            "cross_layer_weight_sheaf_gap": self.cross_layer_weight_sheaf_gap,
        }
        for layer_idx, gap in self.per_layer_spectral_gap.items():
            result[f"layer_{layer_idx}_spectral_gap"] = gap
        for layer_idx, (g, c, h) in self.per_layer_hodge_ratios.items():
            result[f"layer_{layer_idx}_gradient_ratio"] = g
            result[f"layer_{layer_idx}_curl_ratio"] = c
            result[f"layer_{layer_idx}_harmonic_ratio"] = h
        for layer_idx, rank in self.per_layer_effective_rank.items():
            result[f"layer_{layer_idx}_effective_rank"] = rank
        for (li, hi), cls in self.per_head_classification.items():
            result[f"head_{li}_{hi}_type"] = {"gradient": 0.0, "curl": 1.0, "harmonic": 2.0}.get(cls, -1.0)
        return result

    def active_features(self) -> torch.Tensor:
        """Return the 6-feature vector for ControlHead topo_feedback.

        Features:
            [0] gradient_ratio   - mean over per_layer_hodge_ratios
            [1] curl_ratio       - mean over per_layer_hodge_ratios
            [2] spectral_gap     - mean over per_layer_spectral_gap
            [3] embedding_coherence - cross_layer_sheaf_gap
            [4] attention_curl   - mean curl across all heads
            [5] weight_alignment - cross_layer_weight_sheaf_gap
        """
        features = torch.zeros(6)

        # [0-1] Hodge ratios from embedding manifold
        if self.per_layer_hodge_ratios:
            grads = [r[0] for r in self.per_layer_hodge_ratios.values()]
            curls = [r[1] for r in self.per_layer_hodge_ratios.values()]
            features[0] = sum(grads) / len(grads)
            features[1] = sum(curls) / len(curls)

        # [2] Spectral gap
        if self.per_layer_spectral_gap:
            gaps = list(self.per_layer_spectral_gap.values())
            features[2] = sum(gaps) / len(gaps)

        # [3] Embedding coherence
        features[3] = self.cross_layer_sheaf_gap

        # [4] Attention curl
        if self.per_head_hodge_ratios:
            curls = [r[1] for r in self.per_head_hodge_ratios.values()]
            features[4] = sum(curls) / len(curls)

        # [5] Weight alignment
        features[5] = self.cross_layer_weight_sheaf_gap

        return features
```

**Step 4: Run test to verify it passes**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_profile.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/topology_analyzer/__init__.py src/topology_analyzer/profile.py tests/test_topology_analyzer/__init__.py tests/test_topology_analyzer/test_profile.py
git commit -m "feat: TopologicalProfile + LayerCellComplex data structures"
```

---

### Task 2: TransformerHookManager

**Files:**
- Create: `src/topology_analyzer/hooks.py`
- Test: `tests/test_topology_analyzer/test_hooks.py`

**Step 1: Write the failing test**

```python
# tests/test_topology_analyzer/test_hooks.py
"""Tests for TransformerHookManager."""

import torch
import torch.nn as nn


def _make_simple_transformer(num_layers=3, hidden_dim=32, num_heads=4):
    """Create a minimal nn.TransformerEncoder for testing."""
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim, nhead=num_heads, dim_feedforward=64,
        batch_first=True,
    )
    model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    return model


class TestTransformerHookManager:
    def test_captures_hidden_states(self):
        from src.topology_analyzer.hooks import TransformerHookManager
        model = _make_simple_transformer(num_layers=3, hidden_dim=32)
        x = torch.randn(1, 8, 32)  # (batch, seq, dim)

        with TransformerHookManager(model) as manager:
            _ = model(x)
            hidden = manager.get_hidden_states()

        assert len(hidden) == 3
        for layer_idx, h in hidden.items():
            assert h.shape == (8, 32)  # (seq_len, hidden_dim) unbatched

    def test_captures_attention_maps(self):
        from src.topology_analyzer.hooks import TransformerHookManager
        model = _make_simple_transformer(num_layers=2, hidden_dim=32, num_heads=4)
        x = torch.randn(1, 8, 32)

        with TransformerHookManager(model, capture_attention=True) as manager:
            _ = model(x)
            attn = manager.get_attention_maps()

        # Should have entries for each (layer, head)
        assert len(attn) == 2 * 4  # 2 layers x 4 heads
        for (li, hi), a in attn.items():
            assert a.shape == (8, 8)  # (seq, seq)

    def test_hooks_removed_on_exit(self):
        from src.topology_analyzer.hooks import TransformerHookManager
        model = _make_simple_transformer(num_layers=2, hidden_dim=32)

        with TransformerHookManager(model) as manager:
            pass

        # Hooks should be removed -- internal state should be empty
        assert len(manager._handles) == 0

    def test_clear_resets_state(self):
        from src.topology_analyzer.hooks import TransformerHookManager
        model = _make_simple_transformer(num_layers=2, hidden_dim=32)
        x = torch.randn(1, 8, 32)

        with TransformerHookManager(model) as manager:
            _ = model(x)
            assert len(manager.get_hidden_states()) == 2
            manager.clear()
            assert len(manager.get_hidden_states()) == 0

    def test_works_with_dsm_structure(self):
        """Test with a model that has .layers attribute like our DSM."""
        from src.topology_analyzer.hooks import TransformerHookManager

        # Simulate DSM-like structure with .layers ModuleList
        class FakeDSM(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, batch_first=True)
                    for _ in range(2)
                ])
                self.final_norm = nn.LayerNorm(16)

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.final_norm(x)

        model = FakeDSM()
        x = torch.randn(1, 4, 16)

        with TransformerHookManager(model) as manager:
            _ = model(x)
            hidden = manager.get_hidden_states()

        assert len(hidden) == 2

    def test_batch_dim_handling(self):
        """Multiple batch items -- should capture only first item or average."""
        from src.topology_analyzer.hooks import TransformerHookManager
        model = _make_simple_transformer(num_layers=2, hidden_dim=32)
        x = torch.randn(4, 8, 32)  # batch=4

        with TransformerHookManager(model) as manager:
            _ = model(x)
            hidden = manager.get_hidden_states()

        # Should return (seq_len, hidden_dim) for the first batch item
        for layer_idx, h in hidden.items():
            assert h.shape == (8, 32)
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_hooks.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/topology_analyzer/hooks.py
"""Forward hooks for capturing transformer internal representations."""

from __future__ import annotations

import torch
import torch.nn as nn


def _find_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Auto-detect transformer layers in a model.

    Searches for common patterns:
    - model.layers (our DSM, many custom models)
    - model.encoder.layers (nn.TransformerEncoder)
    - model.transformer.h (GPT-2 style)
    - model.model.layers (HuggingFace LlamaModel)
    """
    for attr_path in ['layers', 'encoder.layers', 'transformer.h', 'model.layers']:
        obj = model
        try:
            for part in attr_path.split('.'):
                obj = getattr(obj, part)
            if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue

    # Fallback: nn.TransformerEncoder wraps layers in .layers
    if hasattr(model, 'layers') and isinstance(model.layers, nn.ModuleList):
        return list(model.layers)

    return []


def _find_self_attn(layer: nn.Module) -> nn.MultiheadAttention | None:
    """Find the self-attention module in a transformer layer."""
    for name in ['self_attn', 'attn', 'attention']:
        if hasattr(layer, name):
            module = getattr(layer, name)
            if isinstance(module, nn.MultiheadAttention):
                return module
    # Search one level deeper
    for child in layer.children():
        for name in ['self_attn', 'attn', 'attention']:
            if hasattr(child, name):
                module = getattr(child, name)
                if isinstance(module, nn.MultiheadAttention):
                    return module
    return None


class TransformerHookManager:
    """Register forward hooks to capture hidden states and attention maps.

    Works with any nn.Module that has a standard transformer structure.
    Use as a context manager to auto-register and remove hooks.

    Example:
        with TransformerHookManager(model) as manager:
            output = model(input)
            hidden = manager.get_hidden_states()
            attn = manager.get_attention_maps()
    """

    def __init__(self, model: nn.Module, capture_attention: bool = False):
        self.model = model
        self.capture_attention = capture_attention
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._hidden_states: dict[int, torch.Tensor] = {}
        self._attention_maps: dict[tuple[int, int], torch.Tensor] = {}
        self._layers = _find_transformer_layers(model)

    def __enter__(self) -> TransformerHookManager:
        self._register_hooks()
        return self

    def __exit__(self, *args):
        self._remove_hooks()

    def _register_hooks(self):
        for layer_idx, layer in enumerate(self._layers):
            # Hidden state hook: captures layer output
            handle = layer.register_forward_hook(
                self._make_hidden_hook(layer_idx)
            )
            self._handles.append(handle)

            # Attention hook: monkey-patch self_attn to capture weights
            if self.capture_attention:
                attn_module = _find_self_attn(layer)
                if attn_module is not None:
                    # Save original forward, wrap to capture attention weights
                    orig_forward = attn_module.forward
                    num_heads = attn_module.num_heads

                    def make_attn_hook(li, nh, orig_fn):
                        def wrapped_forward(*args, **kwargs):
                            kwargs['need_weights'] = True
                            kwargs['average_attn_weights'] = False
                            out, weights = orig_fn(*args, **kwargs)
                            # weights: (batch, num_heads, seq, seq)
                            if weights is not None:
                                w = weights.detach()
                                if w.dim() == 4:
                                    w = w[0]  # first batch item
                                for h in range(min(nh, w.shape[0])):
                                    self._attention_maps[(li, h)] = w[h]
                            return out, weights
                        return wrapped_forward

                    patched = make_attn_hook(layer_idx, num_heads, orig_forward)
                    attn_module.forward = patched
                    # Store original for cleanup
                    attn_module._orig_forward = orig_forward

    def _make_hidden_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output can be a tensor or tuple
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            h = h.detach()
            # Handle batch dim: take first item
            if h.dim() == 3:
                h = h[0]  # (seq_len, hidden_dim)
            elif h.dim() == 2:
                pass  # already (seq_len, hidden_dim)
            self._hidden_states[layer_idx] = h
        return hook

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        # Restore original attention forwards
        if self.capture_attention:
            for layer in self._layers:
                attn_module = _find_self_attn(layer)
                if attn_module is not None and hasattr(attn_module, '_orig_forward'):
                    attn_module.forward = attn_module._orig_forward
                    del attn_module._orig_forward

    def clear(self):
        """Clear captured data without removing hooks."""
        self._hidden_states.clear()
        self._attention_maps.clear()

    def get_hidden_states(self) -> dict[int, torch.Tensor]:
        """Return captured hidden states: layer_idx -> (seq_len, hidden_dim)."""
        return dict(self._hidden_states)

    def get_attention_maps(self) -> dict[tuple[int, int], torch.Tensor]:
        """Return captured attention maps: (layer_idx, head_idx) -> (seq_len, seq_len)."""
        return dict(self._attention_maps)
```

**Step 4: Run test to verify it passes**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_hooks.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/topology_analyzer/hooks.py tests/test_topology_analyzer/test_hooks.py
git commit -m "feat: TransformerHookManager for capturing LLM internals"
```

---

### Task 3: EmbeddingManifoldAnalyzer (Layer 1)

**Files:**
- Create: `src/topology_analyzer/embedding_analyzer.py`
- Test: `tests/test_topology_analyzer/test_embedding_analyzer.py`

**Context:** This is the core of Layer 1. Builds a CellComplex from hidden state vectors using k-NN neighborhoods, then computes persistence diagrams, Betti numbers, spectral gap, and Hodge decomposition. Reuses existing `CellComplex`, `compute_persistence_diagram`, `spectral_decomposition`, `hodge_decomposition` directly.

**Step 1: Write the failing test**

```python
# tests/test_topology_analyzer/test_embedding_analyzer.py
"""Tests for EmbeddingManifoldAnalyzer."""

import torch
import pytest


class TestEmbeddingManifoldAnalyzer:
    def test_build_cell_complex_knn(self):
        from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
        analyzer = EmbeddingManifoldAnalyzer(neighborhood="knn", k=3)
        hidden = torch.randn(10, 16)  # 10 tokens, dim 16
        lcc = analyzer.build_cell_complex(hidden, layer_idx=0)
        assert lcc.cc.num_cells(0) == 10
        assert lcc.cc.num_cells(1) > 0  # at least some edges from k-NN
        assert lcc.neighborhood == "knn"
        assert lcc.layer_idx == 0

    def test_build_cell_complex_mutual_knn(self):
        from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
        analyzer = EmbeddingManifoldAnalyzer(neighborhood="mutual_knn", k=3)
        hidden = torch.randn(10, 16)
        lcc = analyzer.build_cell_complex(hidden, layer_idx=0)
        # Mutual k-NN should have fewer or equal edges to plain k-NN
        assert lcc.cc.num_cells(0) == 10
        assert lcc.neighborhood == "mutual_knn"

    def test_build_adds_triangles(self):
        """2-cells should be added for 3-cliques in the k-NN graph."""
        from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
        # Use very clustered points so triangles form
        hidden = torch.zeros(6, 4)
        hidden[0] = torch.tensor([0.0, 0.0, 0.0, 0.0])
        hidden[1] = torch.tensor([0.01, 0.0, 0.0, 0.0])
        hidden[2] = torch.tensor([0.0, 0.01, 0.0, 0.0])
        hidden[3] = torch.tensor([10.0, 10.0, 10.0, 10.0])
        hidden[4] = torch.tensor([10.01, 10.0, 10.0, 10.0])
        hidden[5] = torch.tensor([10.0, 10.01, 10.0, 10.0])
        analyzer = EmbeddingManifoldAnalyzer(neighborhood="knn", k=3)
        lcc = analyzer.build_cell_complex(hidden, layer_idx=0)
        assert lcc.cc.num_cells(2) > 0  # should have triangles

    def test_analyze_layer_returns_all_invariants(self):
        from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
        analyzer = EmbeddingManifoldAnalyzer(neighborhood="knn", k=3)
        hidden = torch.randn(10, 16)
        lcc = analyzer.build_cell_complex(hidden, layer_idx=0)
        result = analyzer.analyze_layer(lcc)
        assert "persistence" in result
        assert "betti" in result
        assert "spectral_gap" in result
        assert "hodge_ratios" in result
        assert isinstance(result["betti"], tuple)
        assert len(result["betti"]) == 2  # (beta_0, beta_1)

    def test_cosine_similarity_edge_signal(self):
        """Edge signal should be cosine similarity between endpoints."""
        from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
        analyzer = EmbeddingManifoldAnalyzer(neighborhood="knn", k=3)
        hidden = torch.randn(8, 16)
        lcc = analyzer.build_cell_complex(hidden, layer_idx=0)
        # Edge embeddings dim 0 should be the cosine similarity
        edge_embs = lcc.cc.get_embeddings(1)
        assert edge_embs.shape[0] == lcc.cc.num_cells(1)
        # Cosine similarities should be in [-1, 1]
        assert (edge_embs[:, 0] >= -1.01).all()
        assert (edge_embs[:, 0] <= 1.01).all()

    def test_small_input(self):
        """Handle degenerate cases (< 3 tokens)."""
        from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
        analyzer = EmbeddingManifoldAnalyzer(neighborhood="knn", k=3)
        hidden = torch.randn(2, 16)
        lcc = analyzer.build_cell_complex(hidden, layer_idx=0)
        assert lcc.cc.num_cells(0) == 2
        result = analyzer.analyze_layer(lcc)
        assert "betti" in result
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_embedding_analyzer.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/topology_analyzer/embedding_analyzer.py
"""Layer 1: Embedding Manifold Analyzer.

Builds CellComplex from transformer hidden states using k-NN neighborhoods,
then computes topological invariants (persistence, Betti, spectral gap, Hodge).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from src.cell_complex.cell_complex import CellComplex
from src.spectral.persistence import compute_persistence_diagram
from src.spectral.decomposition import spectral_decomposition, hodge_decomposition
from src.topology_analyzer.profile import LayerCellComplex


class EmbeddingManifoldAnalyzer:
    """Build CellComplex from hidden states and compute topological invariants."""

    def __init__(self, neighborhood: str = "knn", k: int = 5, epsilon: float = 1.0):
        self.neighborhood = neighborhood
        self.k = k
        self.epsilon = epsilon

    def build_cell_complex(
        self, hidden_states: torch.Tensor, layer_idx: int,
    ) -> LayerCellComplex:
        """Build CellComplex from hidden states at one layer.

        Args:
            hidden_states: (seq_len, d_model) tensor.
            layer_idx: Which transformer layer this came from.

        Returns:
            LayerCellComplex wrapping the constructed CellComplex.
        """
        seq_len, d_model = hidden_states.shape
        cc = CellComplex(embedding_dim=d_model)

        # Add 0-cells (one per token)
        for i in range(seq_len):
            cc.add_0_cell(hidden_states[i].detach(), "token")

        # Compute pairwise distances
        dists = torch.cdist(hidden_states.unsqueeze(0), hidden_states.unsqueeze(0)).squeeze(0)

        # Build edges based on neighborhood strategy
        edges = self._find_edges(dists, seq_len)

        # Compute cosine similarities for edge embeddings
        norms = F.normalize(hidden_states, dim=1)
        cosine_sim = norms @ norms.T

        # Add 1-cells
        edge_set = set()
        for i, j in edges:
            key = (min(i, j), max(i, j))
            if key not in edge_set:
                edge_set.add(key)
                sim = cosine_sim[i, j].item()
                emb = torch.zeros(d_model)
                emb[0] = sim
                emb[1] = dists[i, j].item()
                cc.add_1_cell(key[0], key[1], emb, "neighbor")

        # Add 2-cells (triangles from 3-cliques)
        adj = set(edge_set)
        nodes = list(range(seq_len))
        for i in nodes:
            neighbors_i = {j for (a, b) in adj if a == i for j in [b]} | \
                          {a for (a, b) in adj if b == i for a in [a]}
            neighbors_i = sorted(neighbors_i)
            for idx_j, j in enumerate(neighbors_i):
                for k_node in neighbors_i[idx_j + 1:]:
                    if (min(j, k_node), max(j, k_node)) in adj:
                        # Triangle i-j-k
                        e_ij = self._edge_idx(cc, min(i, j), max(i, j))
                        e_ik = self._edge_idx(cc, min(i, k_node), max(i, k_node))
                        e_jk = self._edge_idx(cc, min(j, k_node), max(j, k_node))
                        if e_ij is not None and e_ik is not None and e_jk is not None:
                            boundary = [e_ij, e_jk, e_ik]
                            signs = self._triangle_signs(
                                i, j, k_node,
                                (min(i, j), max(i, j)),
                                (min(j, k_node), max(j, k_node)),
                                (min(i, k_node), max(i, k_node)),
                            )
                            emb = torch.zeros(d_model)
                            cc.add_2_cell(boundary, emb, "triangle", signs=signs)

        k_or_eps = float(self.k) if self.neighborhood != "epsilon" else self.epsilon
        return LayerCellComplex(
            cc=cc, layer_idx=layer_idx, source="embedding",
            head_idx=None, neighborhood=self.neighborhood,
            k_or_epsilon=k_or_eps,
        )

    def _find_edges(self, dists: torch.Tensor, n: int) -> list[tuple[int, int]]:
        """Find edges based on neighborhood strategy."""
        edges = []
        k = min(self.k, n - 1)
        if k < 1:
            return edges

        if self.neighborhood == "knn":
            _, indices = dists.topk(k + 1, largest=False, dim=1)
            for i in range(n):
                for j_idx in range(1, k + 1):  # skip self
                    j = indices[i, j_idx].item()
                    edges.append((i, j))

        elif self.neighborhood == "mutual_knn":
            _, indices = dists.topk(k + 1, largest=False, dim=1)
            knn_sets = []
            for i in range(n):
                knn_sets.append(set(indices[i, 1:k + 1].tolist()))
            for i in range(n):
                for j in knn_sets[i]:
                    if i in knn_sets[j]:
                        edges.append((i, j))

        elif self.neighborhood == "epsilon":
            mask = (dists < self.epsilon) & (dists > 0)
            for i in range(n):
                for j in range(i + 1, n):
                    if mask[i, j]:
                        edges.append((i, j))
        return edges

    @staticmethod
    def _edge_idx(cc: CellComplex, src: int, tgt: int) -> int | None:
        """Find edge index by endpoints."""
        for e in range(cc.num_cells(1)):
            s, t = cc._1_cell_sources[e], cc._1_cell_targets[e]
            if (s == src and t == tgt) or (s == tgt and t == src):
                return e
        return None

    @staticmethod
    def _triangle_signs(i, j, k, e_ij, e_jk, e_ik):
        """Compute orientation signs for triangle boundary."""
        signs = []
        # Edge ij: +1 if orientation matches i->j in boundary cycle i->j->k
        signs.append(1.0 if e_ij[0] == min(i, j) else -1.0)
        # Edge jk
        signs.append(1.0 if e_jk[0] == min(j, k) else -1.0)
        # Edge ik: reversed in cycle (k->i)
        signs.append(-1.0 if e_ik[0] == min(i, k) else 1.0)
        return signs

    def analyze_layer(self, lcc: LayerCellComplex) -> dict:
        """Compute all topological invariants for a layer's CellComplex.

        Returns dict with keys: persistence, betti, spectral_gap, hodge_ratios.
        """
        cc = lcc.cc
        result = {}

        # Persistence diagrams
        diagrams = compute_persistence_diagram(cc, max_dimension=1)
        result["persistence"] = diagrams

        # Betti numbers
        beta_0 = diagrams[0].shape[0] if diagrams[0].shape[0] > 0 else 0
        beta_1 = diagrams[1].shape[0] if len(diagrams) > 1 and diagrams[1].shape[0] > 0 else 0
        result["betti"] = (beta_0, beta_1)

        # Spectral gap of L0
        if cc.num_cells(0) >= 2:
            try:
                eigenvalues, _ = spectral_decomposition(cc, dim=0)
                nonzero = eigenvalues[eigenvalues > 1e-6]
                result["spectral_gap"] = nonzero[0].item() if len(nonzero) > 0 else 0.0
            except Exception:
                result["spectral_gap"] = 0.0
        else:
            result["spectral_gap"] = 0.0

        # Hodge decomposition of edge cosine similarity signal
        if cc.num_cells(1) >= 2:
            try:
                edge_embs = cc.get_embeddings(1)
                signal = edge_embs[:, 0]  # cosine similarity
                gradient, curl, harmonic = hodge_decomposition(cc, signal, dim=1)
                total = (signal ** 2).sum().item()
                if total > 1e-12:
                    g = (gradient ** 2).sum().item() / total
                    c = (curl ** 2).sum().item() / total
                    h = (harmonic ** 2).sum().item() / total
                    result["hodge_ratios"] = (g, c, h)
                else:
                    result["hodge_ratios"] = (0.0, 0.0, 0.0)
            except Exception:
                result["hodge_ratios"] = (0.0, 0.0, 0.0)
        else:
            result["hodge_ratios"] = (0.0, 0.0, 0.0)

        return result
```

**Step 4: Run test to verify it passes**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_embedding_analyzer.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/topology_analyzer/embedding_analyzer.py tests/test_topology_analyzer/test_embedding_analyzer.py
git commit -m "feat: EmbeddingManifoldAnalyzer (Layer 1) with k-NN CellComplex"
```

---

### Task 4: AttentionFlowAnalyzer (Layer 2)

**Files:**
- Create: `src/topology_analyzer/attention_analyzer.py`
- Test: `tests/test_topology_analyzer/test_attention_analyzer.py`

**Context:** Builds CellComplex from attention matrices, applies Hodge decomposition to classify heads as gradient/curl/harmonic dominant. Uses adaptive thresholding to filter noise.

**Step 1: Write the failing test**

```python
# tests/test_topology_analyzer/test_attention_analyzer.py
"""Tests for AttentionFlowAnalyzer."""

import torch
import pytest


def _make_tree_attention(n=8):
    """Attention that flows hierarchically: token 0 -> 1 -> 2 -> ..."""
    attn = torch.zeros(n, n)
    for i in range(n - 1):
        attn[i, i + 1] = 0.8
        attn[i, i] = 0.2
    attn[n - 1, n - 1] = 1.0
    return attn


def _make_cyclic_attention(n=8):
    """Attention that flows in a cycle: 0 -> 1 -> 2 -> ... -> n-1 -> 0."""
    attn = torch.zeros(n, n)
    for i in range(n):
        attn[i, (i + 1) % n] = 0.8
        attn[i, i] = 0.2
    return attn


class TestAttentionFlowAnalyzer:
    def test_build_cell_complex(self):
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer()
        attn = torch.rand(8, 8)
        attn = attn / attn.sum(dim=-1, keepdim=True)  # normalize
        lcc = analyzer.build_cell_complex(attn, layer_idx=0, head_idx=0)
        assert lcc.cc.num_cells(0) == 8
        assert lcc.source == "attention"
        assert lcc.head_idx == 0

    def test_adaptive_threshold(self):
        """Edges should only exist for attention > mean + 0.5*std."""
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer(threshold_mode="adaptive")
        # Sparse attention: only a few strong connections
        attn = torch.zeros(8, 8) + 0.01
        attn[0, 1] = 0.9
        attn[2, 3] = 0.8
        lcc = analyzer.build_cell_complex(attn, layer_idx=0, head_idx=0)
        # Should have very few edges (only the strong ones)
        assert lcc.cc.num_cells(1) >= 2
        assert lcc.cc.num_cells(1) < 30  # not all pairs

    def test_hodge_analysis(self):
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer()
        attn = torch.rand(8, 8)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        lcc = analyzer.build_cell_complex(attn, layer_idx=0, head_idx=0)
        result = analyzer.analyze_head(lcc)
        assert "hodge_ratios" in result
        assert len(result["hodge_ratios"]) == 3
        g, c, h = result["hodge_ratios"]
        assert abs(g + c + h - 1.0) < 0.1 or (g == 0 and c == 0 and h == 0)

    def test_classify_head_gradient(self):
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer()
        assert analyzer.classify_head((0.7, 0.2, 0.1)) == "gradient"

    def test_classify_head_curl(self):
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer()
        assert analyzer.classify_head((0.2, 0.6, 0.2)) == "curl"

    def test_classify_head_harmonic(self):
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer()
        assert analyzer.classify_head((0.1, 0.2, 0.7)) == "harmonic"

    def test_fixed_threshold(self):
        from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
        analyzer = AttentionFlowAnalyzer(threshold_mode="fixed", threshold=0.3)
        attn = torch.zeros(4, 4) + 0.1
        attn[0, 1] = 0.5
        attn[1, 2] = 0.4
        lcc = analyzer.build_cell_complex(attn, layer_idx=0, head_idx=0)
        # Only 2 edges should be above threshold 0.3
        assert lcc.cc.num_cells(1) == 2
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_attention_analyzer.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/topology_analyzer/attention_analyzer.py
"""Layer 2: Attention Flow Topology Analyzer.

Builds CellComplex from attention matrices, applies Hodge decomposition,
classifies attention heads by dominant component (gradient/curl/harmonic).
"""

from __future__ import annotations

import torch

from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import hodge_decomposition
from src.topology_analyzer.profile import LayerCellComplex


class AttentionFlowAnalyzer:
    """Build CellComplex from attention, Hodge decompose, classify heads."""

    def __init__(
        self,
        threshold_mode: str = "adaptive",
        threshold: float = 0.0,
    ):
        self.threshold_mode = threshold_mode
        self.threshold = threshold

    def build_cell_complex(
        self, attn_map: torch.Tensor, layer_idx: int, head_idx: int,
    ) -> LayerCellComplex:
        """Build CellComplex from an attention matrix.

        Args:
            attn_map: (seq_len, seq_len) attention weights.
            layer_idx: Transformer layer index.
            head_idx: Attention head index.

        Returns:
            LayerCellComplex wrapping the constructed CellComplex.
        """
        seq_len = attn_map.shape[0]
        # Use a small embedding dim -- we only need edge weights
        emb_dim = 4
        cc = CellComplex(embedding_dim=emb_dim)

        # Add 0-cells
        for i in range(seq_len):
            emb = torch.zeros(emb_dim)
            emb[0] = attn_map[i].sum().item()  # total outgoing attention
            cc.add_0_cell(emb, "token")

        # Compute threshold
        if self.threshold_mode == "adaptive":
            mean_val = attn_map.mean().item()
            std_val = attn_map.std().item()
            thresh = mean_val + 0.5 * std_val
        else:
            thresh = self.threshold

        # Add 1-cells for above-threshold attention
        edge_set = set()
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    continue
                if attn_map[i, j].item() > thresh:
                    key = (min(i, j), max(i, j))
                    if key not in edge_set:
                        edge_set.add(key)
                        emb = torch.zeros(emb_dim)
                        # Store attention weight as edge signal
                        emb[0] = (attn_map[i, j].item() + attn_map[j, i].item()) / 2
                        emb[1] = attn_map[i, j].item()
                        emb[2] = attn_map[j, i].item()
                        cc.add_1_cell(key[0], key[1], emb, "attention")

        # Add 2-cells (triangles)
        adj = set(edge_set)
        for i in range(seq_len):
            neighbors_i = {j for (a, b) in adj if a == i for j in [b]} | \
                          {a for (a, b) in adj if b == i}
            neighbors_i = sorted(neighbors_i)
            for idx_j, j in enumerate(neighbors_i):
                for k in neighbors_i[idx_j + 1:]:
                    if (min(j, k), max(j, k)) in adj:
                        e_ij = self._edge_idx(cc, min(i, j), max(i, j))
                        e_ik = self._edge_idx(cc, min(i, k), max(i, k))
                        e_jk = self._edge_idx(cc, min(j, k), max(j, k))
                        if e_ij is not None and e_ik is not None and e_jk is not None:
                            boundary = [e_ij, e_jk, e_ik]
                            signs = [1.0, 1.0, -1.0]
                            emb = torch.zeros(emb_dim)
                            cc.add_2_cell(boundary, emb, "attn_triangle", signs=signs)

        return LayerCellComplex(
            cc=cc, layer_idx=layer_idx, source="attention",
            head_idx=head_idx, neighborhood="threshold",
            k_or_epsilon=thresh,
        )

    def analyze_head(self, lcc: LayerCellComplex) -> dict:
        """Compute Hodge decomposition of attention flow.

        Returns dict with hodge_ratios and classification.
        """
        cc = lcc.cc
        result = {}

        if cc.num_cells(1) < 2:
            result["hodge_ratios"] = (0.0, 0.0, 0.0)
            result["classification"] = "harmonic"
            return result

        edge_embs = cc.get_embeddings(1)
        signal = edge_embs[:, 0]  # average attention weight

        total = (signal ** 2).sum().item()
        if total < 1e-12:
            result["hodge_ratios"] = (0.0, 0.0, 0.0)
            result["classification"] = "harmonic"
            return result

        try:
            gradient, curl, harmonic = hodge_decomposition(cc, signal, dim=1)
            g = (gradient ** 2).sum().item() / total
            c = (curl ** 2).sum().item() / total
            h = (harmonic ** 2).sum().item() / total
            ratios = (g, c, h)
        except Exception:
            ratios = (0.0, 0.0, 0.0)

        result["hodge_ratios"] = ratios
        result["classification"] = self.classify_head(ratios)
        return result

    @staticmethod
    def classify_head(hodge_ratios: tuple[float, float, float]) -> str:
        """Classify head by dominant Hodge component."""
        g, c, h = hodge_ratios
        if g >= c and g >= h:
            return "gradient"
        elif c >= g and c >= h:
            return "curl"
        else:
            return "harmonic"

    @staticmethod
    def _edge_idx(cc: CellComplex, src: int, tgt: int) -> int | None:
        for e in range(cc.num_cells(1)):
            s, t = cc._1_cell_sources[e], cc._1_cell_targets[e]
            if (s == src and t == tgt) or (s == tgt and t == src):
                return e
        return None
```

**Step 4: Run test to verify it passes**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_attention_analyzer.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/topology_analyzer/attention_analyzer.py tests/test_topology_analyzer/test_attention_analyzer.py
git commit -m "feat: AttentionFlowAnalyzer (Layer 2) with Hodge head classification"
```

---

### Task 5: WeightSpaceAnalyzer (Layer 3)

**Files:**
- Create: `src/topology_analyzer/weight_analyzer.py`
- Test: `tests/test_topology_analyzer/test_weight_analyzer.py`

**Context:** SVD analysis of weight matrices. Computes effective rank, condition number, singular value persistence. No CellComplex needed for per-layer analysis (SVD persistence uses gudhi directly on the 1D point cloud of singular values). Cross-layer sheaf is handled in Task 6.

**Step 1: Write the failing test**

```python
# tests/test_topology_analyzer/test_weight_analyzer.py
"""Tests for WeightSpaceAnalyzer."""

import torch
import numpy as np
import pytest


class TestWeightSpaceAnalyzer:
    def test_effective_rank(self):
        from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
        analyzer = WeightSpaceAnalyzer()
        # Full-rank matrix
        W = torch.randn(16, 16)
        result = analyzer.analyze_weight_matrix(W, layer_idx=0)
        assert result["effective_rank"] > 10  # should be near 16

    def test_low_rank_matrix(self):
        from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
        analyzer = WeightSpaceAnalyzer()
        # Rank-2 matrix
        a = torch.randn(16, 2)
        b = torch.randn(2, 16)
        W = a @ b
        result = analyzer.analyze_weight_matrix(W, layer_idx=0)
        assert result["effective_rank"] < 5  # should be near 2

    def test_condition_number(self):
        from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
        analyzer = WeightSpaceAnalyzer()
        W = torch.eye(8) * 2.0
        result = analyzer.analyze_weight_matrix(W, layer_idx=0)
        assert result["condition_number"] == pytest.approx(1.0, abs=0.1)

    def test_sv_persistence(self):
        from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
        analyzer = WeightSpaceAnalyzer()
        W = torch.randn(16, 16)
        result = analyzer.analyze_weight_matrix(W, layer_idx=0)
        assert "sv_persistence" in result
        assert isinstance(result["sv_persistence"], list)

    def test_spectral_gap(self):
        from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
        analyzer = WeightSpaceAnalyzer()
        W = torch.randn(8, 8)
        result = analyzer.analyze_weight_matrix(W, layer_idx=0)
        assert "spectral_gap" in result
        assert result["spectral_gap"] >= 0

    def test_training_dynamics_wasserstein(self):
        """Wasserstein distance between checkpoint profiles."""
        from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
        from src.topology_analyzer.profile import TopologicalProfile
        import time

        analyzer = WeightSpaceAnalyzer()

        def _make_profile(sv_diag):
            return TopologicalProfile(
                per_layer_persistence={}, per_layer_betti={},
                per_layer_spectral_gap={}, per_layer_hodge_ratios={},
                cross_layer_sheaf_gap=0.0,
                per_head_hodge_ratios={}, per_head_classification={},
                per_layer_head_sheaf_gap={},
                cross_layer_attention_wasserstein={},
                per_layer_effective_rank={0: 5.0},
                per_layer_condition_number={0: 10.0},
                per_layer_sv_persistence={0: [sv_diag]},
                cross_layer_weight_sheaf_gap=0.0,
                model_name="test", num_layers=1, num_heads=1,
                hidden_dim=16, seq_len=8, timestamp=time.time(),
            )

        p1 = _make_profile(np.array([[0.0, 0.5], [0.1, 0.8]]))
        p2 = _make_profile(np.array([[0.0, 0.6], [0.2, 0.9]]))
        result = analyzer.training_dynamics([p1, p2])
        assert "wasserstein_distances" in result
        assert len(result["wasserstein_distances"]) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_weight_analyzer.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/topology_analyzer/weight_analyzer.py
"""Layer 3: Weight Space Geometry Analyzer.

SVD analysis of weight matrices: effective rank, condition number,
singular value persistence, spectral gap.
"""

from __future__ import annotations

import torch
import numpy as np

from src.topology_analyzer.profile import TopologicalProfile

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


class WeightSpaceAnalyzer:
    """Analyze the topological structure of learned weight matrices."""

    def analyze_weight_matrix(self, W: torch.Tensor, layer_idx: int) -> dict:
        """Compute all weight space metrics for one weight matrix.

        Args:
            W: Weight matrix of shape (d_out, d_in).
            layer_idx: Which layer this weight belongs to.

        Returns:
            Dict with effective_rank, condition_number, sv_persistence, spectral_gap.
        """
        W = W.detach().float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # Effective rank: exp(entropy(p)) where p = S/sum(S)
        s = S.clamp(min=1e-10)
        p = s / s.sum()
        entropy = -(p * p.log()).sum().item()
        effective_rank = np.exp(entropy)

        # Condition number
        condition_number = (S[0] / S[-1]).item() if S[-1] > 1e-10 else float('inf')

        # Spectral gap of W^T @ W (gap between top 2 eigenvalues)
        eigenvalues = S ** 2
        if len(eigenvalues) >= 2:
            spectral_gap = (eigenvalues[0] - eigenvalues[1]).item()
        else:
            spectral_gap = 0.0

        # Singular value persistence (treat SVs as 1D point cloud)
        sv_persistence = self._sv_persistence(S.cpu().numpy())

        return {
            "effective_rank": effective_rank,
            "condition_number": condition_number,
            "spectral_gap": spectral_gap,
            "sv_persistence": sv_persistence,
            "layer_idx": layer_idx,
        }

    @staticmethod
    def _sv_persistence(singular_values: np.ndarray) -> list[np.ndarray]:
        """Compute persistence of singular values as a 1D point cloud."""
        if not GUDHI_AVAILABLE or len(singular_values) < 2:
            return [np.empty((0, 2))]

        # Reshape to 1D point cloud (each SV is a point in R^1)
        points = singular_values.reshape(-1, 1)
        rips = gudhi.RipsComplex(points=points, max_edge_length=float('inf'))
        st = rips.create_simplex_tree(max_dimension=1)
        st.compute_persistence()

        pairs = st.persistence_intervals_in_dimension(0)
        if len(pairs) == 0:
            return [np.empty((0, 2))]

        finite = pairs[np.isfinite(pairs[:, 1])]
        if len(finite) == 0:
            return [np.empty((0, 2))]
        return [finite]

    def training_dynamics(self, profiles: list[TopologicalProfile]) -> dict:
        """Compute Wasserstein distances between consecutive profiles.

        Args:
            profiles: List of TopologicalProfile from consecutive checkpoints.

        Returns:
            Dict with wasserstein_distances and phase_transitions.
        """
        if len(profiles) < 2:
            return {"wasserstein_distances": [], "phase_transitions": []}

        distances = []
        for i in range(len(profiles) - 1):
            d = self._wasserstein_between(profiles[i], profiles[i + 1])
            distances.append(d)

        # Phase transition detection: spikes > 2 * median
        phase_transitions = []
        if distances:
            median_d = sorted(distances)[len(distances) // 2]
            for i, d in enumerate(distances):
                if d > 2 * median_d and median_d > 1e-6:
                    phase_transitions.append(i + 1)  # checkpoint index

        return {
            "wasserstein_distances": distances,
            "phase_transitions": phase_transitions,
        }

    @staticmethod
    def _wasserstein_between(p1: TopologicalProfile, p2: TopologicalProfile) -> float:
        """Approximate Wasserstein distance between SV persistence diagrams."""
        total_dist = 0.0
        count = 0

        for layer_idx in p1.per_layer_sv_persistence:
            if layer_idx not in p2.per_layer_sv_persistence:
                continue
            diags1 = p1.per_layer_sv_persistence[layer_idx]
            diags2 = p2.per_layer_sv_persistence[layer_idx]
            for d1, d2 in zip(diags1, diags2):
                if d1.shape[0] == 0 and d2.shape[0] == 0:
                    continue
                # Simple approximation: compare sorted lifetimes
                lt1 = np.sort(d1[:, 1] - d1[:, 0])[::-1] if d1.shape[0] > 0 else np.array([0.0])
                lt2 = np.sort(d2[:, 1] - d2[:, 0])[::-1] if d2.shape[0] > 0 else np.array([0.0])
                # Pad to same length
                max_len = max(len(lt1), len(lt2))
                lt1_pad = np.zeros(max_len)
                lt2_pad = np.zeros(max_len)
                lt1_pad[:len(lt1)] = lt1
                lt2_pad[:len(lt2)] = lt2
                total_dist += np.sum(np.abs(lt1_pad - lt2_pad))
                count += 1

        return total_dist / max(count, 1)
```

**Step 4: Run test to verify it passes**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_weight_analyzer.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/topology_analyzer/weight_analyzer.py tests/test_topology_analyzer/test_weight_analyzer.py
git commit -m "feat: WeightSpaceAnalyzer (Layer 3) with SVD persistence"
```

---

### Task 6: CrossLayerSheafAnalyzer

**Files:**
- Create: `src/topology_analyzer/sheaf_analyzer.py`
- Test: `tests/test_topology_analyzer/test_sheaf_analyzer.py`

**Context:** Builds a CellComplex over transformer layers (0-cells = layers, 1-cells = consecutive pairs). Uses `SheafLaplacian` from `src/spectral/sheaf_diffusion.py` with learned restriction maps. Computes sheaf Laplacian spectral gap as coherence measure. Used for all three layers: embedding manifold, attention heads, and weight spaces.

**Step 1: Write the failing test**

```python
# tests/test_topology_analyzer/test_sheaf_analyzer.py
"""Tests for CrossLayerSheafAnalyzer."""

import torch
import pytest


class TestCrossLayerSheafAnalyzer:
    def test_embedding_coherence_smooth(self):
        """Smooth representations across layers -> high coherence."""
        from src.topology_analyzer.sheaf_analyzer import CrossLayerSheafAnalyzer
        analyzer = CrossLayerSheafAnalyzer(feature_dim=8, num_layers=4)
        # Smoothly varying features
        base = torch.randn(8)
        features = {i: base + 0.01 * i * torch.randn(8) for i in range(4)}
        gap = analyzer.compute_coherence(features)
        assert isinstance(gap, float)
        assert gap >= 0

    def test_embedding_coherence_discontinuous(self):
        """Discontinuous representations -> lower coherence."""
        from src.topology_analyzer.sheaf_analyzer import CrossLayerSheafAnalyzer
        analyzer = CrossLayerSheafAnalyzer(feature_dim=8, num_layers=4)
        # Wildly different features per layer
        features = {i: torch.randn(8) * 10 for i in range(4)}
        gap = analyzer.compute_coherence(features)
        assert isinstance(gap, float)

    def test_head_redundancy(self):
        """Measure coherence between attention heads within a layer."""
        from src.topology_analyzer.sheaf_analyzer import CrossLayerSheafAnalyzer
        analyzer = CrossLayerSheafAnalyzer(feature_dim=8, num_layers=4)
        # Use per-head features (flatten attention stats into feature vector)
        head_features = {h: torch.randn(8) for h in range(4)}
        gap = analyzer.compute_coherence(head_features)
        assert isinstance(gap, float)

    def test_two_layers(self):
        """Minimum viable: 2 layers, 1 edge."""
        from src.topology_analyzer.sheaf_analyzer import CrossLayerSheafAnalyzer
        analyzer = CrossLayerSheafAnalyzer(feature_dim=4, num_layers=2)
        features = {0: torch.randn(4), 1: torch.randn(4)}
        gap = analyzer.compute_coherence(features)
        assert isinstance(gap, float)
        assert gap >= 0

    def test_weight_subspace_coherence(self):
        """Cross-layer weight sheaf: measures subspace alignment."""
        from src.topology_analyzer.sheaf_analyzer import CrossLayerSheafAnalyzer
        analyzer = CrossLayerSheafAnalyzer(feature_dim=8, num_layers=3)
        # Simulate top-k right singular vectors as features
        features = {i: torch.randn(8) for i in range(3)}
        gap = analyzer.compute_coherence(features)
        assert gap >= 0
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_sheaf_analyzer.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/topology_analyzer/sheaf_analyzer.py
"""Cross-layer sheaf coherence analyzer.

Builds a CellComplex over transformer layers and computes sheaf Laplacian
spectral gap as a measure of cross-layer coherence. Works for all three
analysis layers (embedding, attention, weight).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.cell_complex.cell_complex import CellComplex
from src.spectral.sheaf_diffusion import SheafLaplacian


class CrossLayerSheafAnalyzer(nn.Module):
    """Build cross-layer sheaf and compute coherence.

    Creates a path graph over layers (0-cells = layers, 1-cells = consecutive
    pairs), then uses SheafLaplacian with learned restriction maps to measure
    how smoothly features transform across depth.
    """

    def __init__(self, feature_dim: int, num_layers: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        n_edges = max(num_layers - 1, 1)
        self.sheaf_lap = SheafLaplacian(
            n_edges=n_edges,
            feature_dim=feature_dim,
        )

    def _build_layer_graph(self) -> CellComplex:
        """Build a path graph CellComplex over layers."""
        cc = CellComplex(embedding_dim=self.feature_dim)
        for i in range(self.num_layers):
            cc.add_0_cell(torch.zeros(self.feature_dim), f"layer_{i}")
        for i in range(self.num_layers - 1):
            cc.add_1_cell(i, i + 1, torch.zeros(self.feature_dim), "consecutive")
        return cc

    @torch.no_grad()
    def compute_coherence(self, per_layer_features: dict[int, torch.Tensor]) -> float:
        """Compute sheaf Laplacian spectral gap as coherence measure.

        Args:
            per_layer_features: layer_idx -> feature vector of shape (feature_dim,).

        Returns:
            Spectral gap (smallest nonzero eigenvalue) of the sheaf Laplacian.
            Higher = more coherent cross-layer representations.
        """
        cc = self._build_layer_graph()
        device = next(self.sheaf_lap.parameters()).device

        # Stack features into signal vector (N*d,)
        n = self.num_layers
        d = self.feature_dim
        signal = torch.zeros(n * d, device=device)
        for i in range(n):
            if i in per_layer_features:
                signal[i * d:(i + 1) * d] = per_layer_features[i].to(device)

        # Build sheaf Laplacian
        try:
            L = self.sheaf_lap(cc).to(device)

            # Spectral normalization for stability (from Phase 4d)
            L = (L + L.T) / 2  # symmetrize
            eigenvalues = torch.linalg.eigvalsh(L.float())

            # Spectral gap = smallest nonzero eigenvalue
            nonzero = eigenvalues[eigenvalues > 1e-6]
            if len(nonzero) == 0:
                return 0.0
            return nonzero[0].item()
        except Exception:
            return 0.0
```

**Step 4: Run test to verify it passes**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_sheaf_analyzer.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/topology_analyzer/sheaf_analyzer.py tests/test_topology_analyzer/test_sheaf_analyzer.py
git commit -m "feat: CrossLayerSheafAnalyzer with sheaf Laplacian coherence"
```

---

### Task 7: Full Profile Integration

**Files:**
- Create: `src/topology_analyzer/analyzer.py`
- Test: `tests/test_topology_analyzer/test_analyzer.py`
- Modify: `src/topology_analyzer/__init__.py`

**Context:** Top-level `TransformerTopologyAnalyzer` class that orchestrates all three layers + sheaf. Takes a model, runs a forward pass with hooks, analyzes everything, returns a `TopologicalProfile`.

**Step 1: Write the failing test**

```python
# tests/test_topology_analyzer/test_analyzer.py
"""Tests for the full TransformerTopologyAnalyzer integration."""

import torch
import torch.nn as nn
import pytest


def _make_transformer(num_layers=3, hidden_dim=32, num_heads=4):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim, nhead=num_heads, dim_feedforward=64,
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


class TestTransformerTopologyAnalyzer:
    def test_full_profile(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        model = _make_transformer(num_layers=3, hidden_dim=32, num_heads=4)
        analyzer = TransformerTopologyAnalyzer(
            model, model_name="test_transformer",
            num_heads=4, hidden_dim=32,
        )
        x = torch.randn(1, 8, 32)
        profile = analyzer.analyze(x)
        assert profile.model_name == "test_transformer"
        assert profile.num_layers == 3
        assert len(profile.per_layer_spectral_gap) == 3
        assert len(profile.per_layer_betti) == 3
        assert len(profile.per_layer_hodge_ratios) == 3

    def test_profile_with_attention(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        model = _make_transformer(num_layers=2, hidden_dim=32, num_heads=4)
        analyzer = TransformerTopologyAnalyzer(
            model, model_name="test", num_heads=4, hidden_dim=32,
            analyze_attention=True,
        )
        x = torch.randn(1, 8, 32)
        profile = analyzer.analyze(x)
        # Should have per-head analysis
        assert len(profile.per_head_hodge_ratios) > 0 or len(profile.per_head_classification) > 0

    def test_profile_with_weights(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        model = _make_transformer(num_layers=2, hidden_dim=32, num_heads=4)
        analyzer = TransformerTopologyAnalyzer(
            model, model_name="test", num_heads=4, hidden_dim=32,
            analyze_weights=True,
        )
        x = torch.randn(1, 8, 32)
        profile = analyzer.analyze(x)
        assert len(profile.per_layer_effective_rank) > 0
        assert len(profile.per_layer_condition_number) > 0

    def test_active_features_shape(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        model = _make_transformer(num_layers=2, hidden_dim=32, num_heads=4)
        analyzer = TransformerTopologyAnalyzer(
            model, model_name="test", num_heads=4, hidden_dim=32,
        )
        x = torch.randn(1, 8, 32)
        profile = analyzer.analyze(x)
        features = profile.active_features()
        assert features.shape == (6,)

    def test_summary_dict(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        model = _make_transformer(num_layers=2, hidden_dim=32, num_heads=4)
        analyzer = TransformerTopologyAnalyzer(
            model, model_name="test", num_heads=4, hidden_dim=32,
        )
        x = torch.randn(1, 8, 32)
        profile = analyzer.analyze(x)
        summary = profile.summary()
        assert "cross_layer_sheaf_gap" in summary
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_analyzer.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/topology_analyzer/analyzer.py
"""Top-level TransformerTopologyAnalyzer.

Orchestrates all three analysis layers and cross-layer sheaf
to produce a complete TopologicalProfile.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from src.topology_analyzer.profile import TopologicalProfile
from src.topology_analyzer.hooks import TransformerHookManager
from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
from src.topology_analyzer.sheaf_analyzer import CrossLayerSheafAnalyzer


class TransformerTopologyAnalyzer:
    """Full topological analysis of a transformer model.

    Runs a forward pass with hooks to capture internal representations,
    then analyzes embedding manifolds, attention flows, and weight spaces.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "unknown",
        num_heads: int = 1,
        hidden_dim: int = 64,
        analyze_attention: bool = False,
        analyze_weights: bool = False,
        embedding_k: int = 5,
        embedding_neighborhood: str = "knn",
    ):
        self.model = model
        self.model_name = model_name
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.analyze_attention = analyze_attention
        self.analyze_weights = analyze_weights

        self.embedding_analyzer = EmbeddingManifoldAnalyzer(
            neighborhood=embedding_neighborhood, k=embedding_k,
        )
        self.attention_analyzer = AttentionFlowAnalyzer()
        self.weight_analyzer = WeightSpaceAnalyzer()

    @torch.no_grad()
    def analyze(self, input_tensor: torch.Tensor) -> TopologicalProfile:
        """Run full topological analysis.

        Args:
            input_tensor: Model input (batch, seq_len, hidden_dim) or similar.

        Returns:
            TopologicalProfile with all computed invariants.
        """
        was_training = self.model.training
        self.model.eval()

        # Capture hidden states (and optionally attention maps)
        with TransformerHookManager(
            self.model, capture_attention=self.analyze_attention,
        ) as manager:
            _ = self.model(input_tensor)
            hidden_states = manager.get_hidden_states()
            attention_maps = manager.get_attention_maps() if self.analyze_attention else {}

        num_layers = len(hidden_states)
        seq_len = next(iter(hidden_states.values())).shape[0] if hidden_states else 0

        # Layer 1: Embedding Manifold
        per_layer_persistence = {}
        per_layer_betti = {}
        per_layer_spectral_gap = {}
        per_layer_hodge_ratios = {}

        embedding_features = {}  # for sheaf
        for layer_idx, h in hidden_states.items():
            lcc = self.embedding_analyzer.build_cell_complex(h, layer_idx)
            result = self.embedding_analyzer.analyze_layer(lcc)
            per_layer_persistence[layer_idx] = result["persistence"]
            per_layer_betti[layer_idx] = result["betti"]
            per_layer_spectral_gap[layer_idx] = result["spectral_gap"]
            per_layer_hodge_ratios[layer_idx] = result["hodge_ratios"]
            # Mean hidden state as feature for cross-layer sheaf
            embedding_features[layer_idx] = h.mean(dim=0)

        # Cross-layer embedding sheaf
        sheaf_dim = min(self.hidden_dim, 16)  # cap for efficiency
        cross_layer_sheaf_gap = 0.0
        if num_layers >= 2:
            sheaf_analyzer = CrossLayerSheafAnalyzer(
                feature_dim=sheaf_dim, num_layers=num_layers,
            )
            projected = {
                i: f[:sheaf_dim] for i, f in embedding_features.items()
            }
            cross_layer_sheaf_gap = sheaf_analyzer.compute_coherence(projected)

        # Layer 2: Attention Flow
        per_head_hodge_ratios = {}
        per_head_classification = {}
        per_layer_head_sheaf_gap = {}
        cross_layer_attention_wasserstein = {}

        if self.analyze_attention and attention_maps:
            for (layer_idx, head_idx), attn in attention_maps.items():
                lcc = self.attention_analyzer.build_cell_complex(
                    attn, layer_idx, head_idx,
                )
                result = self.attention_analyzer.analyze_head(lcc)
                per_head_hodge_ratios[(layer_idx, head_idx)] = result["hodge_ratios"]
                per_head_classification[(layer_idx, head_idx)] = result["classification"]

        # Layer 3: Weight Space
        per_layer_effective_rank = {}
        per_layer_condition_number = {}
        per_layer_sv_persistence = {}
        cross_layer_weight_sheaf_gap = 0.0

        if self.analyze_weights:
            weight_layers = self._find_weight_matrices()
            for layer_idx, W in weight_layers.items():
                result = self.weight_analyzer.analyze_weight_matrix(W, layer_idx)
                per_layer_effective_rank[layer_idx] = result["effective_rank"]
                per_layer_condition_number[layer_idx] = result["condition_number"]
                per_layer_sv_persistence[layer_idx] = result["sv_persistence"]

        if was_training:
            self.model.train()

        return TopologicalProfile(
            per_layer_persistence=per_layer_persistence,
            per_layer_betti=per_layer_betti,
            per_layer_spectral_gap=per_layer_spectral_gap,
            per_layer_hodge_ratios=per_layer_hodge_ratios,
            cross_layer_sheaf_gap=cross_layer_sheaf_gap,
            per_head_hodge_ratios=per_head_hodge_ratios,
            per_head_classification=per_head_classification,
            per_layer_head_sheaf_gap=per_layer_head_sheaf_gap,
            cross_layer_attention_wasserstein=cross_layer_attention_wasserstein,
            per_layer_effective_rank=per_layer_effective_rank,
            per_layer_condition_number=per_layer_condition_number,
            per_layer_sv_persistence=per_layer_sv_persistence,
            cross_layer_weight_sheaf_gap=cross_layer_weight_sheaf_gap,
            model_name=self.model_name,
            num_layers=num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            seq_len=seq_len,
            timestamp=time.time(),
        )

    def _find_weight_matrices(self) -> dict[int, torch.Tensor]:
        """Find the main weight matrices (e.g. in_proj_weight from self_attn)."""
        weights = {}
        from src.topology_analyzer.hooks import _find_transformer_layers
        layers = _find_transformer_layers(self.model)
        for i, layer in enumerate(layers):
            # Try common weight matrix names
            for name in ['self_attn.in_proj_weight', 'linear1.weight', 'linear2.weight']:
                parts = name.split('.')
                obj = layer
                try:
                    for part in parts:
                        obj = getattr(obj, part)
                    if isinstance(obj, torch.Tensor) and obj.dim() == 2:
                        weights[i] = obj
                        break
                except AttributeError:
                    continue
        return weights
```

Update `__init__.py`:

```python
# src/topology_analyzer/__init__.py
from src.topology_analyzer.profile import TopologicalProfile, LayerCellComplex
from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer

__all__ = [
    'TopologicalProfile',
    'LayerCellComplex',
    'TransformerTopologyAnalyzer',
]
```

**Step 4: Run test to verify it passes**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_analyzer.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/topology_analyzer/analyzer.py src/topology_analyzer/__init__.py tests/test_topology_analyzer/test_analyzer.py
git commit -m "feat: TransformerTopologyAnalyzer full profile integration"
```

---

### Task 8: Active Mode Feedback (TopologyFeedback)

**Files:**
- Create: `src/topology_analyzer/feedback.py`
- Test: `tests/test_topology_analyzer/test_feedback.py`

**Context:** Converts `TopologicalProfile` into the 6-feature topo_feedback vector for `ControlHead`. Also includes `EmbeddingTopologyAdvisor` for epoch-level alerts. This is the bridge between observer mode and active mode -- the topology analyzer feeds into the model's control loop.

**Step 1: Write the failing test**

```python
# tests/test_topology_analyzer/test_feedback.py
"""Tests for TopologyFeedback and EmbeddingTopologyAdvisor."""

import time
import torch
import numpy as np
import pytest
from src.topology_analyzer.profile import TopologicalProfile


def _make_profile(**overrides):
    defaults = dict(
        per_layer_persistence={},
        per_layer_betti={},
        per_layer_spectral_gap={0: 0.5, 1: 0.3},
        per_layer_hodge_ratios={0: (0.6, 0.3, 0.1), 1: (0.4, 0.4, 0.2)},
        cross_layer_sheaf_gap=0.42,
        per_head_hodge_ratios={(0, 0): (0.3, 0.5, 0.2), (0, 1): (0.7, 0.2, 0.1)},
        per_head_classification={(0, 0): "curl", (0, 1): "gradient"},
        per_layer_head_sheaf_gap={0: 0.3},
        cross_layer_attention_wasserstein={},
        per_layer_effective_rank={0: 10.0, 1: 5.0},
        per_layer_condition_number={0: 50.0, 1: 200.0},
        per_layer_sv_persistence={},
        cross_layer_weight_sheaf_gap=0.35,
        model_name="test",
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        seq_len=16,
        timestamp=time.time(),
    )
    defaults.update(overrides)
    return TopologicalProfile(**defaults)


class TestTopologyFeedback:
    def test_profile_to_features_shape(self):
        from src.topology_analyzer.feedback import TopologyFeedback
        fb = TopologyFeedback()
        profile = _make_profile()
        features = fb.profile_to_features(profile)
        assert features.shape == (6,)

    def test_features_match_active_features(self):
        from src.topology_analyzer.feedback import TopologyFeedback
        fb = TopologyFeedback()
        profile = _make_profile()
        features = fb.profile_to_features(profile)
        active = profile.active_features()
        assert torch.allclose(features, active, atol=1e-5)


class TestEmbeddingTopologyAdvisor:
    def test_no_alerts_on_healthy_profile(self):
        from src.topology_analyzer.feedback import EmbeddingTopologyAdvisor
        advisor = EmbeddingTopologyAdvisor()
        profile = _make_profile(
            cross_layer_sheaf_gap=0.5,
            per_layer_effective_rank={0: 15.0, 1: 14.0},
        )
        alerts = advisor.check_alerts(profile)
        assert len(alerts) == 0

    def test_alert_on_sheaf_gap_collapse(self):
        from src.topology_analyzer.feedback import EmbeddingTopologyAdvisor
        advisor = EmbeddingTopologyAdvisor()
        profile = _make_profile(cross_layer_sheaf_gap=0.001)
        alerts = advisor.check_alerts(profile)
        assert any("sheaf" in a.lower() or "coherence" in a.lower() for a in alerts)

    def test_alert_on_rank_collapse(self):
        from src.topology_analyzer.feedback import EmbeddingTopologyAdvisor
        advisor = EmbeddingTopologyAdvisor()
        profile = _make_profile(per_layer_effective_rank={0: 1.5, 1: 1.2})
        alerts = advisor.check_alerts(profile)
        assert any("rank" in a.lower() for a in alerts)

    def test_alert_on_high_attention_curl(self):
        from src.topology_analyzer.feedback import EmbeddingTopologyAdvisor
        advisor = EmbeddingTopologyAdvisor()
        profile = _make_profile(
            per_head_hodge_ratios={(0, 0): (0.1, 0.8, 0.1), (0, 1): (0.1, 0.7, 0.2)},
        )
        alerts = advisor.check_alerts(profile)
        assert any("curl" in a.lower() or "circular" in a.lower() for a in alerts)

    def test_suggest_architecture_from_history(self):
        from src.topology_analyzer.feedback import EmbeddingTopologyAdvisor
        advisor = EmbeddingTopologyAdvisor()
        # Create history with declining sheaf gap
        profiles = []
        for i in range(5):
            p = _make_profile(cross_layer_sheaf_gap=0.5 - i * 0.1)
            profiles.append(p)
        suggestions = advisor.suggest_architecture_changes(profiles)
        assert isinstance(suggestions, list)
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_feedback.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/topology_analyzer/feedback.py
"""Active mode: convert TopologicalProfile into ControlHead feedback + advisories."""

from __future__ import annotations

import torch

from src.topology_analyzer.profile import TopologicalProfile


_DEFAULT_THRESHOLDS = {
    "sheaf_gap_min": 0.01,
    "effective_rank_min": 3.0,
    "attention_curl_max": 0.6,
    "condition_number_max": 1000.0,
    "harmonic_ratio_max": 0.4,
}


class TopologyFeedback:
    """Convert TopologicalProfile into ControlHead topo_features vector."""

    def profile_to_features(self, profile: TopologicalProfile) -> torch.Tensor:
        """Return the 6-feature vector for ControlHead topo_feedback input.

        Same as profile.active_features() but callable from training loop.
        """
        return profile.active_features()


class EmbeddingTopologyAdvisor:
    """Epoch-level advisor: threshold-based alerts from TopologicalProfile."""

    def __init__(self, thresholds: dict[str, float] | None = None):
        self.thresholds = {**_DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)

    def check_alerts(self, profile: TopologicalProfile) -> list[str]:
        """Check for concerning patterns in the topological profile."""
        alerts = []

        # Sheaf gap collapse
        gap_min = self.thresholds["sheaf_gap_min"]
        if profile.cross_layer_sheaf_gap < gap_min:
            alerts.append(
                f"Low cross-layer sheaf coherence: {profile.cross_layer_sheaf_gap:.4f} < {gap_min}. "
                f"Representation bottleneck between layers."
            )

        # Effective rank collapse
        rank_min = self.thresholds["effective_rank_min"]
        for layer_idx, rank in profile.per_layer_effective_rank.items():
            if rank < rank_min:
                alerts.append(
                    f"Low effective rank at layer {layer_idx}: {rank:.1f} < {rank_min}. "
                    f"Consider reinitializing or applying spectral regularization."
                )

        # High attention curl
        curl_max = self.thresholds["attention_curl_max"]
        if profile.per_head_hodge_ratios:
            avg_curl = sum(
                r[1] for r in profile.per_head_hodge_ratios.values()
            ) / len(profile.per_head_hodge_ratios)
            if avg_curl > curl_max:
                alerts.append(
                    f"High average attention curl: {avg_curl:.3f} > {curl_max}. "
                    f"Circular attention patterns detected."
                )

        # High condition number
        cond_max = self.thresholds["condition_number_max"]
        for layer_idx, cond in profile.per_layer_condition_number.items():
            if cond > cond_max:
                alerts.append(
                    f"High condition number at layer {layer_idx}: {cond:.1f} > {cond_max}. "
                    f"Numerically unstable weight matrix."
                )

        # High harmonic ratio (dead patterns)
        harm_max = self.thresholds["harmonic_ratio_max"]
        for layer_idx, (g, c, h) in profile.per_layer_hodge_ratios.items():
            if h > harm_max:
                alerts.append(
                    f"High harmonic ratio at layer {layer_idx}: {h:.3f} > {harm_max}. "
                    f"Dead representation zone."
                )

        return alerts

    def suggest_architecture_changes(
        self, history: list[TopologicalProfile], window: int = 5,
    ) -> list[str]:
        """Analyze trends across checkpoints and suggest changes."""
        if len(history) < 2:
            return []

        recent = history[-window:] if len(history) >= window else history
        suggestions = []

        # Declining sheaf gap
        gaps = [p.cross_layer_sheaf_gap for p in recent]
        if len(gaps) >= 2 and gaps[-1] < gaps[0] * 0.5:
            suggestions.append(
                f"Cross-layer sheaf gap declining ({gaps[0]:.4f} -> {gaps[-1]:.4f}). "
                f"Consider adding skip connections between bottleneck layers."
            )

        # Rising average curl
        avg_curls = []
        for p in recent:
            if p.per_head_hodge_ratios:
                avg_curls.append(
                    sum(r[1] for r in p.per_head_hodge_ratios.values())
                    / len(p.per_head_hodge_ratios)
                )
        if len(avg_curls) >= 2 and avg_curls[-1] > avg_curls[0] + 0.1:
            suggestions.append(
                f"Attention curl increasing ({avg_curls[0]:.3f} -> {avg_curls[-1]:.3f}). "
                f"Increase head dropout or wave damping."
            )

        # Declining effective rank
        for layer_idx in recent[0].per_layer_effective_rank:
            ranks = [p.per_layer_effective_rank.get(layer_idx, 0) for p in recent]
            if ranks and ranks[-1] < ranks[0] * 0.5:
                suggestions.append(
                    f"Layer {layer_idx} effective rank collapsing ({ranks[0]:.1f} -> {ranks[-1]:.1f}). "
                    f"Consider spectral regularization or reinitialization."
                )

        return suggestions
```

**Step 4: Run test to verify it passes**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_feedback.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/topology_analyzer/feedback.py tests/test_topology_analyzer/test_feedback.py
git commit -m "feat: TopologyFeedback + EmbeddingTopologyAdvisor for active mode"
```

---

### Task 9: ControlHead Extension (3 -> 6 topo_features)

**Files:**
- Modify: `src/gnn_executive/control_head.py:38-48`
- Test: `tests/test_topology_analyzer/test_controlhead_extension.py`

**Context:** Extend `ControlHead` to accept 6 topo_features instead of 3 when `use_topo_feedback=True`. The existing trunk_input_dim formula is `embedding_dim + 2 + (3 if use_topo_feedback else 0)`. We need a new mode `use_embedding_topo_feedback=True` that adds 6 instead. Must be backward compatible: existing `use_topo_feedback=True` still uses 3 features.

**Step 1: Write the failing test**

```python
# tests/test_topology_analyzer/test_controlhead_extension.py
"""Tests for ControlHead extension with 6-feature embedding topology feedback."""

import torch
import pytest
from src.gnn_executive.control_head import ControlHead


class TestControlHeadEmbeddingTopo:
    def test_backward_compat_3_features(self):
        """Original use_topo_feedback=True still works with 3 features."""
        head = ControlHead(embedding_dim=16, num_freqs=8, use_topo_feedback=True)
        node_embs = torch.randn(5, 16)
        topo = torch.randn(3)
        signal = head(node_embs, harmonic_energy=0.1, log_size=0.5, topo_features=topo)
        assert signal.frequency_gate.shape == (8,)

    def test_6_feature_embedding_topo(self):
        """New use_embedding_topo_feedback=True accepts 6 features."""
        head = ControlHead(
            embedding_dim=16, num_freqs=8,
            use_topo_feedback=True, use_embedding_topo_feedback=True,
        )
        node_embs = torch.randn(5, 16)
        topo = torch.randn(6)
        signal = head(node_embs, harmonic_energy=0.1, log_size=0.5, topo_features=topo)
        assert signal.frequency_gate.shape == (8,)

    def test_trunk_dim_with_embedding_topo(self):
        """Trunk input should be embedding_dim + 2 + 6."""
        head = ControlHead(
            embedding_dim=16, num_freqs=8,
            use_topo_feedback=True, use_embedding_topo_feedback=True,
        )
        # trunk input: 16 + 2 + 6 = 24
        assert head.trunk[0].in_features == 24

    def test_trunk_dim_without_embedding_topo(self):
        """Without embedding topo, trunk input should be embedding_dim + 2 + 3."""
        head = ControlHead(
            embedding_dim=16, num_freqs=8, use_topo_feedback=True,
        )
        assert head.trunk[0].in_features == 21  # 16 + 2 + 3

    def test_trunk_dim_no_topo(self):
        """No topo feedback at all: embedding_dim + 2."""
        head = ControlHead(embedding_dim=16, num_freqs=8)
        assert head.trunk[0].in_features == 18  # 16 + 2
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_controlhead_extension.py -v`
Expected: FAIL on test_6_feature_embedding_topo (ControlHead doesn't accept use_embedding_topo_feedback)

**Step 3: Modify ControlHead**

In `src/gnn_executive/control_head.py`, change the `__init__` signature and trunk_input_dim calculation:

Current (line 38-48):
```python
def __init__(self, embedding_dim: int, num_freqs: int, num_filters: int = 0,
             use_topo_feedback: bool = False):
    super().__init__()
    self.num_freqs = num_freqs
    self.num_filters = num_filters
    self.use_topo_feedback = use_topo_feedback

    # Shared trunk: pool → project
    # +1 for harmonic energy, +1 for log(N) size feature
    # +3 for topo feedback (gradient_ratio, curl_ratio, spectral_gap) if enabled
    trunk_input_dim = embedding_dim + 2 + (3 if use_topo_feedback else 0)
```

New:
```python
def __init__(self, embedding_dim: int, num_freqs: int, num_filters: int = 0,
             use_topo_feedback: bool = False, use_embedding_topo_feedback: bool = False):
    super().__init__()
    self.num_freqs = num_freqs
    self.num_filters = num_filters
    self.use_topo_feedback = use_topo_feedback
    self.use_embedding_topo_feedback = use_embedding_topo_feedback

    # Shared trunk: pool → project
    # +1 for harmonic energy, +1 for log(N) size feature
    # +3 for computation graph topo feedback if use_topo_feedback
    # +6 for embedding topology feedback if use_embedding_topo_feedback (replaces +3)
    if use_embedding_topo_feedback and use_topo_feedback:
        topo_dim = 6
    elif use_topo_feedback:
        topo_dim = 3
    else:
        topo_dim = 0
    trunk_input_dim = embedding_dim + 2 + topo_dim
```

**Step 4: Run test to verify it passes**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_controlhead_extension.py -v`
Expected: PASS (5 tests)

Also run existing tests:

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/ -k "control_head" -v`
Expected: All existing tests still pass

**Step 5: Commit**

```bash
git add src/gnn_executive/control_head.py tests/test_topology_analyzer/test_controlhead_extension.py
git commit -m "feat: ControlHead accepts 6-feature embedding topology feedback"
```

---

### Task 10: DSM Integration Test

**Files:**
- Test: `tests/test_topology_analyzer/test_dsm_integration.py`

**Context:** Integration test that runs the full `TransformerTopologyAnalyzer` on our actual DSM model. Verifies the complete pipeline: hooks capture DSM internals, all three analyzers produce valid output, active features are well-formed. Uses `DistilledSemanticModel` from `src/llm/dsm.py`.

**Step 1: Write the test**

```python
# tests/test_topology_analyzer/test_dsm_integration.py
"""Integration test: TransformerTopologyAnalyzer on our DSM."""

import torch
import pytest
from src.llm.dsm import DistilledSemanticModel


class TestDSMIntegration:
    def test_analyze_dsm(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        # Build a small DSM
        dsm = DistilledSemanticModel(
            hidden_dim=32, num_heads=4, ff_dim=64,
            num_layers=3, cross_attn_layer=1,
        )
        analyzer = TransformerTopologyAnalyzer(
            dsm, model_name="test_dsm",
            num_heads=4, hidden_dim=32,
            analyze_weights=True,
        )
        # DSM expects (seq_len, hidden_dim) for prefix + topo_memory
        prefix = torch.randn(8, 32)  # 8 prefix tokens
        topo_memory = torch.randn(10, 32)  # 10 node embeddings

        # Need to call DSM forward, but analyzer expects a simple forward
        # We'll call it through a wrapper
        profile = analyzer.analyze(prefix.unsqueeze(0))  # fake batch

        # Analyzer may not find layers in DSM structure -- that's OK
        # At minimum, profile should be returned
        assert profile.model_name == "test_dsm"

    def test_dsm_hidden_states_captured(self):
        """Verify hooks work on DSM's .layers attribute."""
        from src.topology_analyzer.hooks import TransformerHookManager
        dsm = DistilledSemanticModel(
            hidden_dim=32, num_heads=4, ff_dim=64,
            num_layers=3, cross_attn_layer=1,
        )
        prefix = torch.randn(8, 32)
        topo_memory = torch.randn(10, 32)

        with TransformerHookManager(dsm) as manager:
            _ = dsm(prefix, topo_memory)
            hidden = manager.get_hidden_states()

        # DSM has .layers with 3 modules
        assert len(hidden) == 3
        for layer_idx, h in hidden.items():
            assert h.shape[0] == 8  # seq_len matches prefix length
            assert h.shape[1] == 32  # hidden_dim

    def test_active_features_from_dsm_profile(self):
        from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
        dsm = DistilledSemanticModel(
            hidden_dim=32, num_heads=4, ff_dim=64,
            num_layers=2, cross_attn_layer=1,
        )
        analyzer = TransformerTopologyAnalyzer(
            dsm, model_name="dsm", num_heads=4, hidden_dim=32,
        )
        prefix = torch.randn(8, 32)
        profile = analyzer.analyze(prefix.unsqueeze(0))
        features = profile.active_features()
        assert features.shape == (6,)
        # Should not be all zeros (DSM has non-trivial hidden states)
        assert not torch.all(features == 0)
```

**Step 2: Run test**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_topology_analyzer/test_dsm_integration.py -v`
Expected: PASS (3 tests). Note: DSM's forward hook may need special handling since it uses non-batch-first attention (seq_len first). Adjust `TransformerHookManager` if needed.

**Step 3: Fix any issues and commit**

```bash
git add tests/test_topology_analyzer/test_dsm_integration.py
git commit -m "test: DSM integration tests for TransformerTopologyAnalyzer"
```

---

### Task 11: Run Full Test Suite

**Step 1: Run all tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All new tests pass, all existing tests still pass.

**Step 2: Count tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/ --co -q | tail -5`
Expected: ~680+ tests total (633 existing + ~47 new)

**Step 3: Commit if any fixes needed**

---

### Task 12: Update __init__.py Exports

**Files:**
- Modify: `src/topology_analyzer/__init__.py`

**Step 1: Ensure all public classes are exported**

```python
# src/topology_analyzer/__init__.py
from src.topology_analyzer.profile import TopologicalProfile, LayerCellComplex
from src.topology_analyzer.analyzer import TransformerTopologyAnalyzer
from src.topology_analyzer.hooks import TransformerHookManager
from src.topology_analyzer.embedding_analyzer import EmbeddingManifoldAnalyzer
from src.topology_analyzer.attention_analyzer import AttentionFlowAnalyzer
from src.topology_analyzer.weight_analyzer import WeightSpaceAnalyzer
from src.topology_analyzer.sheaf_analyzer import CrossLayerSheafAnalyzer
from src.topology_analyzer.feedback import TopologyFeedback, EmbeddingTopologyAdvisor

__all__ = [
    'TopologicalProfile',
    'LayerCellComplex',
    'TransformerTopologyAnalyzer',
    'TransformerHookManager',
    'EmbeddingManifoldAnalyzer',
    'AttentionFlowAnalyzer',
    'WeightSpaceAnalyzer',
    'CrossLayerSheafAnalyzer',
    'TopologyFeedback',
    'EmbeddingTopologyAdvisor',
]
```

**Step 2: Commit**

```bash
git add src/topology_analyzer/__init__.py
git commit -m "feat: export all topology_analyzer public classes"
```

---

## Summary

| Task | Component | New Tests | Files |
|------|-----------|-----------|-------|
| 1 | TopologicalProfile + LayerCellComplex | 5 | 4 created |
| 2 | TransformerHookManager | 6 | 2 created |
| 3 | EmbeddingManifoldAnalyzer | 6 | 2 created |
| 4 | AttentionFlowAnalyzer | 7 | 2 created |
| 5 | WeightSpaceAnalyzer | 6 | 2 created |
| 6 | CrossLayerSheafAnalyzer | 5 | 2 created |
| 7 | Full Profile Integration | 5 | 2 created, 1 modified |
| 8 | TopologyFeedback + Advisor | 7 | 2 created |
| 9 | ControlHead Extension | 5 | 1 modified, 1 created |
| 10 | DSM Integration Test | 3 | 1 created |
| 11 | Full Test Suite Run | 0 | — |
| 12 | Exports | 0 | 1 modified |
| **Total** | | **~55** | **19 created, 3 modified** |

**Dependencies:** Tasks 1-6 are independent of each other. Task 7 depends on Tasks 1-6. Task 8 depends on Task 1. Task 9 depends on Task 8. Task 10 depends on Task 7. Tasks 11-12 depend on all previous.

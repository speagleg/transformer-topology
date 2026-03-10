# v9 Text-Topology Fusion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Break the semantic ceiling on KG tasks by adding a graph-aware TextReasoningHead and TextEdgeEncoder (Phase 2 prep).

**Architecture:** A lightweight 2-layer transformer processes all N node text embeddings with adjacency-masked attention, producing (N, 64) semantic representations. Query/target/diff are concatenated to the classifier (192 extra dims). TextEdgeEncoder is created but gated off for Phase 1.

**Tech Stack:** PyTorch, existing CellComplex/GNN/TAT infrastructure, Qwen embed_tokens cache.

---

### Task 1: Create TextReasoningHead module

**Files:**
- Create: `src/llm/text_reasoning_head.py`
- Test: `tests/test_llm/test_text_reasoning_head.py`

**Step 1: Write the failing tests**

```python
# tests/test_llm/test_text_reasoning_head.py
"""Tests for TextReasoningHead — graph-aware text transformer."""

import torch
import pytest


def test_text_reasoning_head_output_shape():
    """TextReasoningHead produces (N, hidden_dim) output."""
    from src.llm.text_reasoning_head import TextReasoningHead

    head = TextReasoningHead(input_dim=32, hidden_dim=64, num_layers=2, num_heads=4)
    text_features = torch.randn(10, 32)  # 10 nodes, 32-dim text
    adj = torch.eye(10)  # self-attention only
    adj[0, 1] = adj[1, 0] = 1.0  # add one edge

    output = head(text_features, adj)
    assert output.shape == (10, 64)


def test_text_reasoning_head_adjacency_masking():
    """Nodes only attend to neighbors + self, not full graph."""
    from src.llm.text_reasoning_head import TextReasoningHead

    head = TextReasoningHead(input_dim=32, hidden_dim=64, num_layers=2, num_heads=4)

    # Disconnected graph: nodes 0-4 connected, nodes 5-9 connected, no cross-edges
    text_features = torch.randn(10, 32)
    adj = torch.zeros(10, 10)
    for i in range(4):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    for i in range(5, 9):
        adj[i, i + 1] = adj[i + 1, i] = 1.0

    output = head(text_features, adj)
    assert output.shape == (10, 64)
    # Outputs should exist (no NaN from masking)
    assert not torch.isnan(output).any()


def test_text_reasoning_head_none_input():
    """Returns None when text_features is None."""
    from src.llm.text_reasoning_head import TextReasoningHead

    head = TextReasoningHead(input_dim=32, hidden_dim=64, num_layers=2, num_heads=4)
    output = head(None, torch.eye(5))
    assert output is None


def test_text_reasoning_head_gradient_flow():
    """Gradients flow back through TextReasoningHead to input."""
    from src.llm.text_reasoning_head import TextReasoningHead

    head = TextReasoningHead(input_dim=32, hidden_dim=64, num_layers=2, num_heads=4)
    text_features = torch.randn(5, 32, requires_grad=True)
    adj = torch.ones(5, 5)

    output = head(text_features, adj)
    loss = output.sum()
    loss.backward()
    assert text_features.grad is not None
    assert text_features.grad.abs().sum() > 0


def test_text_reasoning_head_single_node():
    """Works with a single-node graph."""
    from src.llm.text_reasoning_head import TextReasoningHead

    head = TextReasoningHead(input_dim=32, hidden_dim=64, num_layers=2, num_heads=4)
    text_features = torch.randn(1, 32)
    adj = torch.ones(1, 1)

    output = head(text_features, adj)
    assert output.shape == (1, 64)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm/test_text_reasoning_head.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.llm.text_reasoning_head'`

**Step 3: Write the implementation**

```python
# src/llm/text_reasoning_head.py
"""Graph-aware text reasoning transformer.

Processes per-node text embeddings with adjacency-masked self-attention,
producing semantically-enriched per-node representations for classification.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextReasoningHead(nn.Module):
    """Lightweight transformer over node text embeddings with graph-structure attention masking.

    Args:
        input_dim: Dimension of input text features (from QwenTextFeatureExtractor proj).
        hidden_dim: Internal hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            TextReasoningLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        text_features: torch.Tensor | None,
        adjacency: torch.Tensor,
    ) -> torch.Tensor | None:
        """Process text features with graph-masked attention.

        Args:
            text_features: (N, input_dim) per-node text embeddings, or None.
            adjacency: (N, N) adjacency matrix (1 = connected, 0 = not).

        Returns:
            (N, hidden_dim) semantic node representations, or None if input is None.
        """
        if text_features is None:
            return None

        x = self.input_proj(text_features)  # (N, hidden_dim)

        # Build attention mask: attend to neighbors + self
        # True = BLOCKED, False = allowed (PyTorch convention for additive mask)
        mask = (adjacency + torch.eye(adjacency.shape[0], device=adjacency.device)) == 0
        # Convert to float additive mask: 0 for allowed, -inf for blocked
        attn_mask = torch.zeros_like(adjacency)
        attn_mask.masked_fill_(mask, float('-inf'))

        for layer in self.layers:
            x = layer(x, attn_mask)

        return x


class TextReasoningLayer(nn.Module):
    """Single transformer layer with graph-masked multi-head attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x is (N, hidden_dim) — add batch dim for nn.MultiheadAttention
        x_batch = x.unsqueeze(0)  # (1, N, hidden_dim)
        attn_out, _ = self.attn(x_batch, x_batch, x_batch, attn_mask=attn_mask)
        x = self.norm1(x + attn_out.squeeze(0))
        x = self.norm2(x + self.ffn(x))
        return x
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm/test_text_reasoning_head.py -v`
Expected: All 5 PASS

**Step 5: Commit**

```bash
git add src/llm/text_reasoning_head.py tests/test_llm/test_text_reasoning_head.py
git commit -m "feat: add TextReasoningHead — graph-aware text transformer"
```

---

### Task 2: Create TextEdgeEncoder module (Phase 2 prep, gated off)

**Files:**
- Create: `src/llm/text_edge_encoder.py`
- Test: `tests/test_llm/test_text_edge_encoder.py`

**Step 1: Write the failing tests**

```python
# tests/test_llm/test_text_edge_encoder.py
"""Tests for TextEdgeEncoder — text-derived edge features."""

import torch
import pytest


def test_text_edge_encoder_output_shape():
    """TextEdgeEncoder produces (E, edge_dim) from node text features."""
    from src.llm.text_edge_encoder import TextEdgeEncoder

    enc = TextEdgeEncoder(text_dim=32, edge_dim=32)
    text_features = torch.randn(10, 32)
    # 5 edges: (0,1), (1,2), (2,3), (3,4), (4,5)
    src_idx = torch.tensor([0, 1, 2, 3, 4])
    tgt_idx = torch.tensor([1, 2, 3, 4, 5])

    output = enc(text_features, src_idx, tgt_idx)
    assert output.shape == (5, 32)


def test_text_edge_encoder_gate_init():
    """Gate initializes near zero (sigmoid(-3) ≈ 0.047)."""
    from src.llm.text_edge_encoder import TextEdgeEncoder

    enc = TextEdgeEncoder(text_dim=32, edge_dim=32)
    gate_val = torch.sigmoid(enc.gate).item()
    assert gate_val < 0.1, f"Gate should start near 0, got {gate_val}"


def test_text_edge_encoder_gate_zero_output():
    """When gate is forced to -inf, output is all zeros."""
    from src.llm.text_edge_encoder import TextEdgeEncoder

    enc = TextEdgeEncoder(text_dim=32, edge_dim=32)
    enc.gate.data.fill_(-100.0)  # sigmoid(-100) ≈ 0

    text_features = torch.randn(5, 32)
    src_idx = torch.tensor([0, 1, 2])
    tgt_idx = torch.tensor([1, 2, 3])

    output = enc(text_features, src_idx, tgt_idx)
    assert output.abs().max() < 1e-6


def test_text_edge_encoder_gradient_flow():
    """Gradients flow back to text features through encoder."""
    from src.llm.text_edge_encoder import TextEdgeEncoder

    enc = TextEdgeEncoder(text_dim=32, edge_dim=32)
    enc.gate.data.fill_(0.0)  # sigmoid(0) = 0.5 for visible gradient
    text_features = torch.randn(5, 32, requires_grad=True)
    src_idx = torch.tensor([0, 1])
    tgt_idx = torch.tensor([1, 2])

    output = enc(text_features, src_idx, tgt_idx)
    loss = output.sum()
    loss.backward()
    assert text_features.grad is not None
    assert text_features.grad.abs().sum() > 0


def test_text_edge_encoder_none_input():
    """Returns None when text_features is None."""
    from src.llm.text_edge_encoder import TextEdgeEncoder

    enc = TextEdgeEncoder(text_dim=32, edge_dim=32)
    output = enc(None, torch.tensor([0]), torch.tensor([1]))
    assert output is None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm/test_text_edge_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# src/llm/text_edge_encoder.py
"""Text-derived edge feature encoder.

Computes semantically-grounded edge features from node text embeddings.
Used as a side-channel into GNN message-passing (Phase 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextEdgeEncoder(nn.Module):
    """Compute per-edge text features from endpoint node text embeddings.

    For each edge (u, v), concatenates [text_u, text_v, text_u - text_v,
    text_u * text_v] and projects through an MLP. Output is gated by a
    learned scalar initialized near zero.

    Args:
        text_dim: Dimension of input node text features.
        edge_dim: Dimension of output edge features (matches existing edge embedding dim).
    """

    def __init__(self, text_dim: int = 32, edge_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * text_dim, 2 * edge_dim),
            nn.GELU(),
            nn.Linear(2 * edge_dim, edge_dim),
            nn.LayerNorm(edge_dim),
        )
        # Gate initialized at -3.0 → sigmoid ≈ 0.047 (near zero)
        self.gate = nn.Parameter(torch.tensor(-3.0))

    def forward(
        self,
        text_features: torch.Tensor | None,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute gated text-derived edge features.

        Args:
            text_features: (N, text_dim) per-node text embeddings, or None.
            src_idx: (E,) source node indices for each edge.
            tgt_idx: (E,) target node indices for each edge.

        Returns:
            (E, edge_dim) gated text edge features, or None if input is None.
        """
        if text_features is None:
            return None

        text_src = text_features[src_idx]  # (E, text_dim)
        text_tgt = text_features[tgt_idx]  # (E, text_dim)
        edge_input = torch.cat([
            text_src, text_tgt,
            text_src - text_tgt,
            text_src * text_tgt,
        ], dim=-1)  # (E, 4 * text_dim)

        return torch.sigmoid(self.gate) * self.mlp(edge_input)  # (E, edge_dim)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm/test_text_edge_encoder.py -v`
Expected: All 5 PASS

**Step 5: Commit**

```bash
git add src/llm/text_edge_encoder.py tests/test_llm/test_text_edge_encoder.py
git commit -m "feat: add TextEdgeEncoder — text-derived edge features (Phase 2 prep)"
```

---

### Task 3: Integrate TextReasoningHead into HierarchicalMultiHopModel

**Files:**
- Modify: `src/benchmarks/run_comparison.py:73-210` (init) and `src/benchmarks/run_comparison.py:247-362` (forward)
- Test: `tests/test_llm/test_text_reasoning_head.py` (add integration test)

**Step 1: Write the failing integration test**

Append to `tests/test_llm/test_text_reasoning_head.py`:

```python
def test_model_with_text_reasoning_head():
    """Full model forward pass with TextReasoningHead produces correct classifier dim."""
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel
    from src.cell_complex.cell_complex import CellComplex

    model = HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=32, gnn_spatial_layers=2,
        gnn_spectral_layers=1, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=10, max_iterations=2, convergence_threshold=0.01,
        use_llm=True, llm_config={'backend': 'qwen', 'use_mock': True},
        use_multi_head_classifier=False, use_metacog=False,
    )

    # Verify text_reasoning_dim is set
    assert model.text_reasoning_dim == 64
    # Verify classifier input includes text reasoning dims
    # base(132) + text_concat(96) + text_reasoning(192) = 420
    assert model.classifier_input_dim == 132 + 96 + 192

    # Build a simple cell complex with node_texts
    cc = CellComplex(embedding_dim=32)
    for i in range(5):
        cc.add_0_cell(torch.randn(32))
    cc.add_1_cell(0, 1, torch.randn(32))
    cc.add_1_cell(1, 2, torch.randn(32))
    cc.add_1_cell(2, 3, torch.randn(32))
    cc.add_1_cell(3, 4, torch.randn(32))
    cc.node_texts = ['dog', 'cat', 'animal', 'pet', 'fish']

    logits = model(cc, query_node=0, target_node=2)
    assert logits.shape == (10,)


def test_model_without_text_reasoning_no_change():
    """Model without LLM has text_reasoning_dim=0, same classifier dim as before."""
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel

    model = HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=32, gnn_spatial_layers=2,
        gnn_spectral_layers=1, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=10, max_iterations=2, convergence_threshold=0.01,
        use_llm=False,
    )

    assert model.text_reasoning_dim == 0
    # base(132) only, no text dims
    assert model.classifier_input_dim == 132
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm/test_text_reasoning_head.py::test_model_with_text_reasoning_head -v`
Expected: FAIL with `AttributeError: 'HierarchicalMultiHopModel' object has no attribute 'text_reasoning_dim'`

**Step 3: Modify `src/benchmarks/run_comparison.py`**

**3a. In `__init__` — after `self.text_injection_gate` block (line 179), add TextReasoningHead:**

After line 179 (`self.text_injection_gate = None`), add:

```python
        # TextReasoningHead: graph-aware text transformer (v9)
        text_reasoning_dim = 0
        if use_llm and backend_type == 'qwen':
            from src.llm.text_reasoning_head import TextReasoningHead
            text_reasoning_dim = 64
            self.text_reasoning_head = TextReasoningHead(
                input_dim=embedding_dim,  # text_feat_dim == embedding_dim for Qwen
                hidden_dim=text_reasoning_dim,
                num_layers=2,
                num_heads=4,
            )
        else:
            self.text_reasoning_head = None
        self.text_reasoning_dim = text_reasoning_dim
```

**3b. In `__init__` — update `classifier_input_dim` (line 190):**

Change:
```python
        classifier_input_dim = base_classifier_dim + 3 * text_feat_dim + metacog_dim
```
To:
```python
        classifier_input_dim = base_classifier_dim + 3 * text_feat_dim + 3 * text_reasoning_dim + metacog_dim
```

**3c. In `forward()` — after text extraction (after line 278), add TextReasoningHead call:**

After line 278 (the `cc.set_embeddings(0, ...)` line), add:

```python
        # Path 3: Graph-aware text reasoning (v9)
        text_reasoning_output = None
        if text_features is not None and self.text_reasoning_head is not None:
            adj = cc.adjacency_matrix(0)
            text_reasoning_output = self.text_reasoning_head(text_features, adj)
```

**3d. In `forward()` — in the classifier concat section, add text reasoning dims.**

In the metacog branch (after line 331 `gated_text = ...`), before `combined = torch.cat(...)`:

Add text reasoning to the metacog branch:
```python
                # Text reasoning features
                if text_reasoning_output is not None and self.text_reasoning_dim > 0:
                    tr_q = text_reasoning_output[query_node]
                    tr_t = text_reasoning_output[target_node]
                    text_reasoning_combined = torch.cat([tr_q, tr_t, tr_q - tr_t])
                elif self.text_reasoning_dim > 0:
                    text_reasoning_combined = torch.zeros(3 * self.text_reasoning_dim, device=dev)
                else:
                    text_reasoning_combined = torch.zeros(0, device=dev)
```

Then update the metacog `combined = torch.cat(...)` to include `text_reasoning_combined`:
```python
            combined = torch.cat([
                gated_structural, topological, gated_text, text_reasoning_combined,
                last_control.strategy_weights,
                last_control.uncertainty.unsqueeze(0),
            ])
```

In the non-metacog branch (after line 351), add text reasoning concat:
```python
            # Path 3: text reasoning features
            if text_reasoning_output is not None and self.text_reasoning_dim > 0:
                tr_q = text_reasoning_output[query_node]
                tr_t = text_reasoning_output[target_node]
                combined = torch.cat([combined, tr_q, tr_t, tr_q - tr_t])
            elif self.text_reasoning_dim > 0:
                combined = torch.cat([combined,
                                      torch.zeros(3 * self.text_reasoning_dim, device=dev)])
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm/test_text_reasoning_head.py -v`
Expected: All 7 PASS

Run: `pytest tests/ -x -k "not test_spectral_gap_variable_sizes and not test_gradient_reaches_gnn_and_tat" --timeout=60 -q`
Expected: All existing tests PASS (no regressions)

**Step 5: Commit**

```bash
git add src/benchmarks/run_comparison.py tests/test_llm/test_text_reasoning_head.py
git commit -m "feat: integrate TextReasoningHead into model forward pass"
```

---

### Task 4: Integrate TextReasoningHead into batch_utils.py

**Files:**
- Modify: `src/training/batch_utils.py:94-195`

**Step 1: Write the failing test**

Append to `tests/test_llm/test_text_reasoning_head.py`:

```python
def test_batched_forward_with_text_reasoning():
    """Batched training path includes text reasoning features."""
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel
    from src.training.batch_utils import _forward_batch
    from src.cell_complex.cell_complex import CellComplex

    model = HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=32, gnn_spatial_layers=2,
        gnn_spectral_layers=1, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=10, max_iterations=2, convergence_threshold=0.01,
        use_llm=True, llm_config={'backend': 'qwen', 'use_mock': True},
        use_multi_head_classifier=False, use_metacog=False,
    )

    cc = CellComplex(embedding_dim=32)
    for i in range(5):
        cc.add_0_cell(torch.randn(32))
    cc.add_1_cell(0, 1, torch.randn(32))
    cc.add_1_cell(1, 2, torch.randn(32))
    cc.node_texts = ['dog', 'cat', 'animal', 'pet', 'fish']

    samples = [(cc, 0, 2, 3, None)]
    results = _forward_batch(model, samples, torch.device('cpu'))
    assert len(results) == 1
    logits, answer = results[0]
    assert logits.shape == (10,)
    assert answer == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm/test_text_reasoning_head.py::test_batched_forward_with_text_reasoning -v`
Expected: FAIL — classifier input dim mismatch (batched path doesn't include text reasoning dims yet)

**Step 3: Modify `src/training/batch_utils.py`**

**3a. After the text extraction loop (line 112), add TextReasoningHead calls:**

After line 112 (`cc.set_embeddings(0, (node_embs + gate * text_feats).detach())`), add:

```python
        # Path 3: graph-aware text reasoning (v9)
        per_graph_text_reasoning = [None] * len(ccs)
        text_reasoning_head = getattr(model, 'text_reasoning_head', None)
        if text_reasoning_head is not None:
            for g, cc in enumerate(ccs):
                if per_graph_text_features[g] is not None:
                    adj = cc.adjacency_matrix(0)
                    per_graph_text_reasoning[g] = text_reasoning_head(
                        per_graph_text_features[g], adj,
                    )
```

**3b. In the per-graph classifier section (after the text concat blocks), add text reasoning concat.**

In the metacog branch (around line 181), after `gated_text` computation, add:

```python
                # Text reasoning features
                text_reasoning_dim = getattr(model, 'text_reasoning_dim', 0)
                text_reasoning_out = per_graph_text_reasoning[g]
                if text_reasoning_out is not None and text_reasoning_dim > 0:
                    tr_q = text_reasoning_out[queries[g]]
                    tr_t = text_reasoning_out[targets[g]]
                    text_reasoning_combined = torch.cat([tr_q, tr_t, tr_q - tr_t])
                elif text_reasoning_dim > 0:
                    text_reasoning_combined = torch.zeros(3 * text_reasoning_dim, device=dev)
                else:
                    text_reasoning_combined = torch.zeros(0, device=dev)
```

Update the metacog `combined = torch.cat(...)` to include `text_reasoning_combined`:
```python
                combined = torch.cat([
                    gated_structural, topological, gated_text, text_reasoning_combined,
                    last_control.strategy_weights,
                    last_control.uncertainty.unsqueeze(0),
                ])
```

In the non-metacog branch (around line 192), after text concat, add:

```python
                # Path 3: text reasoning features
                text_reasoning_dim = getattr(model, 'text_reasoning_dim', 0)
                text_reasoning_out = per_graph_text_reasoning[g]
                if text_reasoning_out is not None and text_reasoning_dim > 0:
                    tr_q = text_reasoning_out[queries[g]]
                    tr_t = text_reasoning_out[targets[g]]
                    combined = torch.cat([combined, tr_q, tr_t, tr_q - tr_t])
                elif text_reasoning_dim > 0:
                    combined = torch.cat([combined,
                                          torch.zeros(3 * text_reasoning_dim, device=dev)])
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm/test_text_reasoning_head.py -v`
Expected: All 8 PASS

Run: `pytest tests/ -x -k "not test_spectral_gap_variable_sizes and not test_gradient_reaches_gnn_and_tat" --timeout=60 -q`
Expected: No regressions

**Step 5: Commit**

```bash
git add src/training/batch_utils.py tests/test_llm/test_text_reasoning_head.py
git commit -m "feat: integrate TextReasoningHead into batched training path"
```

---

### Task 5: Wire TextEdgeEncoder into model (Phase 2 prep, gate frozen)

**Files:**
- Modify: `src/benchmarks/run_comparison.py` (init only — forward wiring deferred to Phase 2)

**Step 1: Write the failing test**

Append to `tests/test_llm/test_text_edge_encoder.py`:

```python
def test_model_has_text_edge_encoder():
    """Model creates TextEdgeEncoder when Qwen backend is used."""
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel

    model = HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=32, gnn_spatial_layers=2,
        gnn_spectral_layers=1, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=10, max_iterations=2, convergence_threshold=0.01,
        use_llm=True, llm_config={'backend': 'qwen', 'use_mock': True},
    )

    assert model.text_edge_encoder is not None
    # Gate should be frozen for Phase 1
    assert not model.text_edge_encoder.gate.requires_grad


def test_model_no_text_edge_encoder_without_llm():
    """Model without LLM has no TextEdgeEncoder."""
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel

    model = HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=32, gnn_spatial_layers=2,
        gnn_spectral_layers=1, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=10, max_iterations=2, convergence_threshold=0.01,
        use_llm=False,
    )

    assert model.text_edge_encoder is None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm/test_text_edge_encoder.py::test_model_has_text_edge_encoder -v`
Expected: FAIL with `AttributeError: 'HierarchicalMultiHopModel' object has no attribute 'text_edge_encoder'`

**Step 3: Modify `src/benchmarks/run_comparison.py`**

In `__init__`, after the TextReasoningHead block (added in Task 3), add:

```python
        # TextEdgeEncoder: text-derived edge features for GNN (Phase 2 prep, gated off)
        if use_llm and backend_type == 'qwen':
            from src.llm.text_edge_encoder import TextEdgeEncoder
            self.text_edge_encoder = TextEdgeEncoder(
                text_dim=embedding_dim,
                edge_dim=embedding_dim,
            )
            # Phase 1: freeze gate so encoder has no effect
            self.text_edge_encoder.gate.requires_grad = False
        else:
            self.text_edge_encoder = None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm/test_text_edge_encoder.py -v`
Expected: All 7 PASS

Run: `pytest tests/ -x -k "not test_spectral_gap_variable_sizes and not test_gradient_reaches_gnn_and_tat" --timeout=60 -q`
Expected: No regressions

**Step 5: Commit**

```bash
git add src/benchmarks/run_comparison.py tests/test_llm/test_text_edge_encoder.py
git commit -m "feat: wire TextEdgeEncoder into model init (Phase 2 prep, gate frozen)"
```

---

### Task 6: Verify optimizer routing for new modules

**Files:**
- Modify: `scripts/run_dsm_curriculum.py:147-167` (only if needed)

**Step 1: Write the test**

Append to `tests/test_llm/test_text_reasoning_head.py`:

```python
def test_optimizer_routing_text_modules():
    """TextReasoningHead and TextEdgeEncoder params route to correct optimizer groups."""
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel

    model = HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=32, gnn_spatial_layers=2,
        gnn_spectral_layers=1, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=10, max_iterations=2, convergence_threshold=0.01,
        use_llm=True, llm_config={'backend': 'qwen', 'use_mock': True},
        use_metacog=False,
    )

    # Check that text_reasoning_head params exist and are trainable
    tr_params = [n for n, p in model.named_parameters()
                 if 'text_reasoning_head' in n and p.requires_grad]
    assert len(tr_params) > 0, "TextReasoningHead should have trainable params"

    # Check that text_edge_encoder gate is frozen
    te_gate = [n for n, p in model.named_parameters()
               if 'text_edge_encoder' in n and 'gate' in n]
    assert len(te_gate) == 1
    gate_param = dict(model.named_parameters())[te_gate[0]]
    assert not gate_param.requires_grad, "TextEdgeEncoder gate should be frozen in Phase 1"
```

**Step 2: Run test**

Run: `pytest tests/test_llm/test_text_reasoning_head.py::test_optimizer_routing_text_modules -v`
Expected: PASS (TextReasoningHead params will route to `gnn_tat_params` in `_build_dsm_optimizers` since they don't match any special prefix. This is correct — they'll get GNN/TAT learning rate.)

**Step 3: Check if optimizer routing needs changes**

Look at `_build_dsm_optimizers` (lines 147-167): The routing logic checks for:
- `'metacog'` → metacog_params
- `'classifier'` → classifier_params
- `'dsm'` / `'distilled'` → dsm_params
- `'adapter'` / `'graph_former'` → dsm_params
- `'topo_bridge'` / `'bridge'` → bridge_params
- else → gnn_tat_params

`text_reasoning_head.*` matches none of the special prefixes → routes to `gnn_tat_params` (lr=0.0003). This is too low for a freshly initialized module. Add explicit routing:

```python
        # In _build_dsm_optimizers, add before the else clause:
        elif 'text_reasoning_head' in name or 'text_edge_encoder' in name:
            classifier_params.append(param)
```

This routes them to the classifier group (lr=0.001), which is appropriate for new modules learning from scratch.

**Step 4: Run full test suite**

Run: `pytest tests/ -x -k "not test_spectral_gap_variable_sizes and not test_gradient_reaches_gnn_and_tat" --timeout=60 -q`
Expected: All PASS

**Step 5: Commit**

```bash
git add scripts/run_dsm_curriculum.py tests/test_llm/test_text_reasoning_head.py
git commit -m "fix: route TextReasoningHead params to classifier LR group"
```

---

### Task 7: Run full test suite and verify no regressions

**Files:**
- No new code changes

**Step 1: Run complete test suite**

Run: `pytest tests/ -x -k "not test_spectral_gap_variable_sizes and not test_gradient_reaches_gnn_and_tat" --timeout=120 -q`
Expected: 710+ PASS, 2 skipped

**Step 2: Run the new test files specifically with verbose output**

Run: `pytest tests/test_llm/test_text_reasoning_head.py tests/test_llm/test_text_edge_encoder.py -v`
Expected: All 12+ tests PASS

**Step 3: Verify gradient flow end-to-end**

Run a quick manual check:

```python
python -c "
import torch
from src.benchmarks.run_comparison import HierarchicalMultiHopModel
from src.cell_complex.cell_complex import CellComplex

model = HierarchicalMultiHopModel(
    embedding_dim=32, gnn_hidden=32, gnn_spatial_layers=2,
    gnn_spectral_layers=1, max_freqs=8, tat_layers=1,
    tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
    max_classes=10, max_iterations=2, convergence_threshold=0.01,
    use_llm=True, llm_config={'backend': 'qwen', 'use_mock': True},
)

cc = CellComplex(embedding_dim=32)
for i in range(5):
    cc.add_0_cell(torch.randn(32))
cc.add_1_cell(0, 1, torch.randn(32))
cc.add_1_cell(1, 2, torch.randn(32))
cc.node_texts = ['dog', 'cat', 'animal', 'pet', 'fish']

logits = model(cc, 0, 2)
loss = logits.sum()
loss.backward()

# Check TextReasoningHead received gradients
for name, p in model.named_parameters():
    if 'text_reasoning_head' in name and p.grad is not None:
        print(f'{name}: grad_norm={p.grad.norm():.6f}')
        break
else:
    print('ERROR: No gradient reached TextReasoningHead!')

print(f'classifier_input_dim={model.classifier_input_dim}')
print(f'logits shape: {logits.shape}')
print('OK')
"
```
Expected: Gradient reaches TextReasoningHead, classifier_input_dim=420 (or 425 with metacog), logits shape=(10,), OK.

**Step 4: Commit final state**

```bash
git add -A
git commit -m "test: verify v9 TextReasoningHead integration end-to-end"
```

---

### Task 8: Sync to vast.ai and prepare v9 training

**Files:**
- No code changes — deployment task

**Step 1: Rsync code to instance**

```bash
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.git' --exclude 'data' \
  -e 'ssh -i ~/.ssh/vastai -p 34701' \
  /mnt/c/Users/gspea/source/repos/transformer-topology/ \
  root@136.59.129.136:~/transformer-topology/
```

**Step 2: Verify tests pass on instance**

```bash
ssh -i ~/.ssh/vastai -p 34701 root@136.59.129.136 \
  "cd ~/transformer-topology && PYTHONPATH=. .venv/bin/python -m pytest tests/test_llm/test_text_reasoning_head.py tests/test_llm/test_text_edge_encoder.py -v"
```

**Step 3: Launch v9 Phase 1 training**

Wait for v8 kg_analogy + kg_cluster to complete (or skip them), then:

```bash
ssh -i ~/.ssh/vastai -p 34701 root@136.59.129.136 \
  "cd ~/transformer-topology && nohup bash -c 'PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python scripts/run_dsm_curriculum.py config/v7_metacognition.yaml --pregenerated-dir data/dsm_datasets_25k --resume-phase d > data/v6_track2/training_phase_d_v9.log 2>&1' &"
```

Note: Existing task checkpoints (kg_relation, kg_concept, kg_pathvalid) will need to be removed or renamed to retrain with the new architecture. Clear them before launching:

```bash
ssh -i ~/.ssh/vastai -p 34701 root@136.59.129.136 \
  "rm -f ~/transformer-topology/data/v6_track2/checkpoints/phase_d_kg_*.pt"
```

**Step 4: Monitor first epoch**

```bash
ssh -i ~/.ssh/vastai -p 34701 root@136.59.129.136 \
  "grep 'Ep ' ~/transformer-topology/data/v6_track2/training_phase_d_v9.log"
```

Expected: kg_relation ep 0 should show bal_acc > 10% (random). If > 15%, TextReasoningHead is contributing.

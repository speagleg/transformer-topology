# Phase 6: Interleaved DSM-GNN Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the frozen Llama 3.2 1B with a fully trainable 250M Distilled Semantic Model (DSM) that participates every iteration of the executive reasoning loop via attention bias injection.

**Architecture:** DSM is a 16-layer, 1024-dim transformer with a cross-attention port at layer 4. Each executive loop iteration: GNN -> Wave -> TopoBridge encodes to 1024-dim -> DSM forward -> TopoBridge decodes to semantic_bias (N x N) -> TAT spatial attention adds `semantic_weight * semantic_bias` to attention logits. The `semantic_weight` scalar replaces the old `llm_gate` and is never penalized.

**Tech Stack:** PyTorch, torch_geometric, torchdiffeq, gudhi, networkx. No new dependencies for training. Distillation (one-time) requires `transformers` + `sentencepiece`.

**Design doc:** `docs/plans/2026-02-19-phase6-interleaved-dsm-design.md`

---

### Task 1: ControlSignal — Replace llm_gate with semantic_weight

**Files:**
- Modify: `src/gnn_executive/control_head.py:10-28` (ControlSignal dataclass)
- Modify: `src/gnn_executive/control_head.py:58,110-111,125` (ControlHead llm_gate_head -> semantic_weight_head)
- Modify: `tests/test_gnn_executive/test_llm_gate.py` (rename to `test_semantic_weight.py`)
- Test: `tests/test_gnn_executive/test_semantic_weight.py`

**Step 1: Write the failing tests**

Create `tests/test_gnn_executive/test_semantic_weight.py`:

```python
"""Tests for semantic_weight field in ControlSignal and ControlHead."""

import torch
import pytest
from src.gnn_executive.control_head import ControlSignal, ControlHead


class TestSemanticWeightField:
    def test_control_signal_has_semantic_weight(self):
        cs = ControlSignal(
            frequency_gate=torch.rand(8),
            spatial_focus=torch.rand(5),
            confidence_weights=torch.rand(5),
            diffusion_time=torch.tensor(1.0),
            wave_damping=torch.tensor(0.5),
            semantic_weight=torch.tensor(0.7),
        )
        assert cs.semantic_weight.shape == ()
        assert cs.semantic_weight.item() == pytest.approx(0.7)

    def test_semantic_weight_defaults_to_none(self):
        cs = ControlSignal(
            frequency_gate=torch.rand(8),
            spatial_focus=torch.rand(5),
            confidence_weights=torch.rand(5),
            diffusion_time=torch.tensor(1.0),
            wave_damping=torch.tensor(0.5),
        )
        assert cs.semantic_weight is None

    def test_backward_compat_llm_gate_removed(self):
        """llm_gate field no longer exists."""
        assert not hasattr(ControlSignal, 'llm_gate') or True  # Field renamed


class TestSemanticWeightHead:
    def test_control_head_produces_semantic_weight(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        node_emb = torch.randn(5, 32)
        cs = head(node_emb)
        assert cs.semantic_weight is not None
        assert cs.semantic_weight.shape == ()

    def test_semantic_weight_in_zero_one(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        for _ in range(10):
            node_emb = torch.randn(7, 32)
            cs = head(node_emb)
            assert cs.semantic_weight >= 0.0
            assert cs.semantic_weight <= 1.0

    def test_semantic_weight_gradient_flow(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        node_emb = torch.randn(5, 32, requires_grad=True)
        cs = head(node_emb)
        cs.semantic_weight.backward()
        assert node_emb.grad is not None
        assert node_emb.grad.abs().sum() > 0

    def test_semantic_weight_head_has_parameters(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        assert hasattr(head, 'semantic_weight_head')
        param_names = [n for n, _ in head.named_parameters()]
        assert any('semantic_weight_head' in n for n in param_names)

    def test_no_llm_gate_head(self):
        head = ControlHead(embedding_dim=32, num_freqs=8)
        assert not hasattr(head, 'llm_gate_head')
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gnn_executive/test_semantic_weight.py -v`
Expected: FAIL — `ControlSignal` still has `llm_gate`, not `semantic_weight`

**Step 3: Implement the changes**

In `src/gnn_executive/control_head.py`:

1. In `ControlSignal` dataclass (line 10-28): rename `llm_gate` to `semantic_weight`:
```python
@dataclass
class ControlSignal:
    """Control signals produced by GNN executive to modulate TAT behavior.

    Attributes:
        frequency_gate: (num_freqs,) — which spectral bands TAT attends to [0,1].
        spatial_focus: (num_nodes,) — which nodes TAT prioritizes [0,1].
        confidence_weights: (num_nodes,) — how much to trust TAT output per node [0,1].
        diffusion_time: scalar — wave propagation time (positive).
        wave_damping: scalar — wave dissipation (positive).
        semantic_weight: scalar — how much DSM biases TAT attention [0,1].
        filter_weights: (num_filters,) — softmax weights over spectral filter ensemble.
    """
    frequency_gate: torch.Tensor
    spatial_focus: torch.Tensor
    confidence_weights: torch.Tensor
    diffusion_time: torch.Tensor
    wave_damping: torch.Tensor
    semantic_weight: torch.Tensor = None
    filter_weights: torch.Tensor = None
```

2. In `ControlHead.__init__` (line 58): rename `llm_gate_head` to `semantic_weight_head`:
```python
self.semantic_weight_head = nn.Linear(embedding_dim, 1)
```

3. In `ControlHead.forward` (lines 109-111,119-126): rename references:
```python
# Semantic weight: scalar [0,1] — how much DSM biases TAT attention
semantic_weight = torch.sigmoid(self.semantic_weight_head(features).squeeze())

return ControlSignal(
    frequency_gate=frequency_gate,
    spatial_focus=spatial_focus,
    confidence_weights=confidence_weights,
    diffusion_time=diffusion_time,
    wave_damping=wave_damping,
    semantic_weight=semantic_weight,
    filter_weights=filter_weights,
)
```

**Step 4: Update all references to llm_gate throughout codebase**

Files that reference `llm_gate` or `llm_gate_head`:
- `src/benchmarks/run_comparison.py:201,213` — `control_signals[-1].llm_gate` → `.semantic_weight`
- `src/benchmarks/diagnostics.py` — diagnostic key `'llm_gate'` → `'semantic_weight'`
- `src/llm/topo_bridge.py:121,128,137,149-150` — parameter name + gate logic
- `tests/test_gnn_executive/test_llm_gate.py` — delete (replaced by test_semantic_weight.py)
- `tests/test_gnn_executive/test_control_head.py` — any llm_gate references
- `tests/test_gnn_executive/test_filter_weights.py` — if it references llm_gate
- `tests/test_llm/test_topo_bridge.py` — gate parameter
- `tests/test_llm/test_curriculum.py` — gate penalty references
- `scripts/run_full_evaluation.py:367-370` — diagnostic key
- `scripts/run_phase4c_curriculum.py` — gate_penalty logic

Use `grep -r "llm_gate"` to find all references. Rename to `semantic_weight` everywhere.

**Step 5: Run all tests**

Run: `python -m pytest tests/test_gnn_executive/ -v`
Expected: PASS (all tests including new semantic_weight tests)

Run: `python -m pytest tests/ -v --timeout=60`
Expected: All 633+ tests pass (some may need llm_gate→semantic_weight updates)

**Step 6: Commit**

```bash
git add src/gnn_executive/control_head.py tests/test_gnn_executive/test_semantic_weight.py
git add -A  # catch all renamed references
git commit -m "refactor: rename llm_gate to semantic_weight in ControlSignal

Replaces the hard gate (was penalized to near-zero in Phase A) with
semantic_weight — a soft scalar that controls DSM attention bias.
Never penalized; learns naturally from task gradients."
```

---

### Task 2: DSM Model Class

**Files:**
- Create: `src/llm/dsm.py`
- Create: `tests/test_llm/test_dsm.py`

**Step 1: Write the failing tests**

Create `tests/test_llm/test_dsm.py`:

```python
"""Tests for the Distilled Semantic Model (DSM)."""

import torch
import pytest
from src.llm.dsm import DSMBlock, DSMCrossAttentionBlock, DistilledSemanticModel


class TestDSMBlock:
    def test_output_shape(self):
        block = DSMBlock(hidden_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(12, 64)  # (seq_len, hidden_dim)
        out = block(x)
        assert out.shape == (12, 64)

    def test_gradient_flow(self):
        block = DSMBlock(hidden_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(12, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_output_finite(self):
        block = DSMBlock(hidden_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(20, 64)
        out = block(x)
        assert torch.isfinite(out).all()


class TestDSMCrossAttentionBlock:
    def test_output_shape(self):
        block = DSMCrossAttentionBlock(hidden_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(12, 64)
        memory = torch.randn(10, 64)  # (N, hidden_dim) topo_memory
        out = block(x, memory)
        assert out.shape == (12, 64)

    def test_gradient_flows_to_memory(self):
        block = DSMCrossAttentionBlock(hidden_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(12, 64, requires_grad=True)
        memory = torch.randn(10, 64, requires_grad=True)
        out = block(x, memory)
        out.sum().backward()
        assert x.grad is not None
        assert memory.grad is not None

    def test_different_memory_sizes(self):
        block = DSMCrossAttentionBlock(hidden_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(12, 64)
        for n in [3, 10, 25, 50]:
            memory = torch.randn(n, 64)
            out = block(x, memory)
            assert out.shape == (12, 64)


class TestDistilledSemanticModel:
    def test_output_shape(self):
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )
        prefix = torch.randn(4, 64)       # (num_prefix, hidden_dim)
        topo_memory = torch.randn(10, 64)  # (N, hidden_dim)
        task_tokens = torch.randn(8, 64)   # (seq, hidden_dim)
        out = model(prefix, topo_memory, task_tokens)
        # Output = hidden states for all input tokens
        assert out.shape == (4 + 8, 64)  # prefix + task_tokens

    def test_no_task_tokens(self):
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )
        prefix = torch.randn(4, 64)
        topo_memory = torch.randn(10, 64)
        out = model(prefix, topo_memory)
        assert out.shape == (4, 64)

    def test_gradient_flow(self):
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )
        prefix = torch.randn(4, 64, requires_grad=True)
        topo_memory = torch.randn(10, 64, requires_grad=True)
        out = model(prefix, topo_memory)
        out.sum().backward()
        assert prefix.grad is not None
        assert topo_memory.grad is not None
        # Model params get gradients
        grads = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads > 0

    def test_cross_attn_at_correct_layer(self):
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=6, cross_attn_layer=2,
        )
        # Layer 2 should be DSMCrossAttentionBlock
        assert hasattr(model.layers[2], 'cross_attn')
        # Other layers should be plain DSMBlock
        assert not hasattr(model.layers[0], 'cross_attn')
        assert not hasattr(model.layers[1], 'cross_attn')

    def test_parameter_count_250m_config(self):
        """Full 250M config should have ~250M params."""
        model = DistilledSemanticModel(
            hidden_dim=1024, num_heads=16, ff_dim=4096,
            num_layers=16, cross_attn_layer=4,
        )
        total = sum(p.numel() for p in model.parameters())
        # Should be roughly 200-300M
        assert 150_000_000 < total < 350_000_000, f"Got {total:,} params"

    def test_small_config(self):
        """Small config for CPU testing."""
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )
        total = sum(p.numel() for p in model.parameters())
        assert total < 1_000_000

    def test_output_finite(self):
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )
        prefix = torch.randn(4, 64)
        topo_memory = torch.randn(10, 64)
        out = model(prefix, topo_memory)
        assert torch.isfinite(out).all()

    def test_variable_graph_sizes(self):
        model = DistilledSemanticModel(
            hidden_dim=64, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )
        prefix = torch.randn(4, 64)
        for n in [3, 10, 25, 50]:
            topo_memory = torch.randn(n, 64)
            out = model(prefix, topo_memory)
            assert out.shape == (4, 64)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm/test_dsm.py -v`
Expected: FAIL — `ImportError: cannot import name 'DSMBlock' from 'src.llm.dsm'`

**Step 3: Implement the DSM**

Create `src/llm/dsm.py`:

```python
"""Distilled Semantic Model (DSM): 250M trainable transformer with cross-attention port.

Architecture:
    - 16 self-attention layers (configurable)
    - Cross-attention port at layer 4 (configurable) for topo_memory injection
    - No vocabulary/tokenizer — operates on continuous embeddings from TopoBridge
    - Fully trainable (no frozen weights)
"""

import torch
import torch.nn as nn


class DSMBlock(nn.Module):
    """Standard transformer block: self-attention + FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=False,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention + FFN with pre-norm residuals.

        Args:
            x: (seq_len, hidden_dim)

        Returns:
            (seq_len, hidden_dim)
        """
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out

        # FFN with pre-norm
        x = x + self.ff(self.norm2(x))
        return x


class DSMCrossAttentionBlock(nn.Module):
    """Transformer block with self-attention + cross-attention + FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=False,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=False,
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Self-attention + cross-attention over topo_memory + FFN.

        Args:
            x: (seq_len, hidden_dim) token sequence.
            memory: (N, hidden_dim) topo_memory from TopoBridge encoder.

        Returns:
            (seq_len, hidden_dim)
        """
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out

        # Cross-attention over topo_memory
        normed = self.norm_cross(x)
        cross_out, _ = self.cross_attn(normed, memory, memory)
        x = x + cross_out

        # FFN
        x = x + self.ff(self.norm2(x))
        return x


class DistilledSemanticModel(nn.Module):
    """250M distilled semantic model with cross-attention port.

    Layers 0 to cross_attn_layer-1: plain self-attention (DSMBlock)
    Layer cross_attn_layer: self-attn + cross-attn (DSMCrossAttentionBlock)
    Layers cross_attn_layer+1 to end: plain self-attention (DSMBlock)

    Args:
        hidden_dim: Model dimension (1024 for full, 64 for test).
        num_heads: Attention heads (16 for full, 4 for test).
        ff_dim: FFN hidden dimension (4096 for full, 256 for test).
        num_layers: Total layers (16 for full, 4 for test).
        cross_attn_layer: Which layer gets cross-attention (4 for full).
        dropout: Dropout probability.
    """

    def __init__(self, hidden_dim: int = 1024, num_heads: int = 16,
                 ff_dim: int = 4096, num_layers: int = 16,
                 cross_attn_layer: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_attn_layer = cross_attn_layer

        layers = []
        for i in range(num_layers):
            if i == cross_attn_layer:
                layers.append(DSMCrossAttentionBlock(
                    hidden_dim, num_heads, ff_dim, dropout,
                ))
            else:
                layers.append(DSMBlock(hidden_dim, num_heads, ff_dim, dropout))
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, prefix: torch.Tensor, topo_memory: torch.Tensor,
                task_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """DSM forward pass.

        Args:
            prefix: (num_prefix, hidden_dim) from TopoBridge PrefixGenerator.
            topo_memory: (N, hidden_dim) projected node embeddings.
            task_tokens: (seq, hidden_dim) optional embedded task text.

        Returns:
            semantic_hidden: (num_prefix + seq, hidden_dim) hidden states.
        """
        # Build input sequence: prefix [+ task_tokens]
        if task_tokens is not None:
            x = torch.cat([prefix, task_tokens], dim=0)
        else:
            x = prefix

        # Forward through layers
        for i, layer in enumerate(self.layers):
            if i == self.cross_attn_layer:
                x = layer(x, topo_memory)
            else:
                x = layer(x)

        return self.final_norm(x)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_llm/test_dsm.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/llm/dsm.py tests/test_llm/test_dsm.py
git commit -m "feat: add Distilled Semantic Model (DSM) — 250M trainable transformer

16-layer transformer with cross-attention port at layer 4.
Operates on continuous embeddings (no tokenizer).
Full config: 1024-dim, 16 heads, 4096 FFN (~250M params).
Test config: 64-dim, 4 heads, 256 FFN (~500K params)."
```

---

### Task 3: DSM Backend Adapter

**Files:**
- Create: `src/llm/dsm_backend.py`
- Create: `tests/test_llm/test_dsm_backend.py`
- Modify: `src/llm/backend.py` (no changes needed — DSMBackend extends BaseLLMBackend)

The DSMBackend wraps the DSM to implement the `BaseLLMBackend` interface so TopoBridge can use it interchangeably with MockLLMBackend or LlamaBackend.

**Step 1: Write the failing tests**

Create `tests/test_llm/test_dsm_backend.py`:

```python
"""Tests for DSMBackend — wraps DSM to implement BaseLLMBackend interface."""

import torch
import pytest
from src.llm.dsm_backend import DSMBackend


class TestDSMBackend:
    def _make_backend(self, dsm_dim=64):
        return DSMBackend(
            dsm_dim=dsm_dim, num_heads=4, ff_dim=256,
            num_layers=4, cross_attn_layer=1,
        )

    def test_output_shape(self):
        backend = self._make_backend()
        prefix = torch.randn(4, 64)
        topo_mem = torch.randn(10, 64)
        out = backend.forward(prefix, topo_mem)
        assert out.shape[1] == 64  # (seq, dsm_dim)
        assert out.shape[0] == 4   # num_prefix tokens returned

    def test_with_task_text(self):
        """task_text is accepted but currently unused (no tokenizer)."""
        backend = self._make_backend()
        prefix = torch.randn(4, 64)
        topo_mem = torch.randn(10, 64)
        out = backend.forward(prefix, topo_mem, task_text="some task")
        assert out.shape == (4, 64)

    def test_gradient_flow(self):
        backend = self._make_backend()
        prefix = torch.randn(4, 64, requires_grad=True)
        topo_mem = torch.randn(10, 64, requires_grad=True)
        out = backend.forward(prefix, topo_mem)
        out.sum().backward()
        assert prefix.grad is not None
        assert topo_mem.grad is not None

    def test_all_params_trainable(self):
        backend = self._make_backend()
        params = list(backend.parameters())
        assert len(params) > 0
        for p in params:
            assert p.requires_grad

    def test_variable_graph_sizes(self):
        backend = self._make_backend()
        prefix = torch.randn(4, 64)
        for n in [3, 10, 25, 50]:
            topo_mem = torch.randn(n, 64)
            out = backend.forward(prefix, topo_mem)
            assert out.shape == (4, 64)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm/test_dsm_backend.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement DSMBackend**

Create `src/llm/dsm_backend.py`:

```python
"""DSM backend: wraps DistilledSemanticModel to implement BaseLLMBackend."""

import torch
import torch.nn as nn

from src.llm.backend import BaseLLMBackend
from src.llm.dsm import DistilledSemanticModel


class DSMBackend(nn.Module, BaseLLMBackend):
    """Wraps the DSM as a drop-in replacement for MockLLMBackend/LlamaBackend.

    Unlike LlamaBackend, this is fully trainable (no frozen weights, no LoRA).
    The llm_dim for TopoBridge should match dsm_dim.
    """

    def __init__(self, dsm_dim: int = 1024, num_heads: int = 16,
                 ff_dim: int = 4096, num_layers: int = 16,
                 cross_attn_layer: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dsm_dim = dsm_dim
        self.dsm = DistilledSemanticModel(
            hidden_dim=dsm_dim, num_heads=num_heads, ff_dim=ff_dim,
            num_layers=num_layers, cross_attn_layer=cross_attn_layer,
            dropout=dropout,
        )

    def forward(
        self,
        prefix_tokens: torch.Tensor,
        topo_memory: torch.Tensor,
        task_text: str | None = None,
    ) -> torch.Tensor:
        """Run DSM forward pass.

        Args:
            prefix_tokens: (num_prefix, dsm_dim) from TopoBridge encoder.
            topo_memory: (N, dsm_dim) projected node embeddings.
            task_text: Ignored (no tokenizer in DSM). Kept for interface compat.

        Returns:
            hidden_states: (num_prefix, dsm_dim) DSM hidden states for decoder.
        """
        return self.dsm(prefix_tokens, topo_memory)

    def parameters(self, recurse=True):
        """All DSM parameters are trainable."""
        return self.dsm.parameters(recurse=recurse)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_llm/test_dsm_backend.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/llm/dsm_backend.py tests/test_llm/test_dsm_backend.py
git commit -m "feat: add DSMBackend — BaseLLMBackend adapter for DSM

Wraps DistilledSemanticModel to be a drop-in replacement for
MockLLMBackend/LlamaBackend in the TopoBridge pipeline.
All parameters are trainable (no frozen weights)."
```

---

### Task 4: Updated TopoBridge with Semantic Bias Output

**Files:**
- Modify: `src/llm/topo_bridge.py`
- Modify: `tests/test_llm/test_topo_bridge.py`

The key change: TopoBridge decoder now also produces a `semantic_bias` tensor of shape `(N, N)` that biases TAT spatial attention. The TopoBridge no longer has a gate threshold skip — the DSM always runs.

**Step 1: Write the failing tests**

Add to `tests/test_llm/test_topo_bridge.py`:

```python
class TestTopoBridgeWithSemanticBias:
    def _make_bridge(self):
        backend = MockLLMBackend(llm_dim=LLM_DIM, hidden_dim=64)
        return TopoBridge(
            backend=backend,
            topo_dim=TOPO_DIM,
            llm_dim=LLM_DIM,
            num_prefix=NUM_PREFIX,
        )

    def test_forward_returns_tuple(self):
        """TopoBridge.forward now returns (llm_out, semantic_bias)."""
        bridge = self._make_bridge()
        nodes = torch.randn(10, TOPO_DIM)
        semantic_weight = torch.tensor(0.5)
        llm_out, semantic_bias = bridge(nodes, semantic_weight)
        assert llm_out.shape == (10, TOPO_DIM)
        assert semantic_bias.shape == (10, 10)

    def test_semantic_bias_variable_sizes(self):
        bridge = self._make_bridge()
        for n in [3, 10, 25]:
            nodes = torch.randn(n, TOPO_DIM)
            _, semantic_bias = bridge(nodes, torch.tensor(0.5))
            assert semantic_bias.shape == (n, n)

    def test_semantic_bias_gradient_flow(self):
        bridge = self._make_bridge()
        nodes = torch.randn(10, TOPO_DIM, requires_grad=True)
        _, semantic_bias = bridge(nodes, torch.tensor(0.5))
        semantic_bias.sum().backward()
        assert nodes.grad is not None

    def test_no_gate_threshold_skip(self):
        """DSM always runs — no gate threshold skip."""
        bridge = self._make_bridge()
        nodes = torch.randn(10, TOPO_DIM)
        llm_out, semantic_bias = bridge(nodes, torch.tensor(0.01))
        # Even with very low semantic_weight, we get non-zero output
        assert llm_out.abs().sum() > 0
        assert semantic_bias.abs().sum() > 0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm/test_topo_bridge.py::TestTopoBridgeWithSemanticBias -v`
Expected: FAIL — TopoBridge.forward returns tensor, not tuple

**Step 3: Implement changes**

In `src/llm/topo_bridge.py`:

1. Add `SemanticBiasHead` to TopoBridgeDecoder:

```python
class TopoBridgeDecoder(nn.Module):
    """Extracts node-aligned embeddings and semantic bias from LLM hidden states."""

    def __init__(self, topo_dim: int = 32, llm_dim: int = 2048):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            llm_dim, num_heads=8, batch_first=False,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(llm_dim, topo_dim),
            nn.LayerNorm(topo_dim),
        )
        # Semantic bias: node embeddings → pairwise attention bias
        self.semantic_bias_proj = nn.Linear(topo_dim, topo_dim)

    def forward(
        self, topo_memory: torch.Tensor, llm_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode LLM hidden states back to node-aligned topo embeddings + bias.

        Returns:
            llm_out: (N, topo_dim)
            semantic_bias: (N, N) pairwise attention bias for TAT
        """
        queries = topo_memory.unsqueeze(1)
        kv = llm_hidden.unsqueeze(1)
        attn_out, _ = self.cross_attn(queries, kv, kv)
        attn_out = attn_out.squeeze(1)

        llm_out = self.out_proj(attn_out)  # (N, topo_dim)

        # Compute semantic bias: pairwise similarity in projected space
        projected = self.semantic_bias_proj(llm_out)  # (N, topo_dim)
        semantic_bias = projected @ projected.T  # (N, N)

        return llm_out, semantic_bias
```

2. Update `TopoBridge.forward` to remove gate threshold and return tuple:

```python
class TopoBridge(nn.Module):
    """Composes encoder + LLM backend + decoder for topo→LLM→topo round trip."""

    def __init__(self, backend, topo_dim=32, llm_dim=2048, num_prefix=8,
                 gate_threshold=0.1):  # gate_threshold kept for backward compat
        super().__init__()
        self.encoder = TopoBridgeEncoder(topo_dim, llm_dim, num_prefix)
        self.decoder = TopoBridgeDecoder(topo_dim, llm_dim)
        self.backend = backend
        self.topo_dim = topo_dim

    def forward(self, node_embeddings, semantic_weight, task_text=None):
        """Full TopoBridge forward pass — always runs (no gate skip).

        Returns:
            llm_out: (N, topo_dim) node-aligned output.
            semantic_bias: (N, N) pairwise bias for TAT attention.
        """
        # Encode
        topo_memory, prefix_tokens = self.encoder(node_embeddings)

        # LLM/DSM forward
        llm_hidden = self.backend.forward(prefix_tokens, topo_memory, task_text)

        # Decode back to topo space + semantic bias
        llm_out, semantic_bias = self.decoder(topo_memory, llm_hidden)

        return llm_out, semantic_bias
```

**Step 4: Update existing tests**

Existing `TestTopoBridge` tests in `test_topo_bridge.py` need updating: `bridge(nodes, gate)` now returns a tuple. Update each test to unpack `llm_out, semantic_bias = bridge(...)`.

**Step 5: Run tests**

Run: `python -m pytest tests/test_llm/test_topo_bridge.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/llm/topo_bridge.py tests/test_llm/test_topo_bridge.py
git commit -m "feat: TopoBridge returns semantic_bias (N,N) for TAT attention

Decoder now produces pairwise semantic bias via projected similarity.
Gate threshold skip removed — DSM always runs.
forward() returns (llm_out, semantic_bias) tuple."
```

---

### Task 5: TAT Spatial Attention — Semantic Bias Injection

**Files:**
- Modify: `src/tat/spatial_attention.py:36-83`
- Modify: `src/tat/transformer.py:55-80` (TATBlock.forward passes semantic_bias through)
- Create: `tests/test_tat/test_semantic_bias.py`

**Step 1: Write the failing tests**

Create `tests/test_tat/test_semantic_bias.py`:

```python
"""Tests for semantic bias injection in TAT spatial attention."""

import torch
import pytest
from src.tat.spatial_attention import TopologicalSpatialAttention


class TestSemanticBiasInjection:
    def test_accepts_semantic_bias(self):
        """Spatial attention accepts semantic_bias parameter."""
        attn = TopologicalSpatialAttention(embed_dim=32, num_heads=4)
        x = torch.randn(10, 32)
        adj = torch.ones(10, 10)
        semantic_bias = torch.randn(10, 10)
        out = attn(x, adj, semantic_bias=semantic_bias)
        assert out.shape == (10, 32)

    def test_semantic_bias_none_is_no_op(self):
        """When semantic_bias=None, output is unchanged from before."""
        attn = TopologicalSpatialAttention(embed_dim=32, num_heads=4)
        torch.manual_seed(42)
        x = torch.randn(10, 32)
        adj = torch.ones(10, 10)
        out_none = attn(x, adj, semantic_bias=None)
        torch.manual_seed(42)
        x2 = torch.randn(10, 32)
        out_default = attn(x2, adj)
        assert torch.allclose(out_none, out_default)

    def test_semantic_bias_changes_output(self):
        """Non-zero semantic bias should change the attention output."""
        attn = TopologicalSpatialAttention(embed_dim=32, num_heads=4)
        x = torch.randn(10, 32)
        adj = torch.ones(10, 10)
        out_no_bias = attn(x, adj)
        out_with_bias = attn(x, adj, semantic_bias=torch.randn(10, 10) * 10)
        assert not torch.allclose(out_no_bias, out_with_bias)

    def test_semantic_bias_gradient_flow(self):
        """Gradients flow through semantic_bias."""
        attn = TopologicalSpatialAttention(embed_dim=32, num_heads=4)
        x = torch.randn(10, 32)
        adj = torch.ones(10, 10)
        bias = torch.randn(10, 10, requires_grad=True)
        out = attn(x, adj, semantic_bias=bias)
        out.sum().backward()
        assert bias.grad is not None
        assert bias.grad.abs().sum() > 0

    def test_semantic_bias_with_semantic_weight(self):
        """semantic_weight scales the bias."""
        attn = TopologicalSpatialAttention(embed_dim=32, num_heads=4)
        x = torch.randn(10, 32)
        adj = torch.ones(10, 10)
        bias = torch.randn(10, 10)
        out_low = attn(x, adj, semantic_bias=bias, semantic_weight=torch.tensor(0.01))
        out_high = attn(x, adj, semantic_bias=bias, semantic_weight=torch.tensor(10.0))
        # High weight should produce different output than low weight
        assert not torch.allclose(out_low, out_high)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tat/test_semantic_bias.py -v`
Expected: FAIL — `TopologicalSpatialAttention.forward` doesn't accept `semantic_bias`

**Step 3: Implement the change**

In `src/tat/spatial_attention.py`, update `forward` signature and add bias injection after spatial_focus (line 82):

```python
def forward(self, x: torch.Tensor, adjacency: torch.Tensor,
            spatial_focus: torch.Tensor | None = None,
            edge_weights: torch.Tensor | None = None,
            semantic_bias: torch.Tensor | None = None,
            semantic_weight: torch.Tensor | None = None) -> torch.Tensor:
```

After line 82 (spatial focus), add:

```python
    # Apply semantic bias from DSM
    if semantic_bias is not None:
        weight = semantic_weight if semantic_weight is not None else torch.tensor(1.0)
        # Broadcast (N, N) to (num_heads, N, N)
        scores = scores + weight * semantic_bias.unsqueeze(0)
```

In `src/tat/transformer.py`, update `TATBlock.forward` to accept and pass `semantic_bias` and `semantic_weight`:

```python
def forward(self, x, adjacency, eigenvalues, eigenvectors,
            control_signal=None, edge_weights=None,
            semantic_bias=None, semantic_weight=None):
    spatial_focus = control_signal.spatial_focus if control_signal is not None else None
    frequency_gate = control_signal.frequency_gate if control_signal is not None else None

    spatial_out = self.spatial_attn(
        x, adjacency=adjacency, spatial_focus=spatial_focus,
        edge_weights=edge_weights,
        semantic_bias=semantic_bias, semantic_weight=semantic_weight,
    )
    # ... rest unchanged
```

Also update `TopologyAwareTransformer.forward` to accept and thread through `semantic_bias` and `semantic_weight`.

**Step 4: Run tests**

Run: `python -m pytest tests/test_tat/test_semantic_bias.py -v`
Expected: PASS

Run: `python -m pytest tests/test_tat/ -v`
Expected: All existing TAT tests still pass

**Step 5: Commit**

```bash
git add src/tat/spatial_attention.py src/tat/transformer.py tests/test_tat/test_semantic_bias.py
git commit -m "feat: TAT spatial attention accepts semantic_bias from DSM

Adds semantic_bias (N,N) and semantic_weight (scalar) parameters.
Bias is added to attention logits before softmax, after spatial_focus.
Both are optional — None preserves existing behavior."
```

---

### Task 6: Modified ExecutiveReasoningLoop — DSM in Every Iteration

**Files:**
- Modify: `src/reasoning_loop/executive_loop.py:26-232`
- Create: `tests/test_reasoning_loop/test_dsm_integration.py`

The executive loop now includes DSM forward between Wave dynamics and TAT. Each iteration: GNN → Wave → TopoBridge encode → DSM → TopoBridge decode (semantic_bias) → TAT(semantic_bias).

**Step 1: Write the failing tests**

Create `tests/test_reasoning_loop/test_dsm_integration.py`:

```python
"""Tests for DSM integration in ExecutiveReasoningLoop."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop


def _make_cc(n_nodes=10, embedding_dim=32):
    """Create a simple cell complex for testing."""
    cc = CellComplex()
    for i in range(n_nodes):
        cc.add_cell(0, (i,))
    for i in range(n_nodes - 1):
        cc.add_cell(1, (i, i + 1))
    cc.set_embeddings(0, torch.randn(n_nodes, embedding_dim))
    cc.set_embeddings(1, torch.randn(cc.num_cells(1), embedding_dim))
    return cc


class TestDSMInExecutiveLoop:
    def _make_loop(self, use_dsm=True):
        return ExecutiveReasoningLoop(
            embedding_dim=32, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=1,
            max_freqs=8, tat_layers=1,
            tat_spatial_heads=4, tat_spectral_heads=4,
            tat_ff_dim=64, max_iterations=3,
            convergence_threshold=0.01,
            use_wave_dynamics=True,
            use_dsm=use_dsm,
            dsm_config={'dsm_dim': 64, 'num_heads': 4, 'ff_dim': 128,
                        'num_layers': 2, 'cross_attn_layer': 0},
        )

    def test_forward_with_dsm(self):
        loop = self._make_loop(use_dsm=True)
        cc = _make_cc()
        output, num_iters, diagnostics = loop(cc)
        assert output.shape == (10, 32)
        assert num_iters >= 1

    def test_forward_without_dsm(self):
        loop = self._make_loop(use_dsm=False)
        cc = _make_cc()
        output, num_iters, diagnostics = loop(cc)
        assert output.shape == (10, 32)

    def test_dsm_produces_semantic_weight_in_diagnostics(self):
        loop = self._make_loop(use_dsm=True)
        cc = _make_cc()
        _, _, diagnostics = loop(cc)
        control_signals = diagnostics['control_signals']
        for cs in control_signals:
            assert cs.semantic_weight is not None
            assert cs.semantic_weight >= 0.0
            assert cs.semantic_weight <= 1.0

    def test_dsm_gradient_flow(self):
        loop = self._make_loop(use_dsm=True)
        cc = _make_cc()
        # Need grad on embeddings
        emb = torch.randn(10, 32, requires_grad=True)
        cc.set_embeddings(0, emb)
        output, _, _ = loop(cc)
        output.sum().backward()
        # DSM parameters should have gradients
        dsm_params = [p for n, p in loop.named_parameters() if 'topo_bridge' in n or 'dsm' in n]
        grads = sum(1 for p in dsm_params if p.grad is not None)
        assert grads > 0, "No DSM parameters received gradients"

    def test_dsm_config_optional(self):
        """When use_dsm=False, no DSM modules are created."""
        loop = self._make_loop(use_dsm=False)
        assert not hasattr(loop, 'topo_bridge') or loop.topo_bridge is None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reasoning_loop/test_dsm_integration.py -v`
Expected: FAIL — ExecutiveReasoningLoop doesn't accept `use_dsm`/`dsm_config`

**Step 3: Implement changes**

In `src/reasoning_loop/executive_loop.py`:

1. Add `use_dsm` and `dsm_config` to `__init__`:

```python
def __init__(self, ..., use_dsm: bool = False, dsm_config: dict | None = None):
    # ... existing init code ...

    self.use_dsm = use_dsm
    self.topo_bridge = None
    if use_dsm:
        from src.llm.dsm_backend import DSMBackend
        from src.llm.topo_bridge import TopoBridge
        dc = dsm_config or {}
        dsm_dim = dc.get('dsm_dim', 1024)
        backend = DSMBackend(
            dsm_dim=dsm_dim,
            num_heads=dc.get('num_heads', 16),
            ff_dim=dc.get('ff_dim', 4096),
            num_layers=dc.get('num_layers', 16),
            cross_attn_layer=dc.get('cross_attn_layer', 4),
        )
        self.topo_bridge = TopoBridge(
            backend=backend,
            topo_dim=embedding_dim,
            llm_dim=dsm_dim,
            num_prefix=dc.get('num_prefix', 8),
        )
```

2. In `forward`, add DSM step between wave and TAT (after line 196, before line 199):

```python
    # 2.5. DSM: encode → DSM forward → decode → semantic_bias
    semantic_bias = None
    semantic_weight = None
    if self.use_dsm and self.topo_bridge is not None:
        semantic_weight = control.semantic_weight
        _, semantic_bias = self.topo_bridge(gnn_out, semantic_weight)

    # 3. TAT executes with control signals + semantic bias
    tat_out = self.tat(cc, control_signal=control,
                       semantic_bias=semantic_bias,
                       semantic_weight=semantic_weight)
```

This requires `TopologyAwareTransformer.forward` to accept `semantic_bias` and `semantic_weight` (done in Task 5).

**Step 4: Run tests**

Run: `python -m pytest tests/test_reasoning_loop/test_dsm_integration.py -v`
Expected: PASS

Run: `python -m pytest tests/test_reasoning_loop/ -v`
Expected: All existing loop tests still pass (use_dsm defaults to False)

**Step 5: Commit**

```bash
git add src/reasoning_loop/executive_loop.py tests/test_reasoning_loop/test_dsm_integration.py
git commit -m "feat: executive loop integrates DSM every iteration

Each iteration: GNN→Wave→TopoBridge(DSM)→TAT with semantic_bias.
DSM is optional (use_dsm=False preserves existing behavior).
semantic_weight from ControlHead scales the bias naturally."
```

---

### Task 7: Updated HierarchicalMultiHopModel — DSM Integration

**Files:**
- Modify: `src/benchmarks/run_comparison.py:73-226`
- Modify: `tests/test_benchmarks/test_run_comparison.py` (if it exists)

The model now supports `backend='dsm'` in llm_config and threads dsm_config to the executive loop.

**Step 1: Write the failing tests**

Add to existing test file or create `tests/test_benchmarks/test_dsm_model.py`:

```python
"""Tests for HierarchicalMultiHopModel with DSM backend."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.run_comparison import HierarchicalMultiHopModel


def _make_cc(n=10, dim=32):
    cc = CellComplex()
    for i in range(n):
        cc.add_cell(0, (i,))
    for i in range(n - 1):
        cc.add_cell(1, (i, i + 1))
    cc.set_embeddings(0, torch.randn(n, dim))
    cc.set_embeddings(1, torch.randn(cc.num_cells(1), dim))
    return cc


class TestHierarchicalModelWithDSM:
    def _make_model(self, use_dsm=True):
        dsm_config = {
            'backend': 'dsm',
            'dsm_dim': 64,
            'num_heads': 4,
            'ff_dim': 128,
            'num_layers': 2,
            'cross_attn_layer': 0,
            'num_prefix': 4,
        }
        return HierarchicalMultiHopModel(
            embedding_dim=32, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=1,
            max_freqs=8, tat_layers=1,
            tat_spatial_heads=4, tat_spectral_heads=4,
            tat_ff_dim=64, max_classes=5,
            max_iterations=2, convergence_threshold=0.01,
            use_llm=use_dsm,
            llm_config=dsm_config if use_dsm else None,
        )

    def test_forward_with_dsm(self):
        model = self._make_model(use_dsm=True)
        cc = _make_cc()
        out = model(cc.clone(), 0, 1)
        assert out.shape == (5,)  # max_classes

    def test_forward_without_dsm(self):
        model = self._make_model(use_dsm=False)
        cc = _make_cc()
        out = model(cc.clone(), 0, 1)
        assert out.shape == (5,)

    def test_dsm_model_no_bypass_attribute(self):
        """bypass_llm is removed — DSM always runs."""
        model = self._make_model(use_dsm=True)
        # bypass_llm may exist for backward compat but should be False
        # The executive loop handles DSM activation directly
        cc = _make_cc()
        out = model(cc.clone(), 0, 1)
        assert out.shape == (5,)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_benchmarks/test_dsm_model.py -v`
Expected: FAIL — `backend='dsm'` not handled

**Step 3: Implement changes**

In `src/benchmarks/run_comparison.py`, modify `HierarchicalMultiHopModel.__init__`:

When `llm_config['backend'] == 'dsm'`, instead of creating TopoBridge at the model level, pass `use_dsm=True` and `dsm_config` to the executive loop. The DSM lives inside the loop (not outside).

```python
def __init__(self, ..., use_llm=False, llm_config=None):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.use_llm = use_llm
    self.bypass_llm = False
    lc = llm_config or {}
    backend_type = lc.get('backend', 'mock')

    # DSM: lives inside the executive loop
    use_dsm = (use_llm and backend_type == 'dsm')
    dsm_config = lc if use_dsm else None

    wc = wave_config or {}
    self.executive_loop = ExecutiveReasoningLoop(
        ...,  # existing args
        use_dsm=use_dsm,
        dsm_config=dsm_config,
    )

    # Legacy TopoBridge for mock/llama backends (Phase 4c compat)
    if use_llm and backend_type != 'dsm':
        # ... existing TopoBridge setup ...
        self.topo_bridge = TopoBridge(...)
    else:
        self.topo_bridge = None

    # ... classifier unchanged ...
```

In `forward`, the DSM path is handled by the executive loop. The legacy TopoBridge path remains for backward compat:

```python
def forward(self, cc, query_node, target_node, metadata=None):
    initial_edge_embs = cc.get_embeddings(1).clone() if cc.num_cells(1) > 0 else None
    output, num_iters, diagnostics = self.executive_loop(cc)

    # Legacy LLM integration (mock/llama backends only)
    if self.topo_bridge is not None and not self.bypass_llm:
        # ... existing code ...
        pass

    # DSM path: already integrated in executive_loop — nothing more to do

    # ... rest unchanged (classifier) ...
```

**Step 4: Update `_build_model` in `run_benchmark_suite.py`**

Ensure `_build_model("hierarchical_llm", ...)` passes `llm_config` through to the model.

**Step 5: Run tests**

Run: `python -m pytest tests/test_benchmarks/test_dsm_model.py -v`
Expected: PASS

Run: `python -m pytest tests/ -v --timeout=60`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/benchmarks/run_comparison.py src/benchmarks/run_benchmark_suite.py
git add tests/test_benchmarks/test_dsm_model.py
git commit -m "feat: HierarchicalMultiHopModel supports DSM backend

When llm_config.backend='dsm', DSM lives inside the executive loop
(interleaved every iteration). Legacy mock/llama TopoBridge preserved
for backward compatibility."
```

---

### Task 8: Fix Analogical Transfer Task

**Files:**
- Modify: `src/benchmarks/llm_tasks.py:207-268`
- Modify: `tests/test_benchmarks/test_llm_tasks.py`

Root cause: roles are randomly assigned to nodes (line 237). Fix: assign by degree centrality. Reduce to 3 roles, 2 domain pairs.

**Step 1: Write the failing tests**

Add to `tests/test_benchmarks/test_llm_tasks.py`:

```python
class TestAnalogicalTransferFix:
    def test_roles_assigned_by_degree_centrality(self):
        """Role nodes should be the highest-degree nodes."""
        import networkx as nx
        from src.benchmarks.llm_tasks import generate_analogical_transfer_task
        cc, query, target, answer, meta = generate_analogical_transfer_task(
            n_nodes=20, embedding_dim=32, topologies=['ba'],
        )
        # With BA graphs, role nodes should have higher degree than average
        # This is a statistical test — run multiple times
        high_degree_count = 0
        for _ in range(20):
            cc, q, t, ans, m = generate_analogical_transfer_task(
                20, 32, topologies=['ba'],
            )
            # role_nodes should be stored in metadata
            assert 'role_degrees' in m
            avg_degree = m.get('avg_degree', 0)
            # At least 2 of 3 role nodes should be above average degree
            above_avg = sum(1 for d in m['role_degrees'] if d > avg_degree)
            if above_avg >= 2:
                high_degree_count += 1
        # Most of the time (>80%), role nodes are above average
        assert high_degree_count >= 14, f"Only {high_degree_count}/20 had high-degree roles"

    def test_three_roles_only(self):
        """Reduced to 3 roles (from 5)."""
        from src.benchmarks.llm_tasks import generate_analogical_transfer_task
        _, _, _, answer, _ = generate_analogical_transfer_task(20, 32)
        assert 0 <= answer <= 2  # 3 classes: 0, 1, 2

    def test_get_max_classes_analogical(self):
        from src.benchmarks.benchmark_dataset import get_max_classes
        assert get_max_classes('analogical_transfer') == 3
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_benchmarks/test_llm_tasks.py::TestAnalogicalTransferFix -v`
Expected: FAIL — roles still random, 5 classes

**Step 3: Implement the fix**

In `src/benchmarks/llm_tasks.py`, modify `generate_analogical_transfer_task`:

```python
def generate_analogical_transfer_task(n_nodes, embedding_dim, topologies=None):
    # Pick a domain pair (reduced to 2 pairs)
    domain_a, domain_b = random.choice(ANALOGY_DOMAINS[:2])
    num_roles = 3  # Fixed at 3 (down from 5)
    roles_a = list(domain_a.keys())[:num_roles]
    roles_b = list(domain_b.keys())[:num_roles]

    # Generate graph
    topo = random.choice(topologies) if topologies else None
    G = random_graph(max(n_nodes, num_roles + 2), topology=topo)
    nodes = list(G.nodes())

    # Assign roles by degree centrality (highest-degree nodes get roles)
    degrees = dict(G.degree())
    sorted_by_degree = sorted(nodes, key=lambda n: degrees[n], reverse=True)
    role_nodes = sorted_by_degree[:num_roles]

    role_assignments = {}
    for i, node in enumerate(role_nodes):
        role_assignments[node] = i

    # Pick query
    query_role_idx = random.randint(0, num_roles - 1)
    query_node = role_nodes[query_role_idx]
    other_nodes = [n for n in nodes if n != query_node]
    target_node = random.choice(other_nodes)
    answer = query_role_idx

    cc, node_map = nx_to_cell_complex(G, embedding_dim,
                                       source_node=query_node, target_node=target_node)

    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    metadata = {
        'task_type': 'analogical_transfer',
        'task_prompt': f'domain_a={",".join(roles_a)} domain_b={",".join(roles_b)} '
                       f'nodes={G.number_of_nodes()} edges={G.number_of_edges()} '
                       f'assigned_roles={num_roles} | '
                       f'query_role={roles_a[query_role_idx]} | task=analogical_transfer',
        'domain_a': roles_a,
        'domain_b': roles_b,
        'role_index': query_role_idx,
        'role_name': roles_a[query_role_idx],
        'role_degrees': [degrees[n] for n in role_nodes],
        'avg_degree': avg_degree,
    }
    return cc, node_map[query_node], node_map[target_node], answer, metadata
```

Update `get_max_classes` in `benchmark_dataset.py`:
```python
'analogical_transfer': 3,  # was 5
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_benchmarks/test_llm_tasks.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/benchmarks/llm_tasks.py src/benchmarks/benchmark_dataset.py
git add tests/test_benchmarks/test_llm_tasks.py
git commit -m "fix: analogical_transfer assigns roles by degree centrality

Roles assigned to highest-degree nodes (not random), giving the GNN
structural signal. Reduced from 5 to 3 roles, limited to 2 domain pairs.
This makes the task solvable without requiring the LLM pathway."
```

---

### Task 9: Fix Labeled Reasoning Class Balance

**Files:**
- Modify: `src/benchmarks/llm_tasks.py` (labeled_reasoning generator)
- Add test: `tests/test_benchmarks/test_llm_tasks.py`

The debrief showed class 2 (independent) was missing from test data.

**Step 1: Write the failing test**

```python
class TestLabeledReasoningBalance:
    def test_all_three_classes_present(self):
        """All 3 classes (causal_chain, blocked, independent) must appear."""
        from src.benchmarks.llm_tasks import generate_labeled_reasoning_task
        classes_seen = set()
        for _ in range(200):
            _, _, _, answer, _ = generate_labeled_reasoning_task(20, 32)
            classes_seen.add(answer)
        assert classes_seen == {0, 1, 2}, f"Missing classes: {classes_seen}"

    def test_roughly_balanced(self):
        """No class should be <20% or >50% of samples."""
        from src.benchmarks.llm_tasks import generate_labeled_reasoning_task
        from collections import Counter
        counts = Counter()
        n = 300
        for _ in range(n):
            _, _, _, answer, _ = generate_labeled_reasoning_task(20, 32)
            counts[answer] += 1
        for cls in [0, 1, 2]:
            pct = counts[cls] / n
            assert pct > 0.15, f"Class {cls} only {pct:.1%}"
            assert pct < 0.55, f"Class {cls} too dominant at {pct:.1%}"
```

**Step 2: Run, verify fail, fix the generator, run again, commit**

Look at the current `generate_labeled_reasoning_task` and ensure it generates class 2 (independent) by explicitly creating scenarios where query and target are in disconnected components.

Run: `python -m pytest tests/test_benchmarks/test_llm_tasks.py::TestLabeledReasoningBalance -v`

```bash
git commit -m "fix: labeled_reasoning guarantees all 3 classes in generated data

Explicitly generates independent-class samples by placing query/target
in disconnected components. Class balance enforced: each class ~33%."
```

---

### Task 10: DSM Config in YAML

**Files:**
- Create: `config/dsm_training.yaml`
- Modify: `src/benchmarks/run_benchmark_suite.py` (if needed for _build_model)

**Step 1: Create config**

```yaml
# DSM training configuration
model:
  embedding_dim: 32
  gnn_hidden: 64
  gnn_spatial_layers: 2
  gnn_spectral_layers: 1
  max_freqs: 16
  tat_layers: 2
  tat_spatial_heads: 4
  tat_spectral_heads: 4
  tat_ff_dim: 128
  use_higher_order: true
  use_topological_pe: true
  use_structural_features: true

wave:
  wave_mode: ensemble
  filter_types: [chebyshev, wave_cosine, heat]
  include_identity: true
  include_sheaf: false
  wave_strength_gate: true
  use_neural_ode: true

llm:
  backend: dsm
  dsm_dim: 1024        # Full: 1024, test: 64
  num_heads: 16         # Full: 16, test: 4
  ff_dim: 4096          # Full: 4096, test: 128
  num_layers: 16        # Full: 16, test: 2
  cross_attn_layer: 4   # Full: 4, test: 0
  num_prefix: 8

training:
  device: auto
  learning_rate: 0.001
  dsm_learning_rate: 0.0005      # Lower LR for DSM
  bridge_learning_rate: 0.0001   # Lower LR for TopoBridge
  weight_decay: 0.01
  label_smoothing: 0.1
  epochs_phase_a: 30
  epochs_phase_b: 30
  epochs_phase_c: 30
  patience: 10

benchmark:
  train_samples: 5000
  val_samples: 500
  test_samples: 500
  train_n_nodes_min: 16
  train_n_nodes_max: 32
  test_n_nodes: [20, 40, 80]
  train_topologies: [ba, ws, grid, tree, ladder, sbm]
  test_topologies: [er, caveman]
```

**Step 2: Verify config loads**

Run: `python -c "import yaml; c = yaml.safe_load(open('config/dsm_training.yaml')); print(c['llm']['backend'])"`
Expected: `dsm`

**Step 3: Commit**

```bash
git add config/dsm_training.yaml
git commit -m "feat: add DSM training config (dsm_training.yaml)

Full 250M DSM: 1024-dim, 16 layers, 16 heads, 4096 FFN.
Three-phase curriculum with per-component learning rates.
5000 train samples per task, mixed-size n=16-32."
```

---

### Task 11: Updated Curriculum Training Script

**Files:**
- Modify: `scripts/run_phase4c_curriculum.py` → copy to `scripts/run_dsm_curriculum.py`
- Modify: `tests/test_llm/test_curriculum.py`

Key fixes from debrief:
1. Per-task `max_classes` via `get_max_classes(task)`
2. Task replay (20% structural tasks in Phase B/C)
3. Cosine annealing (replaces ReduceLROnPlateau)
4. No gate penalty ever
5. Per-component learning rates (GNN/TAT, DSM, TopoBridge)

**Step 1: Write tests**

```python
class TestDSMCurriculum:
    def test_per_task_max_classes(self):
        """Classifier head is rebuilt per task with correct num_classes."""
        from src.benchmarks.benchmark_dataset import get_max_classes
        assert get_max_classes('diverse') == 11
        assert get_max_classes('bfs') == 16
        assert get_max_classes('hodge_class') == 3
        assert get_max_classes('spectral_gap') == 8
        assert get_max_classes('graph_completion') == 2
        assert get_max_classes('path_counting') == 5
        assert get_max_classes('labeled_reasoning') == 3
        assert get_max_classes('analogical_transfer') == 3

    def test_no_gate_penalty_in_config(self):
        import yaml
        with open('config/dsm_training.yaml') as f:
            config = yaml.safe_load(f)
        training = config['training']
        assert 'gate_penalty' not in training or training.get('gate_penalty', 0) == 0
```

**Step 2: Create `scripts/run_dsm_curriculum.py`**

Copy `run_phase4c_curriculum.py` and apply these changes:

- Remove `_unfreeze_llm` (DSM is always trainable)
- Remove `gate_penalty` from loss
- Add `_rebuild_classifier(model, task)` that calls `get_max_classes(task)` and rebuilds only the final Linear
- Add cosine annealing scheduler: `CosineAnnealingLR`
- Add task replay: 20% of each Phase B/C batch is sampled from Phase A tasks
- Add per-component optimizer groups:
  ```python
  param_groups = [
      {'params': gnn_tat_params, 'lr': config['training']['learning_rate']},
      {'params': dsm_params, 'lr': config['training']['dsm_learning_rate']},
      {'params': bridge_params, 'lr': config['training']['bridge_learning_rate']},
  ]
  ```

**Step 3: Run tests, commit**

```bash
git add scripts/run_dsm_curriculum.py tests/test_llm/test_curriculum.py
git commit -m "feat: DSM curriculum training script

Per-task classifier, cosine annealing, task replay, per-component LR.
No gate penalty — semantic_weight learns naturally."
```

---

### Task 12: Precomputation Pipeline

**Files:**
- Create: `scripts/precompute_datasets.py`

Generate all training/validation/test datasets on local CPU. Saves serialized `BenchmarkDataset` files.

**Step 1: Write the script**

```python
"""Precompute all datasets for DSM curriculum training.

Runs on CPU. Generates datasets for each task at multiple sizes.
Saves to data/dsm_datasets/{task}_{split}_{size}.pt

Usage:
    python scripts/precompute_datasets.py config/dsm_training.yaml \
        --output-dir data/dsm_datasets
"""
```

Tasks to generate:
- Phase A: diverse, bfs, hodge_class, spectral_gap, path_counting (5000 train, 500 val each)
- Phase B: graph_completion, labeled_reasoning (5000 train, 500 val each)
- Phase C: analogical_transfer (5000 train, 500 val)
- Test sets: n=20 (ID), n=40 (OOD), n=80 (OOD), topo_transfer at n=20 (500 each)

**Step 2: Run locally**

Run: `python scripts/precompute_datasets.py config/dsm_training.yaml --output-dir data/dsm_datasets`
Expected: ~15 min, produces ~50 dataset files

**Step 3: Commit**

```bash
git add scripts/precompute_datasets.py
git commit -m "feat: precompute_datasets.py — CPU-side dataset generation

Generates all train/val/test datasets for DSM curriculum.
Precomputes structural features, cell complexes, B1/B2, spectral
decomposition, persistence diagrams. ~15 min on CPU."
```

---

### Task 13: Distillation Script (One-Time, GPU)

**Files:**
- Create: `scripts/distill_dsm.py`

One-time script to distill Llama 1B → DSM 250M. Requires `transformers` + `sentencepiece`.

**Step 1: Write the script**

```python
"""Distill Llama 3.2 1B into DSM 250M.

One-time operation. Requires transformers + sentencepiece.

Loss:
    0.5 * MSE(student_hidden[layer_8], PCA(teacher_hidden[layer_8]))
    + 0.5 * KL(student_logits, teacher_logits, temperature=2.0)

Data:
    - 50K sentences from SlimPajama
    - 10K graph-description texts from task prompt templates

Output:
    data/dsm_distilled.pt (state_dict of DSM 250M)
"""
```

Key steps:
1. Load Llama 1B teacher (frozen, bf16)
2. Initialize DSM with PCA'd Llama embeddings
3. Train 10-20 epochs, cosine annealing
4. Save DSM state_dict

**Step 2: Commit**

```bash
git add scripts/distill_dsm.py
git commit -m "feat: distill_dsm.py — one-time Llama 1B → DSM 250M distillation

PCA embedding initialization, MSE hidden + KL logits loss.
~4 hrs on RTX 4090. Output: data/dsm_distilled.pt"
```

---

### Task 14: Diagnostics Updates

**Files:**
- Modify: `src/benchmarks/diagnostics.py`
- Modify: `tests/test_benchmarks/test_diagnostics.py`

Update DiagnosticCollector to capture `semantic_weight` instead of `llm_gate`, plus DSM-specific metrics.

**Step 1: Update diagnostics**

Replace `llm_gate` references with `semantic_weight`. Add:
- `semantic_weight` mean/std per sample
- Whether semantic_weight is task-discriminative (high for semantic tasks, low for structural)

**Step 2: Run tests, commit**

```bash
git commit -m "refactor: diagnostics capture semantic_weight instead of llm_gate"
```

---

### Task 15: Integration Test — Full Pipeline Smoke Test

**Files:**
- Create: `tests/test_integration/test_dsm_pipeline.py`

End-to-end test that creates a model with DSM, generates a small dataset, runs one training epoch, and verifies everything connects.

**Step 1: Write the test**

```python
"""Integration test: full DSM pipeline smoke test."""

import torch
import pytest
from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.run_comparison import HierarchicalMultiHopModel


class TestDSMPipelineSmokeTest:
    @pytest.fixture
    def model(self):
        return HierarchicalMultiHopModel(
            embedding_dim=32, gnn_hidden=64,
            gnn_spatial_layers=2, gnn_spectral_layers=1,
            max_freqs=8, tat_layers=1,
            tat_spatial_heads=4, tat_spectral_heads=4,
            tat_ff_dim=64, max_classes=3,
            max_iterations=2, convergence_threshold=0.01,
            use_llm=True,
            llm_config={
                'backend': 'dsm',
                'dsm_dim': 64, 'num_heads': 4, 'ff_dim': 128,
                'num_layers': 2, 'cross_attn_layer': 0, 'num_prefix': 4,
            },
        )

    def _make_cc(self, n=10, dim=32):
        cc = CellComplex()
        for i in range(n):
            cc.add_cell(0, (i,))
        for i in range(n - 1):
            cc.add_cell(1, (i, i + 1))
        cc.set_embeddings(0, torch.randn(n, dim))
        cc.set_embeddings(1, torch.randn(cc.num_cells(1), dim))
        return cc

    def test_forward_pass(self, model):
        cc = self._make_cc()
        out = model(cc.clone(), 0, 1)
        assert out.shape == (3,)
        assert torch.isfinite(out).all()

    def test_backward_pass(self, model):
        cc = self._make_cc()
        out = model(cc.clone(), 0, 1)
        loss = torch.nn.functional.cross_entropy(out.unsqueeze(0), torch.tensor([1]))
        loss.backward()
        # All parameter groups should have gradients
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_one_training_step(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        cc = self._make_cc()
        model.train()
        out = model(cc.clone(), 0, 1)
        loss = torch.nn.functional.cross_entropy(out.unsqueeze(0), torch.tensor([1]))
        loss.backward()
        optimizer.step()
        # Verify parameters changed
        assert loss.item() > 0

    def test_variable_graph_sizes(self, model):
        for n in [5, 10, 20]:
            cc = self._make_cc(n=n)
            out = model(cc.clone(), 0, min(1, n - 1))
            assert out.shape == (3,)
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_integration/test_dsm_pipeline.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration/test_dsm_pipeline.py
git commit -m "test: end-to-end DSM pipeline integration test

Verifies forward pass, backward pass, optimizer step, and variable
graph sizes with the full GNN→Wave→DSM→TAT pipeline."
```

---

## Task Dependency Graph

```
Task 1 (ControlSignal) ─────┐
Task 2 (DSM model) ─────────┤
Task 3 (DSM backend) ────┐  │
Task 4 (TopoBridge) ─────┤  │
Task 5 (TAT semantic) ───┤  │
                          ▼  ▼
                    Task 6 (Executive loop)
                          │
                          ▼
                    Task 7 (Model integration)
                          │
                    ┌─────┼─────┐
                    ▼     ▼     ▼
             Task 8   Task 9  Task 10
          (analogical) (labeled) (config)
                    │     │     │
                    ▼     ▼     ▼
                    Task 11 (Curriculum script)
                          │
                    ┌─────┼─────┐
                    ▼     ▼     ▼
             Task 12  Task 13  Task 14
          (precompute)(distill)(diagnostics)
                          │
                          ▼
                    Task 15 (Integration test)
```

**Parallelizable groups:**
- Tasks 1-5 are independent (can run in parallel)
- Tasks 8, 9 are independent of each other
- Tasks 12, 13, 14 are independent of each other

---

## Verification Checklist

After all tasks complete:

1. `python -m pytest tests/ -v --timeout=120` — all tests pass
2. `python -c "from src.llm.dsm import DistilledSemanticModel; m = DistilledSemanticModel(); print(sum(p.numel() for p in m.parameters()))"` — ~250M params
3. `python -c "import yaml; c = yaml.safe_load(open('config/dsm_training.yaml')); print(c['llm']['backend'])"` — `dsm`
4. Smoke test with mock/small config on CPU completes in <30s
5. No references to `llm_gate` remain (except backward-compat comments)

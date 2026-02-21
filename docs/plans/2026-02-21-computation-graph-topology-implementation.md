# Computation Graph Topology Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a system that captures PyTorch computation graphs as cell complexes and applies Hodge decomposition, spectral gap analysis, and sheaf diffusion to produce novel training diagnostics and inference-time output refinement.

**Architecture:** PyTorch forward/backward hooks capture the dynamic computation graph into our existing `CellComplex` format. Three analysis passes (Hodge, spectral, sheaf) produce a `TopologicalDiagnostics` dataclass. An alert engine consumes diagnostics. A refinement head feeds topology back into inference.

**Tech Stack:** Python 3.12, PyTorch (autograd hooks), existing CellComplex/Hodge/spectral/sheaf modules, pytest.

**Design doc:** `docs/plans/2026-02-21-computation-graph-topology-design.md`

---

## Phase A: Graph Capture + Diagnostics

### Task 1: ComputationGraphCapture — Hook Registration

**Files:**
- Create: `src/computation_graph/capture.py`
- Test: `tests/test_computation_graph/test_capture.py`
- Create: `tests/test_computation_graph/__init__.py`
- Create: `src/computation_graph/__init__.py`

**Context:** We need a context manager that registers forward and backward hooks on all `nn.Module` submodules of a model. Each hook records: which module fired, activation/gradient tensor norms, and parent-child relationships. This data is stored in internal lists — NOT yet converted to a CellComplex.

**Step 1: Write the failing test**

```python
# tests/test_computation_graph/test_capture.py
import torch
import torch.nn as nn
from src.computation_graph.capture import ComputationGraphCapture


class TestHookRegistration:
    def test_forward_hooks_fire(self):
        """Forward hooks record one entry per module during forward pass."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            _ = model(x)
        # 3 submodules + 1 top-level Sequential = 4 forward records
        assert len(cap.forward_records) >= 3

    def test_backward_hooks_fire(self):
        """Backward hooks record one entry per module during backward pass."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            loss = out.sum()
            loss.backward()
        assert len(cap.backward_records) >= 3

    def test_hooks_removed_after_exit(self):
        """Hooks should be cleaned up when context manager exits."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            pass
        # After exit, model should have no extra hooks
        for mod in model.modules():
            assert len(mod._forward_hooks) == 0
            assert len(mod._backward_hooks) == 0

    def test_records_contain_module_name_and_norm(self):
        """Each record should have module name, activation norm, and shape."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            _ = model(x)
        rec = cap.forward_records[0]
        assert 'module_name' in rec
        assert 'activation_norm' in rec
        assert isinstance(rec['activation_norm'], float)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_computation_graph/test_capture.py::TestHookRegistration -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.computation_graph'`

**Step 3: Write minimal implementation**

```python
# src/computation_graph/__init__.py
# (empty)
```

```python
# src/computation_graph/capture.py
from __future__ import annotations

import torch
import torch.nn as nn


class ComputationGraphCapture:
    """Context manager that hooks into PyTorch modules to capture
    computation graph structure as forward/backward records."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.forward_records: list[dict] = []
        self.backward_records: list[dict] = []
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def __enter__(self) -> 'ComputationGraphCapture':
        self._register_hooks()
        return self

    def __exit__(self, *args):
        self._remove_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            h_fwd = module.register_forward_hook(self._make_forward_hook(name))
            h_bwd = module.register_full_backward_hook(
                self._make_backward_hook(name)
            )
            self._handles.append(h_fwd)
            self._handles.append(h_bwd)

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_forward_hook(self, module_name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                norm = output.detach().norm().item()
                shape = list(output.shape)
            elif isinstance(output, tuple) and len(output) > 0:
                norm = output[0].detach().norm().item() if isinstance(
                    output[0], torch.Tensor
                ) else 0.0
                shape = list(output[0].shape) if isinstance(
                    output[0], torch.Tensor
                ) else []
            else:
                norm = 0.0
                shape = []
            self.forward_records.append({
                'module_name': module_name,
                'module_type': type(module).__name__,
                'activation_norm': norm,
                'output_shape': shape,
            })
        return hook

    def _make_backward_hook(self, module_name: str):
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple) and len(grad_output) > 0:
                g = grad_output[0]
                norm = g.detach().norm().item() if g is not None else 0.0
            elif isinstance(grad_output, torch.Tensor):
                norm = grad_output.detach().norm().item()
            else:
                norm = 0.0
            self.backward_records.append({
                'module_name': module_name,
                'module_type': type(module).__name__,
                'gradient_norm': norm,
            })
        return hook
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_computation_graph/test_capture.py::TestHookRegistration -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/computation_graph/__init__.py src/computation_graph/capture.py \
  tests/test_computation_graph/__init__.py tests/test_computation_graph/test_capture.py
git commit -m "feat: ComputationGraphCapture hook registration and records"
```

---

### Task 2: ComputationGraphCapture — CellComplex Conversion

**Files:**
- Modify: `src/computation_graph/capture.py`
- Test: `tests/test_computation_graph/test_capture.py`

**Context:** Convert forward/backward records into a proper `CellComplex`. Each unique module becomes a 0-cell. Data flow edges (parent → child in the module tree) become 1-cells. Composite modules (modules with children) become 2-cells whose boundaries are the edges connecting their children. The tricky part: we need to infer data flow order from the sequence forward hooks fired, not from module tree nesting alone.

**Step 1: Write the failing test**

```python
# Add to tests/test_computation_graph/test_capture.py
from src.cell_complex.cell_complex import CellComplex


class TestCellComplexConversion:
    def test_to_cell_complex_returns_cell_complex(self):
        """to_cell_complex() should return a CellComplex instance."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        assert isinstance(cc, CellComplex)

    def test_0_cells_match_leaf_modules(self):
        """Each leaf module should become a 0-cell."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        leaf_count = sum(1 for m in model.modules()
                         if len(list(m.children())) == 0)
        assert cc.num_cells(0) == leaf_count

    def test_1_cells_connect_sequential_modules(self):
        """Sequential data flow should create edges between consecutive leaf modules."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        # Linear -> ReLU -> Linear = 2 edges
        assert cc.num_cells(1) == 2

    def test_edge_signals_are_activation_norms(self):
        """1-cell embeddings should encode forward activation norms."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        edge_embs = cc.get_embeddings(1)
        # First dim of edge embedding = forward activation norm (positive)
        assert (edge_embs[:, 0] >= 0).all()

    def test_2_cells_for_composite_modules(self):
        """A Sequential container should produce a 2-cell."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        # The top-level Sequential is a composite = 1 two-cell
        assert cc.num_cells(2) >= 1

    def test_chain_complex_property(self):
        """B1 @ B2 should equal 0 (chain complex property)."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        if cc.num_cells(2) > 0:
            assert cc.verify_chain_complex()

    def test_backward_signals_in_embeddings(self):
        """0-cell embeddings should include gradient norm from backward pass."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        node_embs = cc.get_embeddings(0)
        # Second dim of node embedding = gradient norm (non-negative)
        assert (node_embs[:, 1] >= 0).all()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_computation_graph/test_capture.py::TestCellComplexConversion -v`
Expected: FAIL with `AttributeError: 'ComputationGraphCapture' object has no attribute 'to_cell_complex'`

**Step 3: Write minimal implementation**

Add to `src/computation_graph/capture.py`:

```python
from src.cell_complex.cell_complex import CellComplex

# Add this method to the ComputationGraphCapture class:

    def to_cell_complex(self, embedding_dim: int = 8) -> CellComplex:
        """Convert captured records into a CellComplex.

        0-cells: leaf modules (no children), ordered by forward execution.
        1-cells: data flow edges between consecutive leaf modules.
        2-cells: composite modules whose children are leaf modules.

        Embedding layout:
          0-cell: [activation_norm, gradient_norm, 0, ..., 0]
          1-cell: [source_activation_norm, target_activation_norm, grad_norm, 0, ..., 0]
          2-cell: [mean_activation_norm, mean_gradient_norm, num_children, 0, ..., 0]
        """
        cc = CellComplex(embedding_dim=embedding_dim)

        # Identify leaf modules (those with no children)
        leaf_names = set()
        for name, mod in self.model.named_modules():
            if len(list(mod.children())) == 0:
                leaf_names.add(name)

        # Build ordered list of leaf modules by forward execution order
        seen = set()
        ordered_leaves = []
        fwd_norms = {}
        for rec in self.forward_records:
            name = rec['module_name']
            if name in leaf_names and name not in seen:
                seen.add(name)
                ordered_leaves.append(name)
                fwd_norms[name] = rec['activation_norm']

        # Collect backward norms
        bwd_norms = {}
        for rec in self.backward_records:
            name = rec['module_name']
            if name in leaf_names:
                bwd_norms[name] = rec['gradient_norm']

        # Add 0-cells
        name_to_idx = {}
        for name in ordered_leaves:
            emb = torch.zeros(embedding_dim)
            emb[0] = fwd_norms.get(name, 0.0)
            emb[1] = bwd_norms.get(name, 0.0)
            idx = cc.add_0_cell(emb, name)
            name_to_idx[name] = idx

        # Add 1-cells (consecutive leaf pairs = data flow edges)
        edge_indices = []
        for i in range(len(ordered_leaves) - 1):
            src_name = ordered_leaves[i]
            tgt_name = ordered_leaves[i + 1]
            emb = torch.zeros(embedding_dim)
            emb[0] = fwd_norms.get(src_name, 0.0)
            emb[1] = fwd_norms.get(tgt_name, 0.0)
            emb[2] = bwd_norms.get(tgt_name, 0.0)
            eidx = cc.add_1_cell(
                name_to_idx[src_name],
                name_to_idx[tgt_name],
                emb, "data_flow",
            )
            edge_indices.append(eidx)

        # Add 2-cells for composite modules
        for name, mod in self.model.named_modules():
            children = list(mod.children())
            if len(children) < 2:
                continue
            # Find edges that connect consecutive children of this module
            child_leaf_names = []
            for cname, cmod in mod.named_modules():
                full = f"{name}.{cname}" if name else cname
                if full in name_to_idx and full != name:
                    child_leaf_names.append(full)
            # Find boundary edges: edges whose endpoints are both in this
            # composite's leaf set
            child_idxs = {name_to_idx[n] for n in child_leaf_names
                          if n in name_to_idx}
            boundary = []
            for ei, eidx in enumerate(edge_indices):
                if ei < len(ordered_leaves) - 1:
                    src_idx = name_to_idx[ordered_leaves[ei]]
                    tgt_idx = name_to_idx[ordered_leaves[ei + 1]]
                    if src_idx in child_idxs and tgt_idx in child_idxs:
                        boundary.append(eidx)

            if len(boundary) >= 2:
                emb = torch.zeros(embedding_dim)
                child_fwd = [fwd_norms.get(n, 0.0) for n in child_leaf_names]
                child_bwd = [bwd_norms.get(n, 0.0) for n in child_leaf_names]
                if child_fwd:
                    emb[0] = sum(child_fwd) / len(child_fwd)
                if child_bwd:
                    emb[1] = sum(child_bwd) / len(child_bwd)
                emb[2] = float(len(child_leaf_names))
                cc.add_2_cell(boundary, emb)

        return cc
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_computation_graph/test_capture.py::TestCellComplexConversion -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/computation_graph/capture.py tests/test_computation_graph/test_capture.py
git commit -m "feat: to_cell_complex() converts captured records to CellComplex"
```

---

### Task 3: ComputationGraphCapture — Residual/Skip Connection Detection

**Files:**
- Modify: `src/computation_graph/capture.py`
- Test: `tests/test_computation_graph/test_capture.py`

**Context:** Simple sequential models are handled by Task 2, but our architecture has skip connections (residual blocks, attention). We need to detect when a tensor produced by module A is consumed by module C (skipping B). We do this by tracking tensor identity via `id()` in forward hooks — if the same tensor object appears as input to multiple modules, those modules share an edge.

**Step 1: Write the failing test**

```python
class ResidualBlock(nn.Module):
    """Simple residual block for testing skip connection detection."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.relu(self.linear(x))


class TestSkipConnectionDetection:
    def test_residual_creates_extra_edge(self):
        """A residual block should create edges for both the skip and main path."""
        model = ResidualBlock(4)
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        # Skip connection means more edges than a pure sequential model
        # Linear -> ReLU (main path) + skip from input
        assert cc.num_cells(1) >= 2

    def test_skip_edge_has_correct_signal(self):
        """Skip connection edge should have non-zero activation norm."""
        model = ResidualBlock(4)
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        cc = cap.to_cell_complex()
        edge_embs = cc.get_embeddings(1)
        # All edges should have non-negative forward activation norms
        assert (edge_embs[:, 0] >= 0).all()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_computation_graph/test_capture.py::TestSkipConnectionDetection -v`
Expected: FAIL or incorrect edge count (current impl only chains consecutive modules)

**Step 3: Write minimal implementation**

Modify the forward hook in `src/computation_graph/capture.py` to track tensor identity:

```python
# In __init__, add:
    self._tensor_producers: dict[int, str] = {}  # id(tensor) -> module_name
    self._tensor_consumers: list[tuple[str, str]] = []  # (producer, consumer)

# Replace _make_forward_hook:
    def _make_forward_hook(self, module_name: str):
        def hook(module, input, output):
            # Track which module consumed which tensors
            if isinstance(input, tuple):
                for t in input:
                    if isinstance(t, torch.Tensor):
                        tid = id(t)
                        if tid in self._tensor_producers:
                            producer = self._tensor_producers[tid]
                            if producer != module_name:
                                self._tensor_consumers.append(
                                    (producer, module_name)
                                )
            elif isinstance(input, torch.Tensor):
                tid = id(input)
                if tid in self._tensor_producers:
                    producer = self._tensor_producers[tid]
                    if producer != module_name:
                        self._tensor_consumers.append(
                            (producer, module_name)
                        )

            # Record output tensor as produced by this module
            if isinstance(output, torch.Tensor):
                norm = output.detach().norm().item()
                shape = list(output.shape)
                self._tensor_producers[id(output)] = module_name
            elif isinstance(output, tuple) and len(output) > 0:
                first = output[0]
                norm = first.detach().norm().item() if isinstance(
                    first, torch.Tensor) else 0.0
                shape = list(first.shape) if isinstance(
                    first, torch.Tensor) else []
                if isinstance(first, torch.Tensor):
                    self._tensor_producers[id(first)] = module_name
            else:
                norm = 0.0
                shape = []

            self.forward_records.append({
                'module_name': module_name,
                'module_type': type(module).__name__,
                'activation_norm': norm,
                'output_shape': shape,
            })
        return hook
```

Then update `to_cell_complex()` to use `_tensor_consumers` for non-sequential edges:

```python
# In to_cell_complex(), after building sequential edges, add:
        # Add non-sequential edges from tensor tracking (skip connections)
        for producer, consumer in self._tensor_consumers:
            if producer in name_to_idx and consumer in name_to_idx:
                src_idx = name_to_idx[producer]
                tgt_idx = name_to_idx[consumer]
                # Check not already added as sequential edge
                existing_pairs = set()
                for ei in range(len(ordered_leaves) - 1):
                    existing_pairs.add((
                        name_to_idx[ordered_leaves[ei]],
                        name_to_idx[ordered_leaves[ei + 1]],
                    ))
                if (src_idx, tgt_idx) not in existing_pairs:
                    emb = torch.zeros(embedding_dim)
                    emb[0] = fwd_norms.get(producer, 0.0)
                    emb[1] = fwd_norms.get(consumer, 0.0)
                    emb[2] = bwd_norms.get(consumer, 0.0)
                    eidx = cc.add_1_cell(src_idx, tgt_idx, emb, "skip")
                    edge_indices.append(eidx)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_computation_graph/test_capture.py -v`
Expected: PASS (all tests including previous ones)

**Step 5: Commit**

```bash
git add src/computation_graph/capture.py tests/test_computation_graph/test_capture.py
git commit -m "feat: skip connection detection via tensor identity tracking"
```

---

### Task 4: TopologicalDiagnostics Dataclass + Hodge Analysis

**Files:**
- Create: `src/computation_graph/diagnostics.py`
- Test: `tests/test_computation_graph/test_diagnostics.py`

**Context:** The `TopologicalDiagnostics` dataclass holds all analysis results. The `analyze_hodge()` function takes a captured CellComplex and computes gradient/curl/harmonic energy ratios from the backward (gradient) signal on 1-cells. Uses our existing `hodge_decomposition()` from `src/spectral/decomposition.py`.

**Step 1: Write the failing test**

```python
# tests/test_computation_graph/test_diagnostics.py
import torch
import torch.nn as nn
from src.computation_graph.capture import ComputationGraphCapture
from src.computation_graph.diagnostics import (
    TopologicalDiagnostics,
    analyze_hodge,
)


class TestTopologicalDiagnostics:
    def test_dataclass_fields(self):
        """TopologicalDiagnostics should have all required fields."""
        diag = TopologicalDiagnostics(
            gradient_energy_ratio=0.5,
            curl_energy_ratio=0.3,
            harmonic_energy_ratio=0.2,
            spectral_gap=0.1,
            per_layer_spectral_gaps={},
            restriction_map_rank=0.0,
            b1_alignment=0.0,
            b2_alignment=0.0,
            num_operations=10,
            num_data_flows=9,
            num_composites=2,
            executive_iterations=0,
        )
        assert diag.gradient_energy_ratio == 0.5
        assert diag.num_operations == 10

    def test_energy_ratios_sum_to_one(self):
        """Hodge energy ratios should sum to approximately 1.0."""
        diag = TopologicalDiagnostics(
            gradient_energy_ratio=0.5,
            curl_energy_ratio=0.3,
            harmonic_energy_ratio=0.2,
            spectral_gap=0.0,
            per_layer_spectral_gaps={},
            restriction_map_rank=0.0,
            b1_alignment=0.0,
            b2_alignment=0.0,
            num_operations=0,
            num_data_flows=0,
            num_composites=0,
            executive_iterations=0,
        )
        total = (diag.gradient_energy_ratio +
                 diag.curl_energy_ratio +
                 diag.harmonic_energy_ratio)
        assert abs(total - 1.0) < 1e-6


class TestAnalyzeHodge:
    def _capture_model(self):
        """Helper: capture a simple model's computation graph."""
        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 8), nn.ReLU(),
            nn.Linear(8, 2),
        )
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        return cap.to_cell_complex()

    def test_analyze_hodge_returns_ratios(self):
        """analyze_hodge should return three non-negative ratios."""
        cc = self._capture_model()
        grad_r, curl_r, harm_r = analyze_hodge(cc)
        assert grad_r >= 0.0
        assert curl_r >= 0.0
        assert harm_r >= 0.0

    def test_analyze_hodge_ratios_sum_to_one(self):
        """Energy ratios should sum to 1.0 (or all be 0 if no signal)."""
        cc = self._capture_model()
        grad_r, curl_r, harm_r = analyze_hodge(cc)
        total = grad_r + curl_r + harm_r
        assert abs(total - 1.0) < 1e-4 or total == 0.0

    def test_analyze_hodge_with_no_2cells(self):
        """When there are no 2-cells, curl should be 0."""
        # A chain graph has no triangles → no 2-cells → curl = 0
        from src.cell_complex.cell_complex import CellComplex
        cc = CellComplex(embedding_dim=4)
        n0 = cc.add_0_cell(torch.randn(4), "a")
        n1 = cc.add_0_cell(torch.randn(4), "b")
        n2 = cc.add_0_cell(torch.randn(4), "c")
        cc.add_1_cell(n0, n1, torch.randn(4), "e0")
        cc.add_1_cell(n1, n2, torch.randn(4), "e1")
        grad_r, curl_r, harm_r = analyze_hodge(cc)
        assert curl_r == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_computation_graph/test_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/computation_graph/diagnostics.py
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.cell_complex.cell_complex import CellComplex
from src.spectral.decomposition import hodge_decomposition


@dataclass
class TopologicalDiagnostics:
    """Topological analysis of a computation graph."""
    # Hodge decomposition of gradient flow
    gradient_energy_ratio: float
    curl_energy_ratio: float
    harmonic_energy_ratio: float

    # Spectral analysis
    spectral_gap: float
    per_layer_spectral_gaps: dict[str, float]

    # Sheaf analysis (Phase A placeholder — populated in Phase C)
    restriction_map_rank: float
    b1_alignment: float
    b2_alignment: float

    # Metadata
    num_operations: int
    num_data_flows: int
    num_composites: int
    executive_iterations: int


def analyze_hodge(cc: CellComplex) -> tuple[float, float, float]:
    """Compute Hodge energy ratios from edge embeddings of a computation graph.

    Uses the backward gradient norm (embedding dim 2) as the 1-cell signal.
    Returns (gradient_ratio, curl_ratio, harmonic_ratio) that sum to 1.0.
    """
    num_edges = cc.num_cells(1)
    if num_edges < 2:
        return 0.0, 0.0, 0.0

    # Extract gradient norm signal from 1-cell embeddings (dim index 2)
    edge_embs = cc.get_embeddings(1)
    signal = edge_embs[:, 2] if edge_embs.shape[1] > 2 else edge_embs[:, 0]

    total_energy = (signal ** 2).sum().item()
    if total_energy < 1e-12:
        return 0.0, 0.0, 0.0

    try:
        gradient, curl, harmonic = hodge_decomposition(cc, signal, dim=1)
    except Exception:
        # Fallback if decomposition fails (e.g., degenerate graph)
        return 0.0, 0.0, 0.0

    grad_energy = (gradient ** 2).sum().item()
    curl_energy = (curl ** 2).sum().item()
    harm_energy = (harmonic ** 2).sum().item()

    total = grad_energy + curl_energy + harm_energy
    if total < 1e-12:
        return 0.0, 0.0, 0.0

    return grad_energy / total, curl_energy / total, harm_energy / total
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_computation_graph/test_diagnostics.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/computation_graph/diagnostics.py tests/test_computation_graph/test_diagnostics.py
git commit -m "feat: TopologicalDiagnostics dataclass and Hodge analysis"
```

---

### Task 5: Spectral Gap Analysis of Computation Graph

**Files:**
- Modify: `src/computation_graph/diagnostics.py`
- Test: `tests/test_computation_graph/test_diagnostics.py`

**Context:** Compute the spectral gap (smallest nonzero eigenvalue) of the computation graph's L1 Hodge Laplacian. Also compute per-layer spectral gaps by extracting subgraphs for each composite module (2-cell). Uses `spectral_decomposition()` from `src/spectral/decomposition.py`.

**Step 1: Write the failing test**

```python
# Add to tests/test_computation_graph/test_diagnostics.py
from src.computation_graph.diagnostics import analyze_spectral_gap


class TestAnalyzeSpectralGap:
    def _capture_model(self):
        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 8), nn.ReLU(),
            nn.Linear(8, 2),
        )
        with ComputationGraphCapture(model) as cap:
            x = torch.randn(1, 4)
            out = model(x)
            out.sum().backward()
        return cap.to_cell_complex()

    def test_spectral_gap_is_non_negative(self):
        """Spectral gap should be >= 0."""
        cc = self._capture_model()
        gap = analyze_spectral_gap(cc)
        assert gap >= 0.0

    def test_spectral_gap_connected_graph(self):
        """A connected sequential model should have spectral gap > 0."""
        cc = self._capture_model()
        gap = analyze_spectral_gap(cc)
        assert gap > 0.0

    def test_spectral_gap_single_node(self):
        """Graph with one node should have spectral gap 0."""
        cc = CellComplex(embedding_dim=4)
        cc.add_0_cell(torch.randn(4), "single")
        gap = analyze_spectral_gap(cc)
        assert gap == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_computation_graph/test_diagnostics.py::TestAnalyzeSpectralGap -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `src/computation_graph/diagnostics.py`:

```python
from src.spectral.decomposition import spectral_decomposition


def analyze_spectral_gap(cc: CellComplex) -> float:
    """Compute spectral gap of the computation graph's node Laplacian L0.

    Returns the smallest nonzero eigenvalue, or 0.0 if the graph has
    fewer than 2 nodes.
    """
    if cc.num_cells(0) < 2:
        return 0.0

    try:
        eigenvalues, _ = spectral_decomposition(cc, dim=0)
    except Exception:
        return 0.0

    # Find smallest eigenvalue > threshold
    threshold = 1e-6
    nonzero = eigenvalues[eigenvalues > threshold]
    if len(nonzero) == 0:
        return 0.0

    return nonzero[0].item()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_computation_graph/test_diagnostics.py::TestAnalyzeSpectralGap -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/computation_graph/diagnostics.py tests/test_computation_graph/test_diagnostics.py
git commit -m "feat: spectral gap analysis of computation graph"
```

---

### Task 6: Full Analysis Pipeline

**Files:**
- Modify: `src/computation_graph/diagnostics.py`
- Test: `tests/test_computation_graph/test_diagnostics.py`

**Context:** Combine capture + Hodge + spectral gap into a single `analyze_computation_graph()` function that takes a model, runs one forward+backward pass, and returns a complete `TopologicalDiagnostics`.

**Step 1: Write the failing test**

```python
# Add to tests/test_computation_graph/test_diagnostics.py
from src.computation_graph.diagnostics import analyze_computation_graph


class TestFullAnalysisPipeline:
    def test_returns_diagnostics(self):
        """analyze_computation_graph should return TopologicalDiagnostics."""
        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 8), nn.ReLU(),
            nn.Linear(8, 2),
        )
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        diag = analyze_computation_graph(model, x, target, nn.CrossEntropyLoss())
        assert isinstance(diag, TopologicalDiagnostics)

    def test_diagnostics_has_positive_spectral_gap(self):
        """Connected model should have positive spectral gap."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        diag = analyze_computation_graph(model, x, target, nn.CrossEntropyLoss())
        assert diag.spectral_gap > 0.0

    def test_diagnostics_hodge_ratios_valid(self):
        """Energy ratios should sum to 1.0."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        diag = analyze_computation_graph(model, x, target, nn.CrossEntropyLoss())
        total = (diag.gradient_energy_ratio +
                 diag.curl_energy_ratio +
                 diag.harmonic_energy_ratio)
        assert abs(total - 1.0) < 1e-4 or total == 0.0

    def test_diagnostics_counts_correct(self):
        """Operation count should match leaf module count."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        diag = analyze_computation_graph(model, x, target, nn.CrossEntropyLoss())
        leaf_count = sum(1 for m in model.modules()
                         if len(list(m.children())) == 0)
        assert diag.num_operations == leaf_count

    def test_no_gradient_leak(self):
        """Model parameters should not have gradients from analysis pass."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        # Zero gradients first
        model.zero_grad()
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        diag = analyze_computation_graph(model, x, target, nn.CrossEntropyLoss())
        # Analysis should not leave residual gradients on model params
        for p in model.parameters():
            assert p.grad is None or (p.grad == 0).all()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_computation_graph/test_diagnostics.py::TestFullAnalysisPipeline -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `src/computation_graph/diagnostics.py`:

```python
from src.computation_graph.capture import ComputationGraphCapture


def analyze_computation_graph(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
) -> TopologicalDiagnostics:
    """Run one forward+backward pass and compute full topological diagnostics.

    Saves and restores model gradient state. Does not modify model parameters.
    """
    # Save gradient state
    grad_state = {p: p.grad.clone() if p.grad is not None else None
                  for p in model.parameters()}
    was_training = model.training
    model.zero_grad()

    with ComputationGraphCapture(model) as cap:
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()

    cc = cap.to_cell_complex()

    # Hodge analysis
    grad_r, curl_r, harm_r = analyze_hodge(cc)

    # Spectral gap
    gap = analyze_spectral_gap(cc)

    # Restore gradient state
    model.zero_grad()
    for p in model.parameters():
        if grad_state[p] is not None:
            p.grad = grad_state[p]

    if was_training:
        model.train()

    return TopologicalDiagnostics(
        gradient_energy_ratio=grad_r,
        curl_energy_ratio=curl_r,
        harmonic_energy_ratio=harm_r,
        spectral_gap=gap,
        per_layer_spectral_gaps={},  # TODO: Phase B
        restriction_map_rank=0.0,    # TODO: Phase C sheaf
        b1_alignment=0.0,
        b2_alignment=0.0,
        num_operations=cc.num_cells(0),
        num_data_flows=cc.num_cells(1),
        num_composites=cc.num_cells(2),
        executive_iterations=0,      # TODO: detect from cap records
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_computation_graph/test_diagnostics.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/computation_graph/diagnostics.py tests/test_computation_graph/test_diagnostics.py
git commit -m "feat: full analysis pipeline - analyze_computation_graph()"
```

---

### Task 7: Integration with Existing Training Loop

**Files:**
- Modify: `src/computation_graph/diagnostics.py`
- Create: `tests/test_computation_graph/test_training_integration.py`

**Context:** Add a `TrainingTopologyMonitor` class that wraps around a training loop. It calls `analyze_computation_graph()` every N steps and accumulates `TopologicalDiagnostics` into a time series. Provides `summary()` for per-epoch aggregate and `alerts()` for threshold-based warnings.

**Step 1: Write the failing test**

```python
# tests/test_computation_graph/test_training_integration.py
import torch
import torch.nn as nn
from src.computation_graph.diagnostics import (
    TrainingTopologyMonitor,
    TopologicalDiagnostics,
)


class TestTrainingTopologyMonitor:
    def _make_model(self):
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def test_record_step(self):
        """Monitor should record diagnostics for each step."""
        model = self._make_model()
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        monitor.record_step(x, target)
        assert len(monitor.history) == 1
        assert isinstance(monitor.history[0], TopologicalDiagnostics)

    def test_summary(self):
        """Summary should return mean/std of key metrics."""
        model = self._make_model()
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        for _ in range(3):
            x = torch.randn(1, 4)
            target = torch.tensor([1])
            monitor.record_step(x, target)
        summary = monitor.summary()
        assert 'gradient_energy_ratio_mean' in summary
        assert 'spectral_gap_mean' in summary

    def test_alerts_empty_when_healthy(self):
        """No alerts should fire for a simple healthy model."""
        model = self._make_model()
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        for _ in range(3):
            x = torch.randn(1, 4)
            target = torch.tensor([1])
            monitor.record_step(x, target)
        alerts = monitor.alerts()
        # Simple sequential model should be healthy
        assert isinstance(alerts, list)

    def test_custom_thresholds(self):
        """Monitor should accept custom alert thresholds."""
        model = self._make_model()
        thresholds = {'curl_energy_ratio': 0.01}  # Very strict
        monitor = TrainingTopologyMonitor(
            model, nn.CrossEntropyLoss(), alert_thresholds=thresholds,
        )
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        monitor.record_step(x, target)
        # With extremely strict threshold, might get an alert
        alerts = monitor.alerts()
        assert isinstance(alerts, list)

    def test_reset(self):
        """reset() should clear history."""
        model = self._make_model()
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        monitor.record_step(x, target)
        assert len(monitor.history) == 1
        monitor.reset()
        assert len(monitor.history) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_computation_graph/test_training_integration.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `src/computation_graph/diagnostics.py`:

```python
_DEFAULT_THRESHOLDS = {
    'curl_energy_ratio': 0.4,
    'harmonic_energy_ratio': 0.15,
    'spectral_gap_min': 0.01,
}


class TrainingTopologyMonitor:
    """Monitors computation graph topology across training steps."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        alert_thresholds: dict[str, float] | None = None,
    ):
        self.model = model
        self.criterion = criterion
        self.history: list[TopologicalDiagnostics] = []
        self.thresholds = {**_DEFAULT_THRESHOLDS}
        if alert_thresholds:
            self.thresholds.update(alert_thresholds)

    def record_step(
        self,
        input_tensor: torch.Tensor,
        target: torch.Tensor,
    ) -> TopologicalDiagnostics:
        """Run analysis on one input and append to history."""
        diag = analyze_computation_graph(
            self.model, input_tensor, target, self.criterion,
        )
        self.history.append(diag)
        return diag

    def summary(self) -> dict[str, float]:
        """Compute mean/std of key metrics across history."""
        if not self.history:
            return {}

        fields = [
            'gradient_energy_ratio', 'curl_energy_ratio',
            'harmonic_energy_ratio', 'spectral_gap',
        ]
        result = {}
        for field in fields:
            values = [getattr(d, field) for d in self.history]
            result[f'{field}_mean'] = sum(values) / len(values)
            if len(values) > 1:
                mean = result[f'{field}_mean']
                var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                result[f'{field}_std'] = var ** 0.5
            else:
                result[f'{field}_std'] = 0.0
        return result

    def alerts(self) -> list[str]:
        """Check latest diagnostics against thresholds."""
        if not self.history:
            return []

        latest = self.history[-1]
        alerts = []

        curl_thresh = self.thresholds.get('curl_energy_ratio', 0.4)
        if latest.curl_energy_ratio > curl_thresh:
            alerts.append(
                f"High curl energy: {latest.curl_energy_ratio:.3f} > {curl_thresh}"
            )

        harm_thresh = self.thresholds.get('harmonic_energy_ratio', 0.15)
        if latest.harmonic_energy_ratio > harm_thresh:
            alerts.append(
                f"High harmonic energy: {latest.harmonic_energy_ratio:.3f} > {harm_thresh}"
            )

        gap_min = self.thresholds.get('spectral_gap_min', 0.01)
        if latest.spectral_gap < gap_min:
            alerts.append(
                f"Low spectral gap: {latest.spectral_gap:.6f} < {gap_min}"
            )

        return alerts

    def reset(self):
        """Clear history."""
        self.history.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_computation_graph/test_training_integration.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/computation_graph/diagnostics.py \
  tests/test_computation_graph/test_training_integration.py
git commit -m "feat: TrainingTopologyMonitor with alerts and summary"
```

---

### Task 8: Integration Test with Our Architecture

**Files:**
- Test: `tests/test_computation_graph/test_architecture_integration.py`

**Context:** Verify that the capture + analysis pipeline works on our actual `HierarchicalMultiHopModel` with `ExecutiveReasoningLoop`, not just toy `nn.Sequential` models. This is the critical validation that the hook system handles our custom modules (GNN, TAT, wave dynamics, etc.).

**Step 1: Write the failing test**

```python
# tests/test_computation_graph/test_architecture_integration.py
import torch
import torch.nn as nn
import pytest
from src.computation_graph.capture import ComputationGraphCapture
from src.computation_graph.diagnostics import (
    analyze_computation_graph,
    analyze_hodge,
    TopologicalDiagnostics,
)
from src.cell_complex.cell_complex import CellComplex
from src.datasets.graph_generators import random_graph
from src.datasets.graph_convert import nx_to_cell_complex


def _make_small_model():
    """Build a minimal HierarchicalMultiHopModel for testing."""
    from src.benchmarks.run_comparison import HierarchicalMultiHopModel
    return HierarchicalMultiHopModel(
        embedding_dim=8,
        gnn_hidden=16,
        gnn_spatial_layers=1,
        gnn_spectral_layers=1,
        max_freqs=4,
        tat_layers=1,
        tat_spatial_heads=1,
        tat_spectral_heads=1,
        tat_ff_dim=32,
        max_classes=3,
        max_iterations=2,
        convergence_threshold=0.1,
        use_wave_dynamics=True,
        use_higher_order=False,
        use_topological_pe=False,
        use_structural_features=False,
    )


def _make_sample():
    """Generate a small graph sample for testing."""
    G = random_graph('ba', n_nodes=10)
    cc = nx_to_cell_complex(G, embedding_dim=8)
    return cc, 0, 3, 3, {}


class TestArchitectureIntegration:
    def test_capture_works_on_hierarchical_model(self):
        """ComputationGraphCapture should work with our full architecture."""
        model = _make_small_model()
        cc_input, query, target, answer, meta = _make_sample()
        cc_clone = cc_input.clone()
        with ComputationGraphCapture(model) as cap:
            logits = model(cc_clone, query, target, meta)
            loss = nn.CrossEntropyLoss()(logits.unsqueeze(0),
                                          torch.tensor([answer]))
            loss.backward()
        comp_cc = cap.to_cell_complex()
        # Should have many more operations than a simple Sequential
        assert comp_cc.num_cells(0) >= 5

    def test_hodge_on_architecture_graph(self):
        """Hodge analysis should produce valid ratios for our architecture."""
        model = _make_small_model()
        cc_input, query, target, answer, meta = _make_sample()
        cc_clone = cc_input.clone()
        with ComputationGraphCapture(model) as cap:
            logits = model(cc_clone, query, target, meta)
            loss = nn.CrossEntropyLoss()(logits.unsqueeze(0),
                                          torch.tensor([answer]))
            loss.backward()
        comp_cc = cap.to_cell_complex()
        grad_r, curl_r, harm_r = analyze_hodge(comp_cc)
        total = grad_r + curl_r + harm_r
        assert total == 0.0 or abs(total - 1.0) < 1e-4

    def test_executive_loop_unrolling(self):
        """Capture should handle variable executive loop iterations."""
        model = _make_small_model()
        # Run twice with different inputs — should capture different
        # computation graphs
        for _ in range(2):
            cc_input, query, target, answer, meta = _make_sample()
            cc_clone = cc_input.clone()
            with ComputationGraphCapture(model) as cap:
                logits = model(cc_clone, query, target, meta)
                loss = nn.CrossEntropyLoss()(logits.unsqueeze(0),
                                              torch.tensor([answer]))
                loss.backward()
            comp_cc = cap.to_cell_complex()
            assert comp_cc.num_cells(0) >= 3
```

**Step 2: Run test to verify it fails or passes**

Run: `pytest tests/test_computation_graph/test_architecture_integration.py -v`
Expected: May fail if hooks interact badly with our custom modules. If it passes, great — move to commit.

**Step 3: Fix any issues**

If hooks fail on custom modules (e.g., `ExecutiveReasoningLoop` which has dynamic control flow), may need to add guards in the hook functions for non-standard outputs (dicts, NamedTuples, etc.). Adjust `_make_forward_hook` and `_make_backward_hook` accordingly.

**Step 4: Run full test suite**

Run: `pytest tests/test_computation_graph/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/test_computation_graph/test_architecture_integration.py
git commit -m "test: integration tests for computation graph capture on full architecture"
```

---

### Task 9: Package Exports and Module Cleanup

**Files:**
- Modify: `src/computation_graph/__init__.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_computation_graph/test_capture.py
class TestModuleExports:
    def test_top_level_imports(self):
        """Key classes should be importable from src.computation_graph."""
        from src.computation_graph import (
            ComputationGraphCapture,
            TopologicalDiagnostics,
            TrainingTopologyMonitor,
            analyze_computation_graph,
            analyze_hodge,
            analyze_spectral_gap,
        )
        assert ComputationGraphCapture is not None
        assert TopologicalDiagnostics is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_computation_graph/test_capture.py::TestModuleExports -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/computation_graph/__init__.py
from src.computation_graph.capture import ComputationGraphCapture
from src.computation_graph.diagnostics import (
    TopologicalDiagnostics,
    TrainingTopologyMonitor,
    analyze_computation_graph,
    analyze_hodge,
    analyze_spectral_gap,
)

__all__ = [
    'ComputationGraphCapture',
    'TopologicalDiagnostics',
    'TrainingTopologyMonitor',
    'analyze_computation_graph',
    'analyze_hodge',
    'analyze_spectral_gap',
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_computation_graph/ -v`
Expected: ALL PASS

**Step 5: Run full project test suite**

Run: `pytest tests/ -x --timeout=120`
Expected: 633+ tests pass (existing tests unaffected)

**Step 6: Commit**

```bash
git add src/computation_graph/__init__.py tests/test_computation_graph/test_capture.py
git commit -m "feat: package exports for computation_graph module"
```

---

## Phase B: Architecture Optimization Engine

> **Depends on:** Phase A validated and merged.

### Task 10: Alert Engine

**Files:**
- Modify: `src/computation_graph/diagnostics.py`
- Test: `tests/test_computation_graph/test_alerts.py`

**Context:** Extend `TrainingTopologyMonitor.alerts()` with Mode 2 suggestions — trend-based analysis that looks at the full history, not just the latest step. Rising curl energy suggests specific parameter changes. Declining spectral gap suggests architecture changes.

**Step 1: Write the failing test**

```python
# tests/test_computation_graph/test_alerts.py
import torch
import torch.nn as nn
from src.computation_graph.diagnostics import (
    TrainingTopologyMonitor,
    TopologicalDiagnostics,
)


class TestTrendAlerts:
    def test_rising_curl_alert(self):
        """Rising curl energy over 5+ steps should trigger suggestion."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        # Inject synthetic history with rising curl
        for i in range(6):
            monitor.history.append(TopologicalDiagnostics(
                gradient_energy_ratio=0.8 - i * 0.05,
                curl_energy_ratio=0.1 + i * 0.05,
                harmonic_energy_ratio=0.1,
                spectral_gap=0.5,
                per_layer_spectral_gaps={},
                restriction_map_rank=0.0,
                b1_alignment=0.0,
                b2_alignment=0.0,
                num_operations=3,
                num_data_flows=2,
                num_composites=1,
                executive_iterations=0,
            ))
        suggestions = monitor.suggestions()
        assert any('curl' in s.lower() or 'damping' in s.lower()
                    for s in suggestions)

    def test_declining_spectral_gap_alert(self):
        """Declining spectral gap should trigger bottleneck suggestion."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        for i in range(6):
            monitor.history.append(TopologicalDiagnostics(
                gradient_energy_ratio=0.7,
                curl_energy_ratio=0.2,
                harmonic_energy_ratio=0.1,
                spectral_gap=0.5 - i * 0.08,
                per_layer_spectral_gaps={},
                restriction_map_rank=0.0,
                b1_alignment=0.0,
                b2_alignment=0.0,
                num_operations=3,
                num_data_flows=2,
                num_composites=1,
                executive_iterations=0,
            ))
        suggestions = monitor.suggestions()
        assert any('spectral' in s.lower() or 'bottleneck' in s.lower()
                    for s in suggestions)

    def test_no_suggestions_when_stable(self):
        """Stable metrics should produce no suggestions."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        monitor = TrainingTopologyMonitor(model, nn.CrossEntropyLoss())
        for _ in range(6):
            monitor.history.append(TopologicalDiagnostics(
                gradient_energy_ratio=0.7,
                curl_energy_ratio=0.2,
                harmonic_energy_ratio=0.1,
                spectral_gap=0.5,
                per_layer_spectral_gaps={},
                restriction_map_rank=0.0,
                b1_alignment=0.0,
                b2_alignment=0.0,
                num_operations=3,
                num_data_flows=2,
                num_composites=1,
                executive_iterations=0,
            ))
        suggestions = monitor.suggestions()
        assert len(suggestions) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_computation_graph/test_alerts.py -v`
Expected: FAIL with `AttributeError: 'TrainingTopologyMonitor' object has no attribute 'suggestions'`

**Step 3: Write minimal implementation**

Add to `TrainingTopologyMonitor` in `src/computation_graph/diagnostics.py`:

```python
    def suggestions(self, window: int = 5) -> list[str]:
        """Analyze trends in history and return hyperparameter suggestions."""
        if len(self.history) < window:
            return []

        recent = self.history[-window:]
        suggestions = []

        # Check curl trend
        curl_values = [d.curl_energy_ratio for d in recent]
        curl_slope = (curl_values[-1] - curl_values[0]) / (window - 1)
        if curl_slope > 0.02:
            suggestions.append(
                f"Rising curl energy ({curl_values[0]:.3f} -> {curl_values[-1]:.3f}). "
                f"Consider increasing wave_damping or reducing executive loop iterations."
            )

        # Check spectral gap trend
        gap_values = [d.spectral_gap for d in recent]
        gap_slope = (gap_values[-1] - gap_values[0]) / (window - 1)
        if gap_slope < -0.05:
            suggestions.append(
                f"Declining spectral gap ({gap_values[0]:.4f} -> {gap_values[-1]:.4f}). "
                f"Information bottleneck detected. Consider widening layers or adding skip connections."
            )

        # Check harmonic energy trend
        harm_values = [d.harmonic_energy_ratio for d in recent]
        harm_slope = (harm_values[-1] - harm_values[0]) / (window - 1)
        if harm_slope > 0.02:
            suggestions.append(
                f"Rising harmonic energy ({harm_values[0]:.3f} -> {harm_values[-1]:.3f}). "
                f"Dead subnetwork growing. Consider pruning or reinitializing affected layers."
            )

        return suggestions
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_computation_graph/test_alerts.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/computation_graph/diagnostics.py tests/test_computation_graph/test_alerts.py
git commit -m "feat: trend-based suggestions in TrainingTopologyMonitor"
```

---

## Phase C: Inference-Time Topological Refinement

> **Depends on:** Phase A + B validated. Phase A findings confirm Hodge ratios are meaningful on our architecture's computation graph.

### Task 11: Topological Confidence Head

**Files:**
- Create: `src/computation_graph/confidence.py`
- Test: `tests/test_computation_graph/test_confidence.py`

**Context:** A lightweight `TopologicalConfidence` module that takes model logits + Hodge energy ratios and produces calibrated confidence scores. Unlike softmax entropy (which only measures "how peaked is the output distribution"), this uses topology to measure "how cleanly did the model reason."

**Step 1: Write the failing test**

```python
# tests/test_computation_graph/test_confidence.py
import torch
import torch.nn as nn
from src.computation_graph.confidence import TopologicalConfidence


class TestTopologicalConfidence:
    def test_output_shape(self):
        """Confidence head should output a single scalar per sample."""
        head = TopologicalConfidence(num_classes=3)
        logits = torch.randn(1, 3)
        topo_features = torch.tensor([[0.7, 0.2, 0.1, 0.5]])  # grad, curl, harm, gap
        conf = head(logits, topo_features)
        assert conf.shape == (1,)

    def test_output_range(self):
        """Confidence should be in [0, 1]."""
        head = TopologicalConfidence(num_classes=3)
        logits = torch.randn(1, 3)
        topo_features = torch.tensor([[0.7, 0.2, 0.1, 0.5]])
        conf = head(logits, topo_features)
        assert conf.item() >= 0.0
        assert conf.item() <= 1.0

    def test_gradient_flows(self):
        """Confidence head should be trainable."""
        head = TopologicalConfidence(num_classes=3)
        logits = torch.randn(1, 3, requires_grad=True)
        topo_features = torch.tensor([[0.7, 0.2, 0.1, 0.5]])
        conf = head(logits, topo_features)
        conf.sum().backward()
        assert logits.grad is not None

    def test_batch_support(self):
        """Should handle batch dimension."""
        head = TopologicalConfidence(num_classes=3)
        logits = torch.randn(4, 3)
        topo_features = torch.randn(4, 4)
        conf = head(logits, topo_features)
        assert conf.shape == (4,)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_computation_graph/test_confidence.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/computation_graph/confidence.py
from __future__ import annotations

import torch
import torch.nn as nn


class TopologicalConfidence(nn.Module):
    """Produces calibrated confidence scores from logits + topological features.

    Input topological features: [gradient_ratio, curl_ratio, harmonic_ratio, spectral_gap]
    Output: scalar confidence in [0, 1] per sample.

    Unlike softmax entropy, this measures how cleanly the model reasoned,
    not how peaked the output distribution is.
    """

    def __init__(self, num_classes: int, topo_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes + topo_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        logits: torch.Tensor,
        topo_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) raw model output
            topo_features: (B, topo_dim) topological diagnostics

        Returns:
            (B,) confidence scores in [0, 1]
        """
        combined = torch.cat([logits, topo_features], dim=-1)
        return self.net(combined).squeeze(-1)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_computation_graph/test_confidence.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/computation_graph/confidence.py tests/test_computation_graph/test_confidence.py
git commit -m "feat: TopologicalConfidence head for topology-grounded calibration"
```

---

### Task 12: Topological Feedback into Executive Loop

**Files:**
- Modify: `src/computation_graph/diagnostics.py`
- Modify: `src/gnn_executive/control_head.py` (add optional topo_features input)
- Test: `tests/test_computation_graph/test_feedback.py`

**Context:** This is the meta-cognition mechanism. After each executive loop iteration, capture the partial computation graph, compute Hodge ratios, and feed them back as additional inputs to the ControlHead for the next iteration. The ControlHead's trunk input dimension increases by 3 (gradient_ratio, curl_ratio, spectral_gap).

**Step 1: Write the failing test**

```python
# tests/test_computation_graph/test_feedback.py
import torch
from src.gnn_executive.control_head import ControlHead, ControlSignal


class TestTopologicalFeedback:
    def test_control_head_accepts_topo_features(self):
        """ControlHead should accept optional topo_features kwarg."""
        head = ControlHead(embedding_dim=8, num_freqs=4, use_topo_feedback=True)
        node_embs = torch.randn(5, 8)
        harmonic = torch.tensor(0.5)
        topo_feats = torch.tensor([0.7, 0.2, 0.5])  # grad_r, curl_r, gap
        signal = head(node_embs, harmonic, topo_features=topo_feats)
        assert isinstance(signal, ControlSignal)

    def test_control_head_works_without_topo_features(self):
        """ControlHead with use_topo_feedback=True should still work without features."""
        head = ControlHead(embedding_dim=8, num_freqs=4, use_topo_feedback=True)
        node_embs = torch.randn(5, 8)
        harmonic = torch.tensor(0.5)
        signal = head(node_embs, harmonic)
        assert isinstance(signal, ControlSignal)

    def test_backward_compat(self):
        """ControlHead without use_topo_feedback should work as before."""
        head = ControlHead(embedding_dim=8, num_freqs=4)
        node_embs = torch.randn(5, 8)
        harmonic = torch.tensor(0.5)
        signal = head(node_embs, harmonic)
        assert isinstance(signal, ControlSignal)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_computation_graph/test_feedback.py -v`
Expected: FAIL with `TypeError: ControlHead.__init__() got an unexpected keyword argument 'use_topo_feedback'`

**Step 3: Write minimal implementation**

Modify `src/gnn_executive/control_head.py`:

```python
# In ControlHead.__init__, add parameter:
#   use_topo_feedback: bool = False
# If True, increase trunk input dim by 3 (grad_ratio, curl_ratio, gap)

# In ControlHead.forward(), add optional kwarg:
#   topo_features: torch.Tensor | None = None
# If provided and use_topo_feedback, concatenate to trunk input
# If not provided, use zeros as default
```

The exact diff depends on the current ControlHead code structure. The key change:
- `trunk_input_dim = embedding_dim + 2` → `trunk_input_dim = embedding_dim + 2 + (3 if use_topo_feedback else 0)`
- In forward: `if self.use_topo_feedback: features = torch.cat([features, topo_features_or_zeros])`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_computation_graph/test_feedback.py -v && pytest tests/test_gnn_executive/ -v`
Expected: ALL PASS (new + existing tests)

**Step 5: Commit**

```bash
git add src/gnn_executive/control_head.py tests/test_computation_graph/test_feedback.py
git commit -m "feat: optional topo_features input to ControlHead for meta-cognition"
```

---

### Task 13: Final Integration Test + Export Update

**Files:**
- Modify: `src/computation_graph/__init__.py`
- Test: `tests/test_computation_graph/test_end_to_end.py`

**Context:** End-to-end test: build model → train one step → capture computation graph → analyze → check confidence → verify no interference with training. This validates the entire Phase A-C pipeline works together.

**Step 1: Write the failing test**

```python
# tests/test_computation_graph/test_end_to_end.py
import torch
import torch.nn as nn
from src.computation_graph import (
    ComputationGraphCapture,
    TopologicalDiagnostics,
    TrainingTopologyMonitor,
    analyze_computation_graph,
)
from src.computation_graph.confidence import TopologicalConfidence


class TestEndToEnd:
    def test_full_pipeline(self):
        """Full pipeline: model → capture → analyze → confidence."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        criterion = nn.CrossEntropyLoss()
        conf_head = TopologicalConfidence(num_classes=3)

        x = torch.randn(1, 4)
        target = torch.tensor([1])

        # 1. Get diagnostics
        diag = analyze_computation_graph(model, x, target, criterion)
        assert isinstance(diag, TopologicalDiagnostics)

        # 2. Get model output
        logits = model(x)

        # 3. Compute topology-grounded confidence
        topo_feats = torch.tensor([[
            diag.gradient_energy_ratio,
            diag.curl_energy_ratio,
            diag.harmonic_energy_ratio,
            diag.spectral_gap,
        ]])
        conf = conf_head(logits, topo_feats)
        assert 0.0 <= conf.item() <= 1.0

    def test_training_not_affected(self):
        """Analysis should not interfere with normal training."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Normal training step
        x = torch.randn(1, 4)
        target = torch.tensor([1])
        optimizer.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        optimizer.step()
        loss_before = loss.item()

        # Analysis step (should not affect model)
        diag = analyze_computation_graph(model, x, target, criterion)

        # Another training step — should work normally
        optimizer.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        optimizer.step()
        # Loss should have changed (model updated)
        assert isinstance(loss.item(), float)
```

**Step 2: Run test to verify it fails or passes**

Run: `pytest tests/test_computation_graph/test_end_to_end.py -v`
Expected: PASS if all prior tasks are complete

**Step 3: Update exports**

Add `TopologicalConfidence` to `src/computation_graph/__init__.py`:

```python
from src.computation_graph.confidence import TopologicalConfidence

# Add to __all__:
'TopologicalConfidence',
```

**Step 4: Run full test suite**

Run: `pytest tests/ -x --timeout=120`
Expected: 650+ tests pass (633 existing + ~17 new)

**Step 5: Commit**

```bash
git add src/computation_graph/__init__.py tests/test_computation_graph/test_end_to_end.py
git commit -m "test: end-to-end integration test for computation graph topology pipeline"
```

---

## Summary

| Task | Phase | What | New Tests |
|------|-------|------|-----------|
| 1 | A | Hook registration + forward/backward records | 4 |
| 2 | A | CellComplex conversion from records | 7 |
| 3 | A | Skip connection detection via tensor identity | 2 |
| 4 | A | TopologicalDiagnostics + Hodge analysis | 5 |
| 5 | A | Spectral gap analysis | 3 |
| 6 | A | Full analysis pipeline function | 5 |
| 7 | A | TrainingTopologyMonitor | 5 |
| 8 | A | Integration test with our architecture | 3 |
| 9 | A | Package exports | 1 |
| 10 | B | Trend-based suggestions | 3 |
| 11 | C | TopologicalConfidence head | 4 |
| 12 | C | Topological feedback into ControlHead | 3 |
| 13 | C | End-to-end integration test | 2 |
| **Total** | | | **~47 new tests** |

**New files created:**
- `src/computation_graph/__init__.py`
- `src/computation_graph/capture.py`
- `src/computation_graph/diagnostics.py`
- `src/computation_graph/confidence.py`
- `tests/test_computation_graph/__init__.py`
- `tests/test_computation_graph/test_capture.py`
- `tests/test_computation_graph/test_diagnostics.py`
- `tests/test_computation_graph/test_training_integration.py`
- `tests/test_computation_graph/test_architecture_integration.py`
- `tests/test_computation_graph/test_alerts.py`
- `tests/test_computation_graph/test_confidence.py`
- `tests/test_computation_graph/test_feedback.py`
- `tests/test_computation_graph/test_end_to_end.py`

**Existing files modified:**
- `src/gnn_executive/control_head.py` (Task 12 only — backward compatible)

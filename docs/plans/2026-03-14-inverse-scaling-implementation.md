# Inverse Scaling Research — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MultiScaleLaplacianFilter (L0/L1/L2 parallel processing) and experiment infrastructure to validate inverse size scaling in learned Hodge classification, fix curl blindness, and re-benchmark sheaf diffusion.

**Architecture:** Wrap existing `SpectralFilterEnsemble` with parallel L1/L2 processing paths. Project higher-order signals back to nodes via boundary operators (L1→nodes) and node-triangle incidence (L2→nodes). Learnable gated fusion. Experiment scripts for training, OOD evaluation, ablations, and analysis.

**Tech Stack:** Python 3.12, PyTorch, PyG, torchdiffeq, gudhi, NetworkX, matplotlib

**Spec:** `docs/plans/2026-03-14-inverse-scaling-research-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/spectral/multiscale_filter.py` | Create | `MultiScaleLaplacianFilter` — parallel L0/L1/L2 spectral filtering + gated fusion |
| `src/wave/dynamics.py` | Modify | Add `use_multiscale` option to `MultiFilterDynamics` |
| `src/cell_complex/cell_complex.py` | Modify | Add `node_triangle_incidence()` method |
| `scripts/experiments/inverse_scaling/train.py` | Create | Training script for hodge_class with configurable filters/seeds |
| `scripts/experiments/inverse_scaling/evaluate_ood.py` | Create | OOD evaluation at multiple graph sizes |
| `scripts/experiments/inverse_scaling/audit_class_balance.py` | Create | Exp 0a: class distribution audit |
| `scripts/experiments/inverse_scaling/exact_baseline.py` | Create | Exp 0b: exact hodge decomposition baseline |
| `scripts/experiments/inverse_scaling/run_all.py` | Create | Orchestrator for full experiment suite |
| `scripts/experiments/inverse_scaling/analyze_results.py` | Create | Statistical tests + per-class breakdown |
| `scripts/experiments/inverse_scaling/plot_scaling_curves.py` | Create | Publication figures |
| `tests/test_spectral/test_multiscale_filter.py` | Create | Tests for MultiScaleLaplacianFilter |
| `docs/experiments/inverse-scaling/results.md` | Create | Live results table |

---

## Chunk 1: MultiScaleLaplacianFilter + Node-Triangle Incidence

### Task 1: Node-Triangle Incidence Matrix

**Files:**
- Modify: `src/cell_complex/cell_complex.py`
- Test: `tests/test_spectral/test_multiscale_filter.py`

The chain complex property `B1 @ B2 = 0` means we cannot use boundary operators to project triangle signals to nodes. Instead, compute a node-triangle incidence matrix where entry `[n, t] = 1` if node `n` is a vertex of triangle `t`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_spectral/test_multiscale_filter.py
"""Tests for multi-scale Laplacian filtering."""
import torch
import pytest
from src.cell_complex.cell_complex import CellComplex


def _make_cc_with_triangles(embedding_dim=32):
    """Build a CC with known triangles for testing."""
    cc = CellComplex(embedding_dim=embedding_dim)
    for i in range(5):
        cc.add_0_cell(torch.randn(embedding_dim), cell_type="node")
    # Edges forming two triangles: (0,1,2) and (1,2,3)
    cc.add_1_cell(0, 1, torch.randn(embedding_dim), relation_type="edge")
    cc.add_1_cell(1, 2, torch.randn(embedding_dim), relation_type="edge")
    cc.add_1_cell(0, 2, torch.randn(embedding_dim), relation_type="edge")
    cc.add_1_cell(2, 3, torch.randn(embedding_dim), relation_type="edge")
    cc.add_1_cell(1, 3, torch.randn(embedding_dim), relation_type="edge")
    # 2-cells
    cc.add_2_cell([0, 1, 2], torch.randn(embedding_dim))
    cc.add_2_cell([1, 2, 3], torch.randn(embedding_dim))
    return cc


def test_node_triangle_incidence_shape():
    cc = _make_cc_with_triangles()
    I_nt = cc.node_triangle_incidence()
    assert I_nt.shape == (5, 2)  # 5 nodes, 2 triangles


def test_node_triangle_incidence_values():
    cc = _make_cc_with_triangles()
    I_nt = cc.node_triangle_incidence()
    # Triangle 0 = (0,1,2): nodes 0,1,2 should be 1
    assert I_nt[0, 0] == 1.0
    assert I_nt[1, 0] == 1.0
    assert I_nt[2, 0] == 1.0
    assert I_nt[3, 0] == 0.0
    assert I_nt[4, 0] == 0.0
    # Triangle 1 = (1,2,3): nodes 1,2,3 should be 1
    assert I_nt[1, 1] == 1.0
    assert I_nt[2, 1] == 1.0
    assert I_nt[3, 1] == 1.0
    assert I_nt[0, 1] == 0.0


def test_node_triangle_incidence_no_triangles():
    cc = CellComplex(embedding_dim=32)
    for i in range(3):
        cc.add_0_cell(torch.randn(32), cell_type="node")
    cc.add_1_cell(0, 1, torch.randn(32), relation_type="edge")
    I_nt = cc.node_triangle_incidence()
    assert I_nt.shape == (3, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_spectral/test_multiscale_filter.py::test_node_triangle_incidence_shape -v`
Expected: FAIL — `node_triangle_incidence` doesn't exist

- [ ] **Step 3: Implement node_triangle_incidence()**

Add to `src/cell_complex/cell_complex.py` in the CellComplex class, after the `boundary_operator` method:

```python
    def node_triangle_incidence(self) -> torch.Tensor:
        """Compute node-triangle incidence matrix.

        Returns a (num_0_cells, num_2_cells) matrix where entry [n, t] = 1
        if node n is a vertex of triangle t. This is NOT a boundary operator
        (B1 @ B2 = 0 by chain complex property) — it's a containment relation.
        """
        n_nodes = self.num_cells(0)
        n_triangles = self.num_cells(2)
        if n_triangles == 0:
            return torch.zeros(n_nodes, 0)

        I_nt = torch.zeros(n_nodes, n_triangles)
        for t_idx, boundary_edges in enumerate(self._2_cell_boundaries):
            nodes = set()
            for e_idx in boundary_edges:
                nodes.add(self._1_cell_sources[e_idx])
                nodes.add(self._1_cell_targets[e_idx])
            for node_idx in nodes:
                if node_idx < n_nodes:
                    I_nt[node_idx, t_idx] = 1.0
        return I_nt
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_spectral/test_multiscale_filter.py -v -k "node_triangle"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cell_complex/cell_complex.py tests/test_spectral/test_multiscale_filter.py
git commit -m "feat: node-triangle incidence matrix for multi-scale Laplacian projection"
```

---

### Task 2: MultiScaleLaplacianFilter Core

**Files:**
- Create: `src/spectral/multiscale_filter.py`
- Test: `tests/test_spectral/test_multiscale_filter.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_spectral/test_multiscale_filter.py`:

```python
from src.spectral.multiscale_filter import MultiScaleLaplacianFilter


def test_multiscale_filter_forward_shape():
    cc = _make_cc_with_triangles(embedding_dim=32)
    msf = MultiScaleLaplacianFilter(embedding_dim=32)
    signal = cc.get_embeddings(0)  # (5, 32)
    diffusion_time = torch.tensor(0.5)
    out = msf(cc, signal, diffusion_time=diffusion_time)
    assert out.shape == signal.shape  # (5, 32)


def test_multiscale_filter_no_triangles():
    """Should work with L0+L1 only when no 2-cells exist."""
    cc = CellComplex(embedding_dim=32)
    for i in range(5):
        cc.add_0_cell(torch.randn(32), cell_type="node")
    cc.add_1_cell(0, 1, torch.randn(32), relation_type="edge")
    cc.add_1_cell(1, 2, torch.randn(32), relation_type="edge")
    cc.add_1_cell(2, 3, torch.randn(32), relation_type="edge")

    msf = MultiScaleLaplacianFilter(embedding_dim=32)
    signal = cc.get_embeddings(0)
    out = msf(cc, signal, diffusion_time=torch.tensor(0.5))
    assert out.shape == signal.shape


def test_multiscale_filter_no_edges():
    """Should fall back to identity when no edges."""
    cc = CellComplex(embedding_dim=32)
    for i in range(3):
        cc.add_0_cell(torch.randn(32), cell_type="node")

    msf = MultiScaleLaplacianFilter(embedding_dim=32)
    signal = cc.get_embeddings(0)
    out = msf(cc, signal, diffusion_time=torch.tensor(0.5))
    assert out.shape == signal.shape


def test_multiscale_gates_are_learnable():
    msf = MultiScaleLaplacianFilter(embedding_dim=32)
    # Should have gate parameters
    param_names = [n for n, _ in msf.named_parameters()]
    assert any("gate" in n for n in param_names)


def test_multiscale_gradients_flow():
    cc = _make_cc_with_triangles(embedding_dim=32)
    msf = MultiScaleLaplacianFilter(embedding_dim=32)
    signal = cc.get_embeddings(0).requires_grad_(True)
    out = msf(cc, signal, diffusion_time=torch.tensor(0.5))
    out.sum().backward()
    assert signal.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_spectral/test_multiscale_filter.py -v -k "multiscale"`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement MultiScaleLaplacianFilter**

Create `src/spectral/multiscale_filter.py`:

```python
"""Multi-scale Laplacian filtering across L0, L1, L2 Hodge Laplacians.

Enables the model to process signals at all three topological scales:
- L0 (nodes): standard graph diffusion
- L1 (edges): edge-level flow processing (curl detection)
- L2 (triangles): higher-order cycle processing

Signals from L1 and L2 are projected back to nodes via boundary operators
(B1 for edges) and node-triangle incidence (I_nt for triangles).
"""

import torch
import torch.nn as nn
from src.cell_complex.cell_complex import CellComplex
from src.spectral.laplacian import hodge_laplacian_0, hodge_laplacian_1, hodge_laplacian_2
from src.spectral.decomposition import spectral_decomposition


class MultiScaleLaplacianFilter(nn.Module):
    """Parallel spectral filtering on L0, L1, L2 with gated fusion.

    Runs the same filter type on all three Hodge Laplacians, projects
    higher-order outputs back to node space, and fuses with learned gates.
    """

    def __init__(
        self,
        embedding_dim: int,
        filter_type: str = "wave_cosine",
        skip_l1: bool = False,
        skip_l2: bool = False,
        **filter_kwargs,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.skip_l1 = skip_l1
        self.skip_l2 = skip_l2

        from src.wave.spectral_filters import create_spectral_filter

        # One filter per Laplacian scale
        self.filter_L0 = create_spectral_filter(filter_type, dim=0, **filter_kwargs)
        self.filter_L1 = create_spectral_filter(filter_type, dim=1, **filter_kwargs)
        self.filter_L2 = create_spectral_filter(filter_type, dim=2, **filter_kwargs)

        # Projection: edge-space (L1 output) → node-space
        self.proj_L1 = nn.Linear(embedding_dim, embedding_dim)
        # Projection: triangle-space (L2 output) → node-space
        self.proj_L2 = nn.Linear(embedding_dim, embedding_dim)

        # Learnable gates (sigmoid-activated)
        self.gate_L0 = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        self.gate_L1 = nn.Parameter(torch.tensor(0.0))
        self.gate_L2 = nn.Parameter(torch.tensor(-1.0))  # start small

    def forward(
        self,
        cc: CellComplex,
        signal: torch.Tensor,
        diffusion_time: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run multi-scale filtering and fuse results.

        Args:
            cc: Cell complex.
            signal: Node signal of shape (N, D).
            diffusion_time: Positive scalar for filter parameterization.

        Returns:
            Fused node signal of shape (N, D).
        """
        device = signal.device
        N = signal.shape[0]
        filter_kwargs = {"diffusion_time": diffusion_time} if diffusion_time is not None else {}

        # ---- L0 path (nodes) ----
        try:
            out_L0 = self.filter_L0(cc, signal, **filter_kwargs)
        except (RuntimeError, ValueError):
            out_L0 = signal

        # ---- L1 path (edges → nodes) ----
        out_L1_node = torch.zeros_like(signal)
        n_edges = cc.num_cells(1)
        if n_edges > 0:
            edge_signal = cc.get_embeddings(1).to(device)
            try:
                filtered_edges = self.filter_L1(cc, edge_signal, **filter_kwargs)
                # Project edges → nodes via B1
                B1 = cc.boundary_operator(1).to(device)
                projected = B1 @ filtered_edges  # (N, D)
                out_L1_node = self.proj_L1(projected)
            except (RuntimeError, ValueError):
                pass

        # ---- L2 path (triangles → nodes) ----
        out_L2_node = torch.zeros_like(signal)
        n_triangles = cc.num_cells(2)
        if n_triangles > 0:
            tri_signal = cc.get_embeddings(2).to(device)
            try:
                filtered_tris = self.filter_L2(cc, tri_signal, **filter_kwargs)
                # Project triangles → nodes via incidence matrix
                I_nt = cc.node_triangle_incidence().to(device)
                projected = I_nt @ filtered_tris  # (N, D)
                out_L2_node = self.proj_L2(projected)
            except (RuntimeError, ValueError):
                pass

        # ---- Gated fusion ----
        g0 = torch.sigmoid(self.gate_L0)
        g1 = torch.sigmoid(self.gate_L1)
        g2 = torch.sigmoid(self.gate_L2)

        fused = g0 * out_L0 + g1 * out_L1_node + g2 * out_L2_node
        return fused
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_spectral/test_multiscale_filter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/spectral/multiscale_filter.py tests/test_spectral/test_multiscale_filter.py
git commit -m "feat: MultiScaleLaplacianFilter with L0/L1/L2 parallel processing"
```

---

### Task 3: Wire MultiScaleLaplacianFilter into MultiFilterDynamics

**Files:**
- Modify: `src/wave/dynamics.py:280-405`
- Test: `tests/test_spectral/test_multiscale_filter.py` (extend)

`MultiFilterDynamics` currently creates filters with a single `laplacian_dim`. Add a `use_multiscale` option that wraps each filter with `MultiScaleLaplacianFilter`.

- [ ] **Step 1: Write failing test**

Append to `tests/test_spectral/test_multiscale_filter.py`:

```python
from src.wave.dynamics import MultiFilterDynamics


def test_multifilter_with_multiscale():
    cc = _make_cc_with_triangles(embedding_dim=32)
    mfd = MultiFilterDynamics(
        embedding_dim=32,
        filter_types=["wave_cosine"],
        use_multiscale=True,
        include_identity=True,
    )
    signal = cc.get_embeddings(0)
    dt = torch.tensor(0.5)
    wd = torch.tensor(0.1)
    out = mfd(cc, signal, dt, wd)
    assert out.shape == signal.shape


def test_multifilter_multiscale_num_paths():
    mfd = MultiFilterDynamics(
        embedding_dim=32,
        filter_types=["chebyshev", "wave_cosine"],
        use_multiscale=True,
        include_identity=True,
    )
    # 2 multiscale filters + identity = 3 paths
    assert mfd.num_filters == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_spectral/test_multiscale_filter.py -v -k "multifilter"`
Expected: FAIL — `use_multiscale` param not recognized

- [ ] **Step 3: Add use_multiscale to MultiFilterDynamics**

In `src/wave/dynamics.py`, modify `MultiFilterDynamics.__init__` (around line 292):

Add `use_multiscale: bool = False` parameter.

When `use_multiscale=True`, replace the filter creation block:

```python
        if use_multiscale:
            from src.spectral.multiscale_filter import MultiScaleLaplacianFilter
            self.filters = nn.ModuleList([
                MultiScaleLaplacianFilter(
                    embedding_dim, filter_type=ft, **filter_kwargs,
                )
                for ft in filter_types
            ])
        else:
            self.filters = nn.ModuleList([
                create_spectral_filter(ft, dim=laplacian_dim, **filter_kwargs)
                for ft in filter_types
            ])
```

The `forward()` method doesn't need changes — `MultiScaleLaplacianFilter.forward()` has the same interface `(cc, signal, diffusion_time=...)`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_spectral/test_multiscale_filter.py -v`
Expected: PASS

- [ ] **Step 5: Run broader wave tests**

Run: `.venv/bin/python -m pytest tests/test_wave/ -x -q`
Expected: PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add src/wave/dynamics.py tests/test_spectral/test_multiscale_filter.py
git commit -m "feat: wire MultiScaleLaplacianFilter into MultiFilterDynamics"
```

---

## Chunk 2: Experiment Infrastructure

### Task 4: Class Balance Audit (Experiment 0a)

**Files:**
- Create: `scripts/experiments/inverse_scaling/audit_class_balance.py`

This is a CPU-only script that generates hodge_class datasets at each test size and reports the actual class distribution. Critical for identifying the curl-underrepresentation confound.

- [ ] **Step 1: Create the audit script**

```python
#!/usr/bin/env python
"""Experiment 0a: Audit hodge_class class balance at each graph size.

Reports the actual gradient/curl/harmonic distribution in generated datasets.
Identifies whether curl is systematically underrepresented at small sizes
(a confound for the inverse scaling claim).

Usage:
    PYTHONPATH=. python scripts/experiments/inverse_scaling/audit_class_balance.py
"""

import sys
from collections import Counter
from src.benchmarks.benchmark_dataset import BenchmarkDataset

TEST_SIZES = [20, 40, 80, 120, 160, 200, 320]
SAMPLES_PER_SIZE = 1000
CLASS_NAMES = {0: "gradient", 1: "curl", 2: "harmonic"}


def main():
    print("Hodge Class Balance Audit")
    print("=" * 60)
    print(f"{'Size':>6} | {'gradient':>10} | {'curl':>10} | {'harmonic':>10} | {'curl %':>8}")
    print("-" * 60)

    for n in TEST_SIZES:
        ds = BenchmarkDataset(
            num_samples=SAMPLES_PER_SIZE,
            task_type="hodge_class",
            n_nodes=n,
            embedding_dim=64,
        )
        labels = [s[3] for s in ds.samples]
        counts = Counter(labels)
        total = sum(counts.values())
        curl_pct = 100 * counts.get(1, 0) / total if total > 0 else 0

        print(f"{n:>6} | {counts.get(0, 0):>10} | {counts.get(1, 0):>10} | "
              f"{counts.get(2, 0):>10} | {curl_pct:>7.1f}%")

    print("=" * 60)
    print("If curl % is significantly lower at small sizes,")
    print("this is a confound for the inverse scaling claim.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create directory and test it runs**

```bash
mkdir -p scripts/experiments/inverse_scaling
PYTHONPATH=. .venv/bin/python scripts/experiments/inverse_scaling/audit_class_balance.py
```

Expected: Table showing class distribution per size. Watch for curl underrepresentation.

- [ ] **Step 3: Commit**

```bash
git add scripts/experiments/inverse_scaling/audit_class_balance.py
git commit -m "feat: class balance audit for hodge inverse scaling (Exp 0a)"
```

---

### Task 5: Exact Decomposition Baseline (Experiment 0b)

**Files:**
- Create: `scripts/experiments/inverse_scaling/exact_baseline.py`

Zero-GPU-cost baseline: compute exact Hodge decomposition on generated data, classify by argmax of component norms.

- [ ] **Step 1: Create the baseline script**

```python
#!/usr/bin/env python
"""Experiment 0b: Exact Hodge decomposition baseline.

Classifies hodge_class samples using exact decomposition (no learned model).
If exact decomposition achieves 100% at all sizes, the inverse scaling
result is about the learning dynamics, not the math.

Usage:
    PYTHONPATH=. python scripts/experiments/inverse_scaling/exact_baseline.py
"""

import torch
from collections import Counter
from src.benchmarks.benchmark_dataset import BenchmarkDataset
from src.spectral.decomposition import hodge_decomposition

TEST_SIZES = [20, 40, 80, 120, 160, 200, 320]
SAMPLES_PER_SIZE = 500
CLASS_NAMES = {0: "gradient", 1: "curl", 2: "harmonic"}


def main():
    print("Exact Hodge Decomposition Baseline")
    print("=" * 70)
    print(f"{'Size':>6} | {'Accuracy':>8} | {'Grad':>6} | {'Curl':>6} | {'Harm':>6} | {'Errors':>6}")
    print("-" * 70)

    for n in TEST_SIZES:
        ds = BenchmarkDataset(
            num_samples=SAMPLES_PER_SIZE,
            task_type="hodge_class",
            n_nodes=n,
            embedding_dim=64,
        )

        correct = 0
        per_class_correct = Counter()
        per_class_total = Counter()
        errors = 0

        for sample in ds.samples:
            cc, _, _, answer = sample[:4]
            n_edges = cc.num_cells(1)
            if n_edges == 0:
                errors += 1
                continue

            edge_embs = cc.get_embeddings(1)
            signal = edge_embs[:, 0]  # Signal stored in dim 0

            try:
                gradient, curl, harmonic = hodge_decomposition(cc, signal, dim=1)
                norms = [gradient.norm().item(), curl.norm().item(),
                         harmonic.norm().item()]
                predicted = norms.index(max(norms))
            except (RuntimeError, ValueError):
                errors += 1
                continue

            per_class_total[answer] += 1
            if predicted == answer:
                correct += 1
                per_class_correct[answer] += 1

        total = sum(per_class_total.values())
        acc = 100 * correct / total if total > 0 else 0
        g_acc = 100 * per_class_correct[0] / max(per_class_total[0], 1)
        c_acc = 100 * per_class_correct[1] / max(per_class_total[1], 1)
        h_acc = 100 * per_class_correct[2] / max(per_class_total[2], 1)

        print(f"{n:>6} | {acc:>7.1f}% | {g_acc:>5.1f}% | {c_acc:>5.1f}% | "
              f"{h_acc:>5.1f}% | {errors:>6}")

    print("=" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test it runs**

```bash
PYTHONPATH=. .venv/bin/python scripts/experiments/inverse_scaling/exact_baseline.py
```

Expected: Near-100% accuracy at all sizes (exact decomposition should perfectly recover the planted signal). Any deviation indicates noise from the 0.1-scale noise added during task generation.

- [ ] **Step 3: Commit**

```bash
git add scripts/experiments/inverse_scaling/exact_baseline.py
git commit -m "feat: exact Hodge decomposition baseline (Exp 0b)"
```

---

### Task 6: Training Script

**Files:**
- Create: `scripts/experiments/inverse_scaling/train.py`
- Create: `scripts/experiments/inverse_scaling/configs/baseline.yaml`

Configurable training script for hodge_class with multiple filter/scale/seed options.

- [ ] **Step 1: Create config**

```yaml
# scripts/experiments/inverse_scaling/configs/baseline.yaml
# Exp 1: Baseline — v11 ensemble on L0 only
experiment:
  name: baseline
  task: hodge_class
  seeds: [42, 123, 456, 789, 1024]
  epochs: 30
  patience: 15

model:
  embedding_dim: 64
  gnn_hidden: 128
  gnn_layers: 2
  tat_layers: 2
  tat_heads: 4
  tat_ff_dim: 256
  wave_mode: ensemble
  filter_types: [chebyshev, wave_cosine, heat]
  include_identity: true
  include_sheaf: false
  use_multiscale: false
  use_topological_pe: true

training:
  train_n_range: [16, 32]
  train_samples: 5000
  val_samples: 500
  learning_rate: 0.0003
  weight_decay: 0.01
  batch_size: 8
  label_smoothing: 0.1
  scheduler: cosine_annealing

output:
  checkpoint_dir: data/experiments/inverse_scaling/checkpoints
  results_dir: data/experiments/inverse_scaling/results
```

- [ ] **Step 2: Create training script**

```python
#!/usr/bin/env python
"""Train hodge_class model for inverse scaling experiments.

Trains a HierarchicalMultiHopModel on hodge_class with configurable
filters, seeds, and multi-scale options. Saves checkpoints and per-epoch
metrics for OOD evaluation.

Usage:
    PYTHONPATH=. python scripts/experiments/inverse_scaling/train.py configs/baseline.yaml
    PYTHONPATH=. python scripts/experiments/inverse_scaling/train.py configs/multiscale.yaml --seed 42
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml

from src.benchmarks.benchmark_dataset import BenchmarkDataset
from src.benchmarks.run_benchmark_suite import _build_model
from src.training.batch_utils import train_epoch_batched, evaluate_batched


def train_one_seed(config: dict, seed: int, device: torch.device):
    """Train a single seed and save checkpoint + metrics."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    exp = config["experiment"]
    mc = config["model"]
    tc = config["training"]
    oc = config["output"]

    task = exp["task"]
    epochs = exp["epochs"]
    patience = exp["patience"]

    print(f"\n{'=' * 60}")
    print(f"Seed {seed} | {exp['name']} | {task}")
    print(f"{'=' * 60}")

    # Generate datasets
    train_ds = BenchmarkDataset(
        num_samples=tc["train_samples"],
        task_type=task,
        n_nodes=sum(tc["train_n_range"]) // 2,
        embedding_dim=mc["embedding_dim"],
        n_nodes_range=tuple(tc["train_n_range"]),
    )
    val_ds = BenchmarkDataset(
        num_samples=tc["val_samples"],
        task_type=task,
        n_nodes=sum(tc["train_n_range"]) // 2,
        embedding_dim=mc["embedding_dim"],
        n_nodes_range=tuple(tc["train_n_range"]),
    )

    # Build model
    wave_config = {
        "wave_mode": mc.get("wave_mode", "ensemble"),
        "filter_types": mc.get("filter_types", ["chebyshev", "wave_cosine", "heat"]),
        "include_identity": mc.get("include_identity", True),
        "wave_strength_gate": True,
        "use_neural_ode": True,
    }
    model = _build_model("symmetric", mc, max_classes=3, device=device,
                          wave_config=wave_config, training_config=tc)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs,
    )

    best_bal_acc = 0.0
    patience_counter = 0
    metrics = []

    for epoch in range(epochs):
        t0 = time.time()

        loss = train_epoch_batched(
            model, train_ds, optimizer,
            batch_size=tc["batch_size"],
            label_smoothing=tc.get("label_smoothing", 0.1),
            device=device,
        )

        val_acc, _val_loss, bal_acc = evaluate_batched(
            model, val_ds, device=device,
        )

        elapsed = time.time() - t0
        scheduler.step()

        is_best = bal_acc > best_bal_acc
        if is_best:
            best_bal_acc = bal_acc
            patience_counter = 0
            # Save best checkpoint
            ckpt_dir = Path(oc["checkpoint_dir"]) / exp["name"]
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / f"seed_{seed}_best.pt")
        else:
            patience_counter += 1

        star = " *" if is_best else ""
        print(f"  Ep {epoch:3d} | loss {loss:.4f} | val {val_acc:.3f} "
              f"bal {bal_acc:.3f} | {elapsed:.0f}s{star}", flush=True)

        metrics.append({
            "epoch": epoch, "loss": loss, "val_acc": val_acc,
            "bal_acc": bal_acc, "time": elapsed, "best": is_best,
        })

        if patience_counter >= patience:
            print(f"  Early stop at epoch {epoch} (patience {patience})")
            break

    # Save metrics
    results_dir = Path(oc["results_dir"]) / exp["name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"seed_{seed}_train.json", "w") as f:
        json.dump({"seed": seed, "best_bal_acc": best_bal_acc, "epochs": metrics}, f, indent=2)

    print(f"  Best bal_acc: {best_bal_acc:.3f}")
    return best_bal_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config file")
    parser.add_argument("--seed", type=int, help="Run single seed (default: all)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = [args.seed] if args.seed else config["experiment"]["seeds"]

    results = {}
    for seed in seeds:
        best = train_one_seed(config, seed, device)
        results[seed] = best

    print(f"\nAll seeds: {results}")
    mean = sum(results.values()) / len(results)
    print(f"Mean best bal_acc: {mean:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create additional configs**

Create `scripts/experiments/inverse_scaling/configs/` with:
- `multiscale.yaml` — same as baseline but `use_multiscale: true`
- `sheaf_only.yaml` — `wave_mode: sheaf`
- `sheaf_ensemble.yaml` — `include_sheaf: true`
- `sheaf_multiscale.yaml` — `include_sheaf: true, use_multiscale: true`

Each is a copy of baseline.yaml with the relevant field changed.

- [ ] **Step 4: Test the training script runs (CPU, 1 seed, 1 epoch)**

```bash
PYTHONPATH=. .venv/bin/python scripts/experiments/inverse_scaling/train.py \
  scripts/experiments/inverse_scaling/configs/baseline.yaml --seed 42
```

Expected: Starts training, completes at least 1 epoch without errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/inverse_scaling/
git commit -m "feat: inverse scaling training script + experiment configs"
```

---

### Task 7: OOD Evaluation Script

**Files:**
- Create: `scripts/experiments/inverse_scaling/evaluate_ood.py`

Evaluates a trained checkpoint at multiple OOD graph sizes. This produces the scaling curves.

- [ ] **Step 1: Create evaluation script**

```python
#!/usr/bin/env python
"""Evaluate trained hodge_class model at multiple OOD graph sizes.

Loads a checkpoint from train.py, generates test datasets at each
target size, and reports per-class accuracy. Produces the scaling
curves that are the core deliverable of this research.

Usage:
    PYTHONPATH=. python scripts/experiments/inverse_scaling/evaluate_ood.py \
        configs/baseline.yaml --seed 42
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import yaml

from src.benchmarks.benchmark_dataset import BenchmarkDataset
from src.benchmarks.run_benchmark_suite import _build_model

TEST_SIZES = [20, 40, 80, 120, 160, 200, 320]
TEST_SAMPLES = 500
CLASS_NAMES = {0: "gradient", 1: "curl", 2: "harmonic"}


def evaluate_at_size(model, n_nodes, embedding_dim, device):
    """Generate test data at given size and evaluate."""
    ds = BenchmarkDataset(
        num_samples=TEST_SAMPLES,
        task_type="hodge_class",
        n_nodes=n_nodes,
        embedding_dim=embedding_dim,
    )

    # Custom eval loop to collect per-class predictions
    # (evaluate_batched returns (accuracy, avg_loss, balanced_accuracy))
    model.eval()
    per_class = {0: {"correct": 0, "total": 0},
                 1: {"correct": 0, "total": 0},
                 2: {"correct": 0, "total": 0}}
    correct = 0
    total = 0

    import torch
    with torch.no_grad():
        for sample in ds.samples:
            cc, query, _, answer = sample[:4]
            cc = cc.clone().to(device)
            logits = model(cc, query.to(device) if hasattr(query, "to") else query)
            pred = int(logits.argmax(dim=-1).item())
            per_class[answer]["total"] += 1
            if pred == answer:
                per_class[answer]["correct"] += 1
                correct += 1
            total += 1

    val_acc = correct / max(total, 1)
    per_class_accs = [
        per_class[c]["correct"] / max(per_class[c]["total"], 1)
        for c in range(3)
    ]
    bal_acc = sum(per_class_accs) / 3

    per_class_acc = {}
    for cls_id, stats in per_class.items():
        acc = stats["correct"] / max(stats["total"], 1)
        per_class_acc[CLASS_NAMES[cls_id]] = {
            "accuracy": acc,
            "count": stats["total"],
        }

    return {
        "n_nodes": n_nodes,
        "val_acc": val_acc,
        "bal_acc": bal_acc,
        "per_class": per_class_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config file")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    exp = config["experiment"]
    mc = config["model"]
    tc = config["training"]
    oc = config["output"]

    # Build model and load checkpoint
    wave_config = {
        "wave_mode": mc.get("wave_mode", "ensemble"),
        "filter_types": mc.get("filter_types", ["chebyshev", "wave_cosine", "heat"]),
        "include_identity": mc.get("include_identity", True),
        "wave_strength_gate": True,
        "use_neural_ode": True,
    }
    model = _build_model("symmetric", mc, max_classes=3, device=device,
                          wave_config=wave_config, training_config=tc)

    ckpt_path = Path(oc["checkpoint_dir"]) / exp["name"] / f"seed_{args.seed}_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Evaluating {exp['name']} seed={args.seed}")
    print(f"{'Size':>6} | {'Bal Acc':>8} | {'Grad':>6} | {'Curl':>6} | {'Harm':>6}")
    print("-" * 50)

    all_results = []
    for n in TEST_SIZES:
        result = evaluate_at_size(model, n, mc["embedding_dim"], device)
        pc = result["per_class"]
        print(f"{n:>6} | {result['bal_acc']:>7.1%} | "
              f"{pc['gradient']['accuracy']:>5.1%} | "
              f"{pc['curl']['accuracy']:>5.1%} | "
              f"{pc['harmonic']['accuracy']:>5.1%}")
        all_results.append(result)

    # Save results
    results_dir = Path(oc["results_dir"]) / exp["name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"seed_{args.seed}_ood.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_dir / f'seed_{args.seed}_ood.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/experiments/inverse_scaling/evaluate_ood.py
git commit -m "feat: OOD evaluation script for inverse scaling curves"
```

---

### Task 8: Run-All Orchestrator + Results Template

**Files:**
- Create: `scripts/experiments/inverse_scaling/run_all.py`
- Create: `docs/experiments/inverse-scaling/results.md`

- [ ] **Step 1: Create orchestrator**

```python
#!/usr/bin/env python
"""Run the full inverse scaling experiment suite.

Orchestrates: Exp 0a (audit) → 0b (exact baseline) → 1-2d (training)
→ OOD evaluation → analysis.

Usage:
    PYTHONPATH=. python scripts/experiments/inverse_scaling/run_all.py
    PYTHONPATH=. python scripts/experiments/inverse_scaling/run_all.py --exp 1
    PYTHONPATH=. python scripts/experiments/inverse_scaling/run_all.py --exp 2a --seed 42
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CONFIGS = {
    "1":  "configs/baseline.yaml",
    "2a": "configs/multiscale.yaml",
    "2b": "configs/sheaf_only.yaml",
    "2c": "configs/sheaf_ensemble.yaml",
    "2d": "configs/sheaf_multiscale.yaml",
}


def run(cmd: list[str]):
    print(f">>> {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR.parent.parent.parent))
    if result.returncode != 0:
        print(f"FAILED with exit code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="Run specific experiment (0a, 0b, 1, 2a-2d)")
    parser.add_argument("--seed", type=int, help="Run single seed")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only eval")
    args = parser.parse_args()

    python = sys.executable

    if args.exp is None or args.exp == "0a":
        print("\n=== Experiment 0a: Class Balance Audit ===")
        run([python, str(SCRIPT_DIR / "audit_class_balance.py")])

    if args.exp is None or args.exp == "0b":
        print("\n=== Experiment 0b: Exact Decomposition Baseline ===")
        run([python, str(SCRIPT_DIR / "exact_baseline.py")])

    experiments = [args.exp] if args.exp and args.exp in CONFIGS else list(CONFIGS.keys())

    for exp_id in experiments:
        config_path = str(SCRIPT_DIR / CONFIGS[exp_id])
        print(f"\n=== Experiment {exp_id}: {CONFIGS[exp_id]} ===")

        if not args.skip_train:
            cmd = [python, str(SCRIPT_DIR / "train.py"), config_path]
            if args.seed:
                cmd += ["--seed", str(args.seed)]
            run(cmd)

        # OOD evaluation for each seed
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        seeds = [args.seed] if args.seed else config["experiment"]["seeds"]
        for seed in seeds:
            run([python, str(SCRIPT_DIR / "evaluate_ood.py"),
                 config_path, "--seed", str(seed)])

    print("\n=== All experiments complete ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create results template**

```markdown
# Inverse Scaling Experiment Results

Live results table. Updated as experiments complete.

## Experiment 0a: Class Balance Audit

| Size | Gradient | Curl | Harmonic | Curl % |
|------|----------|------|----------|--------|
| 20 | — | — | — | — |
| 40 | — | — | — | — |
| 80 | — | — | — | — |
| 120 | — | — | — | — |
| 160 | — | — | — | — |
| 200 | — | — | — | — |
| 320 | — | — | — | — |

## Experiment 0b: Exact Decomposition Baseline

| Size | Overall | Gradient | Curl | Harmonic |
|------|---------|----------|------|----------|
| — | — | — | — | — |

## Scaling Curves (Balanced Accuracy)

| Size | Baseline (1) | MultiScale (2a) | Sheaf (2b) | Sheaf+Ens (2c) | Sheaf+MS (2d) |
|------|-------------|-----------------|-----------|---------------|--------------|
| 20 | — | — | — | — | — |
| 40 | — | — | — | — | — |
| 80 | — | — | — | — | — |
| 120 | — | — | — | — | — |
| 160 | — | — | — | — | — |
| 200 | — | — | — | — | — |
| 320 | — | — | — | — | — |

## Per-Class Accuracy (Best Config)

| Size | Gradient | Curl | Harmonic |
|------|----------|------|----------|
| — | — | — | — |
```

- [ ] **Step 3: Commit**

```bash
mkdir -p docs/experiments/inverse-scaling
git add scripts/experiments/inverse_scaling/run_all.py docs/experiments/inverse-scaling/results.md
git commit -m "feat: experiment orchestrator + results template"
```

---

## Chunk 3: Ablation Configs + Analysis

### Task 9: Ablation Configs

**Files:**
- Create: `scripts/experiments/inverse_scaling/configs/ablation_*.yaml`

Create 5 ablation configs (copies of baseline with one component disabled):

- [ ] **Step 1: Create ablation configs**

- `ablation_no_l1.yaml`: `use_multiscale: true` but skip L1 in MultiScaleLaplacianFilter (add `skip_l1: true` config option)
- `ablation_no_l2.yaml`: `use_multiscale: true` but skip L2
- `ablation_nowave.yaml`: `wave_mode: none` (disable spectral filtering entirely)
- `ablation_no_tat.yaml`: `tat_layers: 0` (GNN only)
- `ablation_no_ode.yaml`: `use_neural_ode: false`

For no-L1 and no-L2, pass `skip_l1=True` or `skip_l2=True` to `MultiScaleLaplacianFilter.__init__`. The flags are stored and checked in `forward()`:
```python
# In __init__:
self.skip_l1 = skip_l1
self.skip_l2 = skip_l2
# In forward():
if not self.skip_l1 and n_edges > 0:
    # L1 processing...
```

- [ ] **Step 2: Commit**

```bash
git add scripts/experiments/inverse_scaling/configs/ablation_*.yaml
git commit -m "feat: ablation experiment configs for inverse scaling"
```

---

### Task 10: Analysis + Plotting Scripts

**Files:**
- Create: `scripts/experiments/inverse_scaling/analyze_results.py`
- Create: `scripts/experiments/inverse_scaling/plot_scaling_curves.py`

- [ ] **Step 1: Create analysis script**

The analysis script loads all seed results, computes means/stds, runs statistical tests (paired permutation test for monotonic scaling), and outputs summary tables.

Key outputs:
- Mean ± std balanced accuracy per size per experiment
- One-sided paired permutation test: accuracy(n) < accuracy(2n) across seeds
- Per-class breakdown table
- Scaling retention ratios (accuracy at n / accuracy at n=20)

- [ ] **Step 2: Create plotting script**

The plotting script generates publication-quality figures:
- Scaling curves with error bars (all experiments on one plot)
- Per-class breakdown (stacked or grouped bar chart)
- Ablation comparison (grouped bar at n=80 and n=160)
- Heatmap (topology × size for Experiment 4)

Use matplotlib with consistent styling.

- [ ] **Step 3: Commit**

```bash
git add scripts/experiments/inverse_scaling/analyze_results.py scripts/experiments/inverse_scaling/plot_scaling_curves.py
git commit -m "feat: analysis and plotting scripts for inverse scaling experiments"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Node-triangle incidence matrix | cell_complex.py |
| 2 | MultiScaleLaplacianFilter core | multiscale_filter.py |
| 3 | Wire into MultiFilterDynamics | dynamics.py |
| 4 | Class balance audit (Exp 0a) | audit_class_balance.py |
| 5 | Exact decomposition baseline (Exp 0b) | exact_baseline.py |
| 6 | Training script + configs | train.py, configs/*.yaml |
| 7 | OOD evaluation script | evaluate_ood.py |
| 8 | Orchestrator + results template | run_all.py, results.md |
| 9 | Ablation configs | configs/ablation_*.yaml |
| 10 | Analysis + plotting | analyze_results.py, plot_scaling_curves.py |

**Tasks 1-3:** Architecture (MultiScaleLaplacianFilter). Must complete before experiments.
**Tasks 4-8:** Experiment infrastructure. Can run Exp 0a/0b on CPU immediately.
**Tasks 9-10:** Ablation + analysis. Can be done after main experiments.

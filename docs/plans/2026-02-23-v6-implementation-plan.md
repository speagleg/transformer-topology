# V6 Semantic Dual-Track Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement dual-track semantic model comparison (WikiText-pretrained DSM vs frozen Qwen2.5-3B + adapter) with shared infrastructure fixes for catastrophic forgetting, batched training, auxiliary contrastive loss, and active dual-topology feedback.

**Architecture:** Two parallel experiments sharing common fixes (multi-head classifier, 50% replay, contrastive loss, batched training, active topology feedback). Track 1 fine-tunes a WikiText-pretrained 250M DSM. Track 2 trains a GraphFormer adapter over a frozen 4-bit Qwen2.5-3B. Both produce identical output types (semantic_features, semantic_bias, graph_embedding) for fair comparison.

**Tech Stack:** PyTorch, transformers (HuggingFace), autoawq (4-bit quantization), datasets (WikiText-103)

**Design Doc:** `docs/plans/2026-02-23-v6-semantic-dual-track-design.md`

---

## Task 1: Multi-Head Classifier

Replace the destructive `_rebuild_classifier()` with persistent per-task classifier heads.

**Files:**
- Create: `src/training/multi_head_classifier.py`
- Test: `tests/test_training/test_multi_head_classifier.py`
- Modify: `src/benchmarks/run_comparison.py:156-167` (replace single classifier)
- Modify: `scripts/run_dsm_curriculum.py:42-54` (remove `_rebuild_classifier`)

**Step 1: Write the failing test**

```python
# tests/test_training/test_multi_head_classifier.py
"""Tests for MultiHeadClassifier."""
import torch
import torch.nn as nn
from src.training.multi_head_classifier import MultiHeadClassifier


class TestMultiHeadClassifier:
    def test_forward_returns_correct_shape(self):
        task_classes = {"diverse": 11, "bfs": 16, "graph_completion": 2}
        clf = MultiHeadClassifier(input_dim=64, task_classes=task_classes)
        x = torch.randn(1, 64)
        out = clf(x, "diverse")
        assert out.shape == (1, 11)

    def test_different_tasks_different_shapes(self):
        task_classes = {"diverse": 11, "bfs": 16, "graph_completion": 2}
        clf = MultiHeadClassifier(input_dim=64, task_classes=task_classes)
        x = torch.randn(1, 64)
        assert clf(x, "diverse").shape == (1, 11)
        assert clf(x, "bfs").shape == (1, 16)
        assert clf(x, "graph_completion").shape == (1, 2)

    def test_heads_persist_across_calls(self):
        """Verify weights are NOT reset between task switches."""
        task_classes = {"diverse": 11, "bfs": 16}
        clf = MultiHeadClassifier(input_dim=64, task_classes=task_classes)
        x = torch.randn(1, 64)

        # Record weights before switching tasks
        w_before = clf.heads["diverse"].weight.clone()
        _ = clf(x, "bfs")  # switch to different task
        w_after = clf.heads["diverse"].weight
        assert torch.equal(w_before, w_after)

    def test_add_task_creates_new_head(self):
        task_classes = {"diverse": 11}
        clf = MultiHeadClassifier(input_dim=64, task_classes=task_classes)
        clf.add_task("new_task", 5)
        x = torch.randn(1, 64)
        assert clf(x, "new_task").shape == (1, 5)

    def test_unknown_task_raises(self):
        task_classes = {"diverse": 11}
        clf = MultiHeadClassifier(input_dim=64, task_classes=task_classes)
        x = torch.randn(1, 64)
        try:
            clf(x, "nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_from_task_registry(self):
        """Build from the full TASK_REGISTRY."""
        from src.benchmarks.benchmark_dataset import TASK_REGISTRY, get_max_classes
        task_classes = {t: get_max_classes(t) for t in TASK_REGISTRY}
        clf = MultiHeadClassifier(input_dim=64, task_classes=task_classes)
        x = torch.randn(1, 64)
        for task, n_cls in task_classes.items():
            assert clf(x, task).shape == (1, n_cls)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training/test_multi_head_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.training.multi_head_classifier'`

**Step 3: Write implementation**

```python
# src/training/multi_head_classifier.py
"""Persistent per-task classifier heads.

Replaces the destructive _rebuild_classifier() pattern that destroyed weights
when switching tasks during curriculum training.
"""
import torch.nn as nn


class MultiHeadClassifier(nn.Module):
    """Dictionary of classifier heads keyed by task name.

    Each head is a simple Linear layer. Heads persist across task switches —
    no weight destruction.
    """

    def __init__(self, input_dim: int, task_classes: dict[str, int]):
        super().__init__()
        self.input_dim = input_dim
        self.heads = nn.ModuleDict({
            task: nn.Linear(input_dim, n_classes)
            for task, n_classes in task_classes.items()
        })

    def forward(self, x, task: str):
        return self.heads[task](x)

    def add_task(self, task: str, n_classes: int):
        """Add a new task head (for tasks discovered at runtime)."""
        self.heads[task] = nn.Linear(self.input_dim, n_classes)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_training/test_multi_head_classifier.py -v`
Expected: All 6 PASS

**Step 5: Commit**

```bash
git add src/training/multi_head_classifier.py tests/test_training/test_multi_head_classifier.py
git commit -m "feat: MultiHeadClassifier — persistent per-task heads, no weight destruction"
```

---

## Task 2: Contrastive Loss for Semantic Features

Give the semantic model a direct gradient signal independent of classification.

**Files:**
- Create: `src/training/contrastive_loss.py`
- Test: `tests/test_training/test_contrastive_loss.py`

**Step 1: Write the failing test**

```python
# tests/test_training/test_contrastive_loss.py
"""Tests for semantic contrastive loss."""
import torch
from src.training.contrastive_loss import SemanticContrastiveLoss


class TestSemanticContrastiveLoss:
    def test_returns_scalar(self):
        loss_fn = SemanticContrastiveLoss(temperature=0.1)
        features = torch.randn(10, 32)
        adjacency = torch.randint(0, 2, (10, 10)).float()
        loss = loss_fn(features, adjacency)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0

    def test_identical_features_low_loss(self):
        """If all features are identical, positive pairs have max similarity."""
        loss_fn = SemanticContrastiveLoss(temperature=0.1)
        features = torch.ones(5, 32)
        # All nodes connected — all pairs are positive
        adjacency = torch.ones(5, 5)
        loss = loss_fn(features, adjacency)
        assert loss.item() >= 0

    def test_gradient_flows(self):
        loss_fn = SemanticContrastiveLoss(temperature=0.1)
        features = torch.randn(8, 32, requires_grad=True)
        adjacency = torch.randint(0, 2, (8, 8)).float()
        loss = loss_fn(features, adjacency)
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_no_positive_pairs_returns_zero(self):
        """If adjacency is all zeros (no positive pairs), loss should be 0."""
        loss_fn = SemanticContrastiveLoss(temperature=0.1)
        features = torch.randn(5, 32)
        adjacency = torch.zeros(5, 5)
        loss = loss_fn(features, adjacency)
        assert loss.item() == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training/test_contrastive_loss.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/training/contrastive_loss.py
"""Auxiliary contrastive loss for semantic features.

Gives the semantic model (DSM or LLM adapter) a direct gradient signal:
nodes that are connected (or structurally similar) should have similar
semantic features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss on node semantic features.

    Positive pairs: nodes connected by an edge (adjacency[i,j] > 0).
    Negative pairs: all other node pairs.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            features: (N, D) semantic features per node
            adjacency: (N, N) binary adjacency matrix (positive pairs)

        Returns:
            Scalar loss (0.0 if no positive pairs).
        """
        N = features.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=features.device)

        # Normalize features
        features = F.normalize(features, dim=-1)

        # Cosine similarity matrix
        sim = features @ features.T / self.temperature  # (N, N)

        # Mask out self-similarity
        mask_self = ~torch.eye(N, dtype=torch.bool, device=features.device)
        positive_mask = (adjacency > 0) & mask_self

        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        # For each node with positive pairs, compute InfoNCE
        # log_softmax over all non-self nodes, then average over positive pairs
        log_prob = sim - torch.logsumexp(
            sim.masked_fill(~mask_self, float('-inf')), dim=-1, keepdim=True
        )

        # Average log-prob over positive pairs
        loss = -(log_prob * positive_mask.float()).sum() / positive_mask.sum()
        return loss
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_training/test_contrastive_loss.py -v`
Expected: All 4 PASS

**Step 5: Commit**

```bash
git add src/training/contrastive_loss.py tests/test_training/test_contrastive_loss.py
git commit -m "feat: SemanticContrastiveLoss — auxiliary InfoNCE for semantic features"
```

---

## Task 3: Feature Replay Buffer

Save penultimate-layer activations from Phase A for efficient replay in Phase B/C.

**Files:**
- Create: `src/training/feature_replay.py`
- Test: `tests/test_training/test_feature_replay.py`

**Step 1: Write the failing test**

```python
# tests/test_training/test_feature_replay.py
"""Tests for feature replay buffer."""
import torch
from src.training.feature_replay import FeatureReplayBuffer


class TestFeatureReplayBuffer:
    def test_save_and_sample(self):
        buf = FeatureReplayBuffer(max_per_task=100)
        features = torch.randn(64)
        buf.save("diverse", features, label=3)
        samples = buf.sample("diverse", n=1)
        assert len(samples) == 1
        feat, label = samples[0]
        assert feat.shape == (64,)
        assert label == 3

    def test_sample_multiple(self):
        buf = FeatureReplayBuffer(max_per_task=100)
        for i in range(20):
            buf.save("bfs", torch.randn(64), label=i % 16)
        samples = buf.sample("bfs", n=10)
        assert len(samples) == 10

    def test_max_capacity(self):
        buf = FeatureReplayBuffer(max_per_task=5)
        for i in range(10):
            buf.save("diverse", torch.randn(64), label=0)
        assert len(buf.buffers["diverse"]) == 5

    def test_sample_empty_task(self):
        buf = FeatureReplayBuffer(max_per_task=100)
        samples = buf.sample("nonexistent", n=5)
        assert len(samples) == 0

    def test_sample_ratio(self):
        buf = FeatureReplayBuffer(max_per_task=100)
        for i in range(50):
            buf.save("diverse", torch.randn(64), label=i % 11)
        # Sample 50% of stored
        samples = buf.sample_ratio("diverse", ratio=0.5)
        assert len(samples) == 25

    def test_all_tasks(self):
        buf = FeatureReplayBuffer(max_per_task=100)
        buf.save("diverse", torch.randn(64), label=0)
        buf.save("bfs", torch.randn(64), label=1)
        assert set(buf.tasks()) == {"diverse", "bfs"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training/test_feature_replay.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/training/feature_replay.py
"""Feature replay buffer for catastrophic forgetting prevention.

Stores penultimate-layer activations from earlier phases for efficient
replay. Cheaper than re-running full forward passes on old data.
"""
import random
from collections import defaultdict

import torch


class FeatureReplayBuffer:
    """Store and sample (feature, label) pairs per task."""

    def __init__(self, max_per_task: int = 1000):
        self.max_per_task = max_per_task
        self.buffers: dict[str, list[tuple[torch.Tensor, int]]] = defaultdict(list)

    def save(self, task: str, features: torch.Tensor, label: int):
        """Store a (feature, label) pair. Evicts oldest if at capacity."""
        buf = self.buffers[task]
        buf.append((features.detach().cpu(), label))
        if len(buf) > self.max_per_task:
            buf.pop(0)

    def sample(self, task: str, n: int) -> list[tuple[torch.Tensor, int]]:
        """Sample n items from task buffer. Returns fewer if buffer smaller."""
        buf = self.buffers.get(task, [])
        if not buf:
            return []
        return random.sample(buf, min(n, len(buf)))

    def sample_ratio(self, task: str, ratio: float) -> list[tuple[torch.Tensor, int]]:
        """Sample ratio * buffer_size items."""
        buf = self.buffers.get(task, [])
        n = int(len(buf) * ratio)
        return self.sample(task, max(1, n)) if buf else []

    def tasks(self) -> list[str]:
        """Return list of tasks with stored features."""
        return list(self.buffers.keys())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_training/test_feature_replay.py -v`
Expected: All 6 PASS

**Step 5: Commit**

```bash
git add src/training/feature_replay.py tests/test_training/test_feature_replay.py
git commit -m "feat: FeatureReplayBuffer — store activations for replay across phases"
```

---

## Task 4: Richer TopoBridge Output

Expand TopoBridge from producing only `semantic_bias` to also producing `semantic_features` and `graph_embedding`. Increase `semantic_bias_proj` init from `std=0.01` to `std=0.1`.

**Files:**
- Modify: `src/llm/topo_bridge.py:100-191` (TopoBridgeDecoder)
- Test: `tests/test_llm/test_topo_bridge_v6.py`

**Step 1: Write the failing test**

```python
# tests/test_llm/test_topo_bridge_v6.py
"""Tests for v6 richer TopoBridge output."""
import torch
from src.llm.topo_bridge import TopoBridgeDecoder


class TestTopoBridgeDecoderV6:
    def test_forward_returns_three_outputs(self):
        """Decoder should return (semantic_out, semantic_bias, semantic_features, graph_embedding)."""
        dec = TopoBridgeDecoder(llm_dim=64, topo_dim=32)
        topo_memory = torch.randn(5, 64)  # 5 nodes
        llm_hidden = torch.randn(8, 64)  # 8 prefix tokens
        result = dec(topo_memory, llm_hidden, semantic_weight=0.5)
        # Should be a 4-tuple now
        assert len(result) == 4
        semantic_out, semantic_bias, semantic_features, graph_embedding = result
        assert semantic_out.shape == (5, 32)
        assert semantic_bias.shape == (5, 5)
        assert semantic_features.shape == (5, 32)
        assert graph_embedding.shape == (32,)

    def test_semantic_bias_init_larger(self):
        """semantic_bias_proj should use std=0.1 (not 0.01)."""
        dec = TopoBridgeDecoder(llm_dim=64, topo_dim=32)
        w = dec.semantic_bias_proj.weight
        # With std=0.1, weights should have larger magnitude than std=0.01
        assert w.std().item() > 0.05  # much larger than old 0.01
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm/test_topo_bridge_v6.py -v`
Expected: FAIL (decoder returns 2-tuple, not 4-tuple)

**Step 3: Modify TopoBridgeDecoder**

In `src/llm/topo_bridge.py`, modify `TopoBridgeDecoder`:

1. Change `__init__` (around line 113-118): increase std from 0.01 to 0.1, add `graph_pool` projection
2. Change `forward` (around line 120-146): return 4-tuple instead of 2-tuple
3. Change `forward_batched` (around line 148-191): return 4-tuple per graph

Specific changes:
- Line 117: `nn.init.normal_(self.semantic_bias_proj.weight, std=0.01)` → `std=0.1`
- Add `self.graph_pool = nn.Linear(llm_dim, topo_dim)` in `__init__`
- `forward()` returns `(semantic_out, semantic_bias, semantic_features, graph_embedding)` where:
  - `semantic_features = semantic_out` (same tensor, separate name for clarity)
  - `graph_embedding = self.graph_pool(llm_hidden.mean(dim=0))`
- `forward_batched()` returns list of 4-tuples

**Step 4: Run test to verify it passes, then run existing tests**

Run: `pytest tests/test_llm/test_topo_bridge_v6.py -v`
Run: `pytest tests/ -x --timeout=60 -k "not test_spectral_gap_variable_sizes"`
Expected: All pass (need to update callers that unpack 2-tuple to handle 4-tuple)

**Step 5: Update callers**

Files that unpack TopoBridge output (update all to handle 4-tuple):
- `src/reasoning_loop/executive_loop.py:231-236` — unpack `_, semantic_bias = self.topo_bridge(...)` → `_, semantic_bias, semantic_features, graph_embedding = self.topo_bridge(...)`
- `src/reasoning_loop/executive_loop.py:339-348` — batched path, same change
- `src/benchmarks/run_comparison.py:213-228` — legacy TopoBridge path

Store `semantic_features` and `graph_embedding` in the loop for later use by TAT and classifier.

**Step 6: Run full test suite**

Run: `pytest tests/ -x --timeout=60 -k "not test_spectral_gap_variable_sizes"`
Expected: All pass

**Step 7: Commit**

```bash
git add src/llm/topo_bridge.py src/reasoning_loop/executive_loop.py src/benchmarks/run_comparison.py tests/test_llm/test_topo_bridge_v6.py
git commit -m "feat: richer TopoBridge output — semantic_features + graph_embedding + larger bias init"
```

---

## Task 5: Integrate MultiHeadClassifier into HierarchicalMultiHopModel

Replace the single `nn.Sequential` classifier with `MultiHeadClassifier` in the model.

**Files:**
- Modify: `src/benchmarks/run_comparison.py:156-167` (classifier init), `:202-241` (forward)
- Modify: `scripts/run_dsm_curriculum.py:42-54` (remove `_rebuild_classifier`)
- Test: `tests/test_multi_head_integration.py`

**Step 1: Write the failing test**

```python
# tests/test_multi_head_integration.py
"""Test MultiHeadClassifier integration with HierarchicalMultiHopModel."""
import torch
from src.benchmarks.run_comparison import HierarchicalMultiHopModel
from src.benchmarks.multi_hop import MultiHopDataset


def _make_model():
    return HierarchicalMultiHopModel(
        embedding_dim=16, gnn_hidden=32,
        gnn_spatial_layers=1, gnn_spectral_layers=1,
        max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2,
        tat_ff_dim=32, max_classes=6,
        max_iterations=2, convergence_threshold=0.1,
        use_wave_dynamics=False, use_higher_order=False,
        use_multi_head_classifier=True,
    )


class TestMultiHeadIntegration:
    def test_model_has_multi_head_classifier(self):
        model = _make_model()
        assert hasattr(model, 'multi_head_classifier')

    def test_forward_with_task_name(self):
        model = _make_model()
        ds = MultiHopDataset(1, min_hops=1, max_hops=5, num_distractors=3, embedding_dim=16)
        cc, query, target, answer = ds[0][:4]
        # forward with task name should use multi-head classifier
        logits = model(cc, query, target, task="diverse")
        assert logits.shape[-1] == 11  # diverse has 11 classes

    def test_forward_without_task_falls_back(self):
        model = _make_model()
        ds = MultiHopDataset(1, min_hops=1, max_hops=5, num_distractors=3, embedding_dim=16)
        cc, query, target, answer = ds[0][:4]
        # forward without task uses legacy classifier
        logits = model(cc, query, target)
        assert logits.shape[-1] == 6  # max_classes default
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_multi_head_integration.py -v`
Expected: FAIL (no `use_multi_head_classifier` param)

**Step 3: Modify HierarchicalMultiHopModel**

In `src/benchmarks/run_comparison.py`:

1. `__init__`: Add `use_multi_head_classifier=False` param. If True, create `MultiHeadClassifier` with all 14 tasks from `TASK_REGISTRY`. Keep legacy classifier as fallback.
2. `forward`: Add `task=None` param. If `task` is set and multi_head_classifier exists, use it. Otherwise use legacy classifier.

**Step 4: Remove `_rebuild_classifier` from curriculum script**

In `scripts/run_dsm_curriculum.py`:
- Remove `_rebuild_classifier` function (lines 42-54)
- Remove all calls to `_rebuild_classifier(model, task)` throughout the script
- Pass `task=task_name` to model forward in train_epoch / train_epoch_with_replay

**Step 5: Run tests**

Run: `pytest tests/test_multi_head_integration.py tests/test_training/test_multi_head_classifier.py -v`
Run: `pytest tests/ -x --timeout=60 -k "not test_spectral_gap_variable_sizes"`
Expected: All pass

**Step 6: Commit**

```bash
git add src/benchmarks/run_comparison.py src/training/multi_head_classifier.py scripts/run_dsm_curriculum.py tests/test_multi_head_integration.py
git commit -m "feat: integrate MultiHeadClassifier into model + remove _rebuild_classifier"
```

---

## Task 6: Class-Weighted Loss

Address class imbalance (labeled_reasoning class 2 = 0% accuracy in v5).

**Files:**
- Create: `src/training/class_weights.py`
- Test: `tests/test_training/test_class_weights.py`

**Step 1: Write the failing test**

```python
# tests/test_training/test_class_weights.py
"""Tests for class weight computation."""
import torch
from src.training.class_weights import compute_class_weights


class TestClassWeights:
    def test_uniform_distribution(self):
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        weights = compute_class_weights(labels, num_classes=3)
        assert weights.shape == (3,)
        # Uniform → equal weights
        assert torch.allclose(weights[0], weights[1], atol=1e-5)

    def test_imbalanced_distribution(self):
        labels = torch.tensor([0, 0, 0, 0, 1])
        weights = compute_class_weights(labels, num_classes=2)
        # Class 1 is rarer → higher weight
        assert weights[1] > weights[0]

    def test_missing_class_gets_max_weight(self):
        labels = torch.tensor([0, 0, 1, 1])
        weights = compute_class_weights(labels, num_classes=3)
        # Class 2 never appears → highest weight
        assert weights[2] >= weights[0]
        assert weights[2] >= weights[1]

    def test_normalized(self):
        labels = torch.tensor([0, 0, 0, 1, 2, 2])
        weights = compute_class_weights(labels, num_classes=3)
        # Should sum to num_classes (standard normalization)
        assert abs(weights.sum().item() - 3.0) < 1e-5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training/test_class_weights.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/training/class_weights.py
"""Compute class weights for imbalanced datasets."""
import torch


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to sum to num_classes.

    Missing classes get max weight (treated as count=1).
    """
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()

    # Replace zero counts with 1 (gives max weight to missing classes)
    counts = counts.clamp(min=1)

    weights = 1.0 / counts
    weights = weights * num_classes / weights.sum()  # normalize
    return weights
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_training/test_class_weights.py -v`
Expected: All 4 PASS

**Step 5: Commit**

```bash
git add src/training/class_weights.py tests/test_training/test_class_weights.py
git commit -m "feat: compute_class_weights — inverse-frequency weighting for imbalanced tasks"
```

---

## Task 7: Switch Curriculum to Batched Training

Replace per-sample sequential training with `train_epoch_batched` / `evaluate_batched`.

**Files:**
- Modify: `src/training/batch_utils.py:29-116` (extend `_forward_batch` for task, topo_features)
- Modify: `scripts/run_dsm_curriculum.py` (switch to batched functions)
- Test: `tests/test_training/test_batched_v6.py`

**Step 1: Write the failing test**

```python
# tests/test_training/test_batched_v6.py
"""Tests for v6 batched training extensions."""
import torch
from src.benchmarks.run_comparison import HierarchicalMultiHopModel
from src.benchmarks.benchmark_dataset import BenchmarkDataset
from src.training.batch_utils import train_epoch_batched, evaluate_batched


class TestBatchedV6:
    def _make_model_and_data(self):
        model = HierarchicalMultiHopModel(
            embedding_dim=16, gnn_hidden=32,
            gnn_spatial_layers=1, gnn_spectral_layers=1,
            max_freqs=8, tat_layers=1,
            tat_spatial_heads=2, tat_spectral_heads=2,
            tat_ff_dim=32, max_classes=6,
            max_iterations=2, convergence_threshold=0.1,
            use_wave_dynamics=False, use_higher_order=False,
        )
        ds = BenchmarkDataset(8, "diverse", 20, 16)
        return model, ds

    def test_train_epoch_batched_with_task(self):
        model, ds = self._make_model_and_data()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = train_epoch_batched(model, ds, opt, batch_size=4, task="diverse")
        assert loss > 0

    def test_evaluate_batched_with_task(self):
        model, ds = self._make_model_and_data()
        acc, loss = evaluate_batched(model, ds, batch_size=4, task="diverse")
        assert 0 <= acc <= 1
        assert loss > 0

    def test_train_with_topo_features(self):
        model, ds = self._make_model_and_data()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        topo_feat = torch.randn(6)
        loss = train_epoch_batched(model, ds, opt, batch_size=4, topo_features=topo_feat)
        assert loss > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training/test_batched_v6.py -v`
Expected: FAIL (train_epoch_batched doesn't accept `task` param)

**Step 3: Extend batch_utils**

In `src/training/batch_utils.py`:

1. Add `task=None` and `topo_features=None` params to `train_epoch_batched()` and `evaluate_batched()`
2. Pass `task` to model forward (for MultiHeadClassifier)
3. Pass `topo_features` to model forward
4. In `_forward_batch()`: add `task` and `topo_features` params, pass through to model

**Step 4: Update curriculum script to use batched training**

In `scripts/run_dsm_curriculum.py`:
1. Import `train_epoch_batched, evaluate_batched` from `src.training.batch_utils`
2. In `_run_phase()`: replace `train_epoch()` / `train_epoch_with_replay()` calls with `train_epoch_batched()`
3. Replace `evaluate()` calls with `evaluate_batched()`
4. Add `batch_size` to training config (default 8)
5. Keep sequential fallback for replay (or implement batched replay)

**Step 5: Run tests**

Run: `pytest tests/test_training/test_batched_v6.py -v`
Run: `pytest tests/ -x --timeout=60 -k "not test_spectral_gap_variable_sizes"`
Expected: All pass

**Step 6: Commit**

```bash
git add src/training/batch_utils.py scripts/run_dsm_curriculum.py tests/test_training/test_batched_v6.py
git commit -m "feat: switch curriculum to batched training — task + topo_features support"
```

---

## Task 8: DSM WikiText Pre-training Script

Create standalone script to pre-train DSM on WikiText-103 as a causal language model.

**Files:**
- Create: `scripts/pretrain_dsm.py`
- Test: (manual smoke test — LM pre-training isn't unit-testable)

**Step 1: Write the pre-training script**

```python
# scripts/pretrain_dsm.py
"""Pre-train DSM on WikiText-103 as a causal language model.

Saves only transformer layer weights (discards text embedding + LM head).
These weights serve as warm initialization for the graph pipeline.

Usage:
    python scripts/pretrain_dsm.py [--epochs 5] [--batch_size 32] [--output data/dsm_pretrained.pt]
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.llm.dsm import DistilledSemanticModel


class DSMForLM(nn.Module):
    """Wrap DSM with text embedding + LM head for language modeling."""

    def __init__(self, vocab_size, dsm_dim, num_heads, ff_dim, num_layers,
                 cross_attn_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dsm_dim)
        self.dsm = DistilledSemanticModel(
            hidden_dim=dsm_dim, num_heads=num_heads, ff_dim=ff_dim,
            num_layers=num_layers, cross_attn_layer=cross_attn_layer,
        )
        self.lm_head = nn.Linear(dsm_dim, vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)  # (batch, seq, dim)
        # DSM expects (seq, hidden_dim) per sample, run per-sample
        outputs = []
        for i in range(x.shape[0]):
            prefix = x[i]  # (seq, dim)
            topo_memory = torch.zeros(1, prefix.shape[-1], device=x.device)
            out = self.dsm(prefix, topo_memory)  # (seq, dim)
            outputs.append(out)
        hidden = torch.stack(outputs, dim=0)  # (batch, seq, dim)
        logits = self.lm_head(hidden)  # (batch, seq, vocab)
        return logits


def load_wikitext(seq_len=512, split="train"):
    """Load WikiText-103 using HuggingFace datasets."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    # Concatenate all text, tokenize, chunk into seq_len blocks
    all_text = "\n\n".join(ds["text"])
    tokens = tokenizer.encode(all_text)
    tokens = torch.tensor(tokens, dtype=torch.long)

    # Chunk
    n_chunks = len(tokens) // seq_len
    tokens = tokens[:n_chunks * seq_len].view(n_chunks, seq_len)
    return tokens, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dsm_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--ff_dim", type=int, default=4096)
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--cross_attn_layer", type=int, default=4)
    parser.add_argument("--output", type=str, default="data/dsm_pretrained.pt")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    print("Loading WikiText-103...")
    tokens, tokenizer = load_wikitext(args.seq_len)
    vocab_size = tokenizer.vocab_size
    print(f"  {len(tokens)} chunks of {args.seq_len} tokens, vocab={vocab_size}")

    loader = DataLoader(tokens, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = DSMForLM(
        vocab_size=vocab_size, dsm_dim=args.dsm_dim, num_heads=args.num_heads,
        ff_dim=args.ff_dim, num_layers=args.num_layers,
        cross_attn_layer=args.cross_attn_layer,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        t0 = time.time()
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1} batch {batch_idx+1}/{len(loader)} loss={avg:.4f}")

        elapsed = time.time() - t0
        avg_loss = total_loss / len(loader)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  ppl={ppl:.1f}  time={elapsed:.0f}s")

    # Save only DSM transformer layers (not embedding or LM head)
    dsm_state = model.dsm.state_dict()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dsm_state, out_path)
    print(f"\nSaved DSM weights ({len(dsm_state)} tensors) to {out_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/pretrain_dsm.py
git commit -m "feat: pretrain_dsm.py — WikiText-103 causal LM pre-training for DSM"
```

---

## Task 9: Gradual Unfreeze for DSM Backend

Add support for loading WikiText checkpoint and gradual layer unfreezing.

**Files:**
- Modify: `src/llm/dsm_backend.py`
- Test: `tests/test_llm/test_dsm_gradual_unfreeze.py`

**Step 1: Write the failing test**

```python
# tests/test_llm/test_dsm_gradual_unfreeze.py
"""Tests for DSM gradual unfreezing."""
import torch
from src.llm.dsm_backend import DSMBackend


class TestDSMGradualUnfreeze:
    def _make_backend(self):
        config = {
            "dsm_dim": 64, "num_heads": 4, "ff_dim": 256,
            "num_layers": 4, "cross_attn_layer": 1,
        }
        return DSMBackend(config)

    def test_freeze_all(self):
        backend = self._make_backend()
        backend.freeze_all()
        for p in backend.dsm.parameters():
            assert not p.requires_grad

    def test_unfreeze_top_n(self):
        backend = self._make_backend()
        backend.freeze_all()
        backend.unfreeze_top_n(2)  # unfreeze layers 3 and 4 (0-indexed: 2,3)
        frozen = sum(1 for p in backend.dsm.layers[:2].parameters() if not p.requires_grad)
        unfrozen = sum(1 for p in backend.dsm.layers[2:].parameters() if p.requires_grad)
        assert frozen > 0
        assert unfrozen > 0

    def test_unfreeze_all(self):
        backend = self._make_backend()
        backend.freeze_all()
        backend.unfreeze_all()
        for p in backend.dsm.parameters():
            assert p.requires_grad

    def test_load_pretrained_weights(self):
        backend = self._make_backend()
        # Simulate pretrained weights
        pretrained = backend.dsm.state_dict()
        # Modify a weight to verify loading works
        key = list(pretrained.keys())[0]
        pretrained[key] = torch.randn_like(pretrained[key])
        backend.load_pretrained(pretrained)
        assert torch.equal(backend.dsm.state_dict()[key], pretrained[key])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm/test_dsm_gradual_unfreeze.py -v`
Expected: FAIL (no freeze_all/unfreeze_top_n methods)

**Step 3: Modify DSMBackend**

Add to `src/llm/dsm_backend.py`:

```python
def freeze_all(self):
    """Freeze all DSM parameters."""
    for p in self.dsm.parameters():
        p.requires_grad = False

def unfreeze_top_n(self, n: int):
    """Unfreeze the top N transformer layers (highest index = closest to output)."""
    num_layers = len(self.dsm.layers)
    for i, layer in enumerate(self.dsm.layers):
        if i >= num_layers - n:
            for p in layer.parameters():
                p.requires_grad = True

def unfreeze_all(self):
    """Unfreeze all DSM parameters."""
    for p in self.dsm.parameters():
        p.requires_grad = True

def load_pretrained(self, state_dict: dict):
    """Load pre-trained weights (from WikiText pre-training)."""
    self.dsm.load_state_dict(state_dict, strict=False)
```

**Step 4: Run tests**

Run: `pytest tests/test_llm/test_dsm_gradual_unfreeze.py -v`
Expected: All 4 PASS

**Step 5: Commit**

```bash
git add src/llm/dsm_backend.py tests/test_llm/test_dsm_gradual_unfreeze.py
git commit -m "feat: DSMBackend gradual unfreeze — freeze_all, unfreeze_top_n, load_pretrained"
```

---

## Task 10: QwenGraphBackend + GraphFormer Adapter (Track 2)

Create the frozen Qwen2.5-3B backend with learned GraphFormer adapter.

**Files:**
- Create: `src/llm/graph_adapter.py`
- Create: `src/llm/qwen_backend.py`
- Test: `tests/test_llm/test_graph_adapter.py`
- Test: `tests/test_llm/test_qwen_backend.py`

**Step 1: Write adapter tests**

```python
# tests/test_llm/test_graph_adapter.py
"""Tests for GraphFormer adapter."""
import torch
from src.llm.graph_adapter import GraphFormerEncoder, GraphFormerDecoder


class TestGraphFormerEncoder:
    def test_output_shape(self):
        enc = GraphFormerEncoder(topo_dim=32, llm_dim=64, num_tokens=16, num_layers=2)
        node_embs = torch.randn(10, 32)  # 10 nodes
        task_id = torch.tensor(0)
        tokens = enc(node_embs, task_id)
        assert tokens.shape == (16, 64)

    def test_different_node_counts(self):
        enc = GraphFormerEncoder(topo_dim=32, llm_dim=64, num_tokens=16, num_layers=2)
        for n in [5, 10, 30]:
            tokens = enc(torch.randn(n, 32), torch.tensor(0))
            assert tokens.shape == (16, 64)


class TestGraphFormerDecoder:
    def test_output_shapes(self):
        dec = GraphFormerDecoder(llm_dim=64, topo_dim=32, num_layers=2)
        node_queries = torch.randn(10, 64)
        llm_hidden = torch.randn(16, 64)
        features, bias, graph_emb = dec(node_queries, llm_hidden)
        assert features.shape == (10, 32)
        assert bias.shape == (10, 10)
        assert graph_emb.shape == (32,)

    def test_gradient_flows(self):
        dec = GraphFormerDecoder(llm_dim=64, topo_dim=32, num_layers=2)
        node_queries = torch.randn(10, 64, requires_grad=True)
        llm_hidden = torch.randn(16, 64)
        features, bias, graph_emb = dec(node_queries, llm_hidden)
        loss = features.sum() + bias.sum() + graph_emb.sum()
        loss.backward()
        assert node_queries.grad is not None
```

**Step 2: Write backend test (mock LLM for CI)**

```python
# tests/test_llm/test_qwen_backend.py
"""Tests for QwenGraphBackend (uses mock LLM for CI)."""
import torch
from src.llm.qwen_backend import QwenGraphBackend


class TestQwenGraphBackend:
    def test_mock_mode_forward(self):
        """Test with mock LLM (no HuggingFace download needed)."""
        config = {"topo_dim": 32, "llm_dim": 64, "num_tokens": 16,
                  "adapter_layers": 2, "num_tasks": 14, "use_mock": True}
        backend = QwenGraphBackend(config)
        node_embs = torch.randn(10, 32)
        task_id = torch.tensor(0)
        features, bias, graph_emb = backend.forward_graph(node_embs, task_id)
        assert features.shape == (10, 32)
        assert bias.shape == (10, 10)
        assert graph_emb.shape == (32,)

    def test_trainable_params_exclude_llm(self):
        config = {"topo_dim": 32, "llm_dim": 64, "num_tokens": 16,
                  "adapter_layers": 2, "num_tasks": 14, "use_mock": True}
        backend = QwenGraphBackend(config)
        trainable = list(backend.trainable_parameters())
        assert len(trainable) > 0
        # All trainable params should be from adapter, not LLM
        for p in trainable:
            assert p.requires_grad
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_llm/test_graph_adapter.py tests/test_llm/test_qwen_backend.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 4: Write GraphFormer adapter**

```python
# src/llm/graph_adapter.py
"""GraphFormer adapter: encode graph topology as LLM-readable tokens, decode back.

Encoder: GNN node embeddings → K "graph tokens" for frozen LLM.
Decoder: LLM hidden states → per-node semantic features + bias + graph embedding.
"""
import torch
import torch.nn as nn


class GraphFormerEncoder(nn.Module):
    """Encode graph topology as K soft tokens for LLM consumption."""

    def __init__(self, topo_dim=32, llm_dim=2048, num_tokens=16, num_layers=2,
                 num_tasks=14):
        super().__init__()
        self.input_proj = nn.Linear(topo_dim, llm_dim)
        self.graph_queries = nn.Parameter(torch.randn(num_tokens, llm_dim) * 0.02)
        self.task_embedding = nn.Embedding(num_tasks, llm_dim)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                llm_dim, nhead=8, dim_feedforward=4 * llm_dim,
                batch_first=False, norm_first=True,
            )
            for _ in range(num_layers)
        ])

    def forward(self, node_embeddings, task_id):
        memory = self.input_proj(node_embeddings)  # (N, llm_dim)
        task_emb = self.task_embedding(task_id)  # (llm_dim,)
        queries = self.graph_queries + task_emb.unsqueeze(0)  # (K, llm_dim)
        for layer in self.layers:
            queries = layer(queries, memory)
        return queries  # (K, llm_dim)


class GraphFormerDecoder(nn.Module):
    """Decode LLM hidden states back to graph-space features."""

    def __init__(self, llm_dim=2048, topo_dim=32, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                llm_dim, nhead=8, dim_feedforward=4 * llm_dim,
                batch_first=False, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.feature_proj = nn.Linear(llm_dim, topo_dim)
        self.bias_proj = nn.Linear(topo_dim, topo_dim)
        self.graph_pool = nn.Linear(llm_dim, topo_dim)

    def forward(self, node_queries, llm_hidden_states):
        for layer in self.layers:
            node_queries = layer(node_queries, llm_hidden_states)
        features = self.feature_proj(node_queries)  # (N, topo_dim)
        bias = features @ self.bias_proj(features).T  # (N, N)
        graph_emb = self.graph_pool(llm_hidden_states.mean(0))  # (topo_dim,)
        return features, bias, graph_emb
```

**Step 5: Write QwenGraphBackend**

```python
# src/llm/qwen_backend.py
"""Frozen Qwen2.5-3B backend with GraphFormer adapter.

The LLM is fully frozen and 4-bit quantized. Only the GraphFormer
encoder/decoder are trainable (~20M params).
"""
import torch
import torch.nn as nn

from src.llm.graph_adapter import GraphFormerEncoder, GraphFormerDecoder


class QwenGraphBackend(nn.Module):
    """Frozen LLM + learned adapter for graph→semantic→graph translation."""

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
            # Mock LLM for testing (no HuggingFace download)
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
            self.llm = None  # loaded lazily

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
        """Full graph→LLM→graph pipeline.

        Args:
            node_embeddings: (N, topo_dim) from GNN
            task_id: scalar tensor (task index)

        Returns:
            (semantic_features, semantic_bias, graph_embedding)
        """
        graph_tokens = self.encoder(node_embeddings, task_id)  # (K, llm_dim)

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
```

**Step 6: Run tests**

Run: `pytest tests/test_llm/test_graph_adapter.py tests/test_llm/test_qwen_backend.py -v`
Expected: All pass

**Step 7: Commit**

```bash
git add src/llm/graph_adapter.py src/llm/qwen_backend.py tests/test_llm/test_graph_adapter.py tests/test_llm/test_qwen_backend.py
git commit -m "feat: GraphFormer adapter + QwenGraphBackend — Track 2 frozen LLM pipeline"
```

---

## Task 11: Qwen Analysis Wrapper for Embedding Topology

Create wrapper so the embedding topology analyzer can hook into Qwen's layers.

**Files:**
- Create: `src/topology_analyzer/qwen_wrapper.py`
- Test: `tests/test_topology_analyzer/test_qwen_wrapper.py`

**Step 1: Write the failing test**

```python
# tests/test_topology_analyzer/test_qwen_wrapper.py
"""Tests for QwenAnalysisWrapper."""
import torch
from src.topology_analyzer.qwen_wrapper import QwenAnalysisWrapper


class TestQwenAnalysisWrapper:
    def test_forward_shape(self):
        """Mock wrapper produces output of correct shape."""
        # Use mock mode (no real Qwen needed)
        wrapper = QwenAnalysisWrapper(llm_dim=64, num_layers=4, use_mock=True)
        x = torch.randn(1, 10, 64)  # (batch, seq, dim)
        out = wrapper(x)
        assert out.dim() == 3  # (batch, seq, dim)

    def test_exposes_layers(self):
        wrapper = QwenAnalysisWrapper(llm_dim=64, num_layers=4, use_mock=True)
        assert hasattr(wrapper, 'layers')
        assert len(wrapper.layers) == 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topology_analyzer/test_qwen_wrapper.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/topology_analyzer/qwen_wrapper.py
"""Wrap Qwen (or mock) for TransformerTopologyAnalyzer.

The embedding topology analyzer needs a model with `.layers` attribute
and `forward(x)` → output. This wraps the frozen Qwen or a mock.
"""
import torch
import torch.nn as nn


class QwenAnalysisWrapper(nn.Module):
    """Wraps a transformer model for topology analysis.

    Provides the `.layers` and single-arg `forward(x)` interface
    that TransformerTopologyAnalyzer expects.
    """

    def __init__(self, llm_dim=2048, num_layers=32, use_mock=False,
                 qwen_model=None):
        super().__init__()
        if use_mock or qwen_model is None:
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    llm_dim, nhead=4, dim_feedforward=4 * llm_dim,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ])
            self._is_mock = True
        else:
            # Extract layers from real Qwen model
            self.layers = qwen_model.model.layers
            self._is_mock = False
            self._qwen = qwen_model

    def forward(self, x):
        if self._is_mock:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            with torch.no_grad():
                out = self._qwen(
                    inputs_embeds=x,
                    output_hidden_states=False,
                )
                return out.last_hidden_state
```

**Step 4: Run tests**

Run: `pytest tests/test_topology_analyzer/test_qwen_wrapper.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/topology_analyzer/qwen_wrapper.py tests/test_topology_analyzer/test_qwen_wrapper.py
git commit -m "feat: QwenAnalysisWrapper — enables embedding topology analysis on Qwen"
```

---

## Task 12: Track-Specific Config Files

Create v6 YAML configs for both tracks.

**Files:**
- Create: `config/v6_track1.yaml`
- Create: `config/v6_track2.yaml`

**Step 1: Write Track 1 config**

```yaml
# config/v6_track1.yaml
# Track 1: WikiText-pretrained DSM (250M), fine-tuned with gradual unfreeze

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
  max_iterations: 5
  convergence_threshold: 0.05
  use_higher_order: true
  use_topological_pe: true
  use_structural_features: true
  use_wave_dynamics: true
  use_topo_feedback: true
  use_embedding_topo_feedback: true
  use_multi_head_classifier: true

wave:
  wave_mode: ensemble
  filter_types: [chebyshev, wave_cosine, heat, identity]
  wave_strength_gate: true
  use_neural_ode: true

llm:
  backend: dsm
  dsm_dim: 1024
  num_heads: 16
  ff_dim: 4096
  num_layers: 16
  cross_attn_layer: 4
  num_prefix: 8
  pretrained_path: data/dsm_pretrained.pt  # WikiText checkpoint
  gradual_unfreeze: true
  freeze_epochs: 5          # freeze DSM for first 5 epochs
  unfreeze_top_n_epochs: 15 # unfreeze top 4 layers at epoch 6-15
  unfreeze_top_n: 4

training:
  device: auto
  learning_rate: 0.001
  dsm_learning_rate: 0.00001   # Lower for pre-trained weights
  bridge_learning_rate: 0.0001
  weight_decay: 0.01
  label_smoothing: 0.1
  batch_size: 8
  epochs_phase_a: 30
  epochs_phase_b: 30
  epochs_phase_c: 30
  patience: 10
  max_norm: 5.0
  accumulation_steps: 4
  replay_ratio: 0.5
  use_class_weights: true
  contrastive_loss_weight: 0.1
  use_batched_training: true

topology_observer:
  enabled: true
  analyze_every: 5
  active_mode: true
  dsm_num_heads: 16
  dsm_hidden_dim: 1024

benchmark:
  train_samples: 5000
  val_samples: 500
  test_samples: 200
  train_n_nodes: 20
  train_n_nodes_min: 16
  train_n_nodes_max: 32
  test_n_nodes: [20, 40, 80]
  train_topologies: [ba, ws, grid, tree, ladder, sbm]
  test_topologies: [er, caveman]
  checkpoint_dir: data/dsm_results/checkpoints
  pregen_dir: data/dsm_datasets
```

**Step 2: Write Track 2 config**

```yaml
# config/v6_track2.yaml
# Track 2: Frozen Qwen2.5-3B (4-bit) + GraphFormer adapter (~20M trainable)

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
  max_iterations: 5
  convergence_threshold: 0.05
  use_higher_order: true
  use_topological_pe: true
  use_structural_features: true
  use_wave_dynamics: true
  use_topo_feedback: true
  use_embedding_topo_feedback: true
  use_multi_head_classifier: true

wave:
  wave_mode: ensemble
  filter_types: [chebyshev, wave_cosine, heat, identity]
  wave_strength_gate: true
  use_neural_ode: true

llm:
  backend: qwen
  qwen_model: Qwen/Qwen2.5-3B-AWQ
  llm_dim: 2048
  num_tokens: 16
  adapter_layers: 2
  extract_layer: 16
  num_tasks: 14

training:
  device: auto
  learning_rate: 0.001
  adapter_learning_rate: 0.0001
  bridge_learning_rate: 0.0001
  weight_decay: 0.01
  label_smoothing: 0.1
  batch_size: 8
  epochs_phase_a: 30
  epochs_phase_b: 30
  epochs_phase_c: 30
  patience: 10
  max_norm: 5.0
  accumulation_steps: 4
  replay_ratio: 0.5
  use_class_weights: true
  contrastive_loss_weight: 0.1
  use_batched_training: true

topology_observer:
  enabled: true
  analyze_every: 5
  active_mode: true
  dsm_num_heads: 16     # Qwen has 16 attention heads per layer
  dsm_hidden_dim: 2048  # Qwen hidden dim

benchmark:
  train_samples: 5000
  val_samples: 500
  test_samples: 200
  train_n_nodes: 20
  train_n_nodes_min: 16
  train_n_nodes_max: 32
  test_n_nodes: [20, 40, 80]
  train_topologies: [ba, ws, grid, tree, ladder, sbm]
  test_topologies: [er, caveman]
  checkpoint_dir: data/dsm_results/checkpoints
  pregen_dir: data/dsm_datasets
```

**Step 3: Commit**

```bash
git add config/v6_track1.yaml config/v6_track2.yaml
git commit -m "feat: v6 track configs — Track 1 (pretrained DSM) + Track 2 (Qwen + adapter)"
```

---

## Task 13: Wire Everything into Curriculum Training Script

Update `run_dsm_curriculum.py` to use all new components: multi-head classifier, batched training, contrastive loss, class weights, feature replay, gradual unfreeze, topology observer active mode.

**Files:**
- Modify: `scripts/run_dsm_curriculum.py`
- This is the largest single change. Modify incrementally.

**Step 1: Update imports and remove _rebuild_classifier**

Add imports for new modules. Remove `_rebuild_classifier` function and all calls.

**Step 2: Update `_build_dsm_optimizer` for Track 2 support**

If backend is "qwen", use adapter params instead of DSM params.

**Step 3: Add contrastive loss + class weights to training loop**

In `train_epoch_batched`:
- After classification loss, compute `semantic_contrastive_loss` on semantic features
- Use class-weighted `CrossEntropyLoss` per task

**Step 4: Add feature replay**

In Phase A: after each epoch, save penultimate features to FeatureReplayBuffer.
In Phase B/C: sample 50% replay from buffer, interleave with new data.

**Step 5: Add gradual unfreeze (Track 1)**

In `_run_phase()`:
- Check epoch count against `freeze_epochs` and `unfreeze_top_n_epochs` from config
- Call `backend.freeze_all()`, `backend.unfreeze_top_n()`, `backend.unfreeze_all()` as appropriate
- Rebuild optimizer after unfreezing (new params need to be in optimizer)

**Step 6: Switch to batched training**

Replace all `train_epoch()` / `evaluate()` calls with `train_epoch_batched()` / `evaluate_batched()`.

**Step 7: Pass task name through everything**

Pass `task=current_task` to all model forward calls and batched functions.

**Step 8: Test manually**

Run: `python scripts/run_dsm_curriculum.py config/v6_track1.yaml` with small overrides (2 epochs, 10 samples)
Expected: No crashes, observer logs appear, multi-head classifier used

**Step 9: Commit**

```bash
git add scripts/run_dsm_curriculum.py
git commit -m "feat: v6 curriculum — batched training, multi-head clf, contrastive loss, gradual unfreeze"
```

---

## Task 14: Update Analysis Script for V6

Add t-SNE visualization of semantic features and track contrastive loss.

**Files:**
- Modify: `scripts/analyze_v5_results.py` → rename to `scripts/analyze_results.py`

**Step 1: Add semantic feature extraction**

After each task evaluation, extract and save semantic features (N x embed_dim per sample) for t-SNE.

**Step 2: Add topology alert history**

Log topology observer alerts and computation graph metrics over training.

**Step 3: Commit**

```bash
git add scripts/analyze_results.py
git commit -m "feat: v6 analysis — semantic feature t-SNE + topology alert history"
```

---

## Task 15: Full Integration Test

Run the complete test suite and verify everything works together.

**Files:**
- All files from Tasks 1-14

**Step 1: Run all new tests**

```bash
pytest tests/test_training/ tests/test_llm/test_graph_adapter.py tests/test_llm/test_qwen_backend.py tests/test_llm/test_dsm_gradual_unfreeze.py tests/test_llm/test_topo_bridge_v6.py tests/test_multi_head_integration.py tests/test_topology_analyzer/test_qwen_wrapper.py -v
```

Expected: All pass

**Step 2: Run full existing test suite**

```bash
pytest tests/ -x --timeout=60 -k "not test_spectral_gap_variable_sizes"
```

Expected: All pass (no regressions)

**Step 3: Smoke test Track 1 config**

```bash
python scripts/run_dsm_curriculum.py config/v6_track1.yaml
```
(With manual override: 2 epochs, 10 samples per task)

Expected: Runs without error, observer logs appear, batched training used

**Step 4: Smoke test Track 2 config (mock mode)**

```bash
python scripts/run_dsm_curriculum.py config/v6_track2.yaml
```
(With `use_mock: true` override for CI)

Expected: Runs without error, GraphFormer adapter trains

**Step 5: Final commit**

```bash
git add -A
git commit -m "test: v6 full integration smoke tests pass"
```

---

## Execution Order & Dependencies

```
Task 1 (MultiHeadClassifier)
Task 2 (ContrastiveLoss)        ─── independent, can parallelize
Task 3 (FeatureReplayBuffer)
Task 6 (ClassWeights)

Task 4 (Richer TopoBridge) → Task 5 (Integrate MultiHead into Model)

Task 8 (WikiText pre-train script)
Task 9 (DSM gradual unfreeze)    ─── Track 1 specific

Task 10 (GraphFormer + Qwen backend)  ─── Track 2 specific
Task 11 (Qwen topology wrapper)

Task 7 (Batched training switch) → depends on Tasks 1, 4, 5

Task 12 (Config files) → depends on all above

Task 13 (Wire into curriculum) → depends on all above

Task 14 (Analysis script update) → after Task 13
Task 15 (Integration test) → after all
```

Tasks 1, 2, 3, 6, 8, 10, 11 can be dispatched in parallel.
Tasks 4 → 5 → 7 → 12 → 13 → 14 → 15 are sequential.

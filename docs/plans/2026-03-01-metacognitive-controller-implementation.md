# MetaCognitive Controller Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a MetaCognitive Controller that gives the model task-conditional gating, confidence-aware reasoning, and strategy selection — enabling it to learn WHEN to use text vs structure per task.

**Architecture:** Extend the existing ControlHead (in GNNExecutive) with task embedding input and 5 new heads (text_gate, structure_gate, uncertainty, iteration_budget, strategy_weights). Gates modulate classifier input features. Classifier heads become MLPs instead of single Linear layers. New Phase E trains metacognition across all 19 tasks simultaneously.

**Tech Stack:** Python 3.12, PyTorch, existing transformer-topology codebase. Test with `pytest`. Config in YAML.

**Design doc:** `docs/plans/2026-03-01-metacognitive-controller-design.md`

---

### Task 1: Extend ControlSignal dataclass with metacognition fields

**Files:**
- Modify: `src/gnn_executive/control_head.py:9-28`
- Test: `tests/test_gnn_executive/test_control_head.py` (existing tests still pass)

**Step 1: Write failing test**

Create `tests/test_gnn_executive/test_metacog_fields.py`:

```python
"""Tests for MetaCognitive Controller ControlSignal extensions."""
import torch
from src.gnn_executive.control_head import ControlSignal


def test_control_signal_has_metacog_fields():
    """ControlSignal should accept new metacog fields with None defaults."""
    cs = ControlSignal(
        frequency_gate=torch.randn(8),
        spatial_focus=torch.randn(20),
        confidence_weights=torch.randn(20),
        diffusion_time=torch.tensor(1.0),
        wave_damping=torch.tensor(0.5),
    )
    # New fields default to None
    assert cs.text_gate is None
    assert cs.structure_gate is None
    assert cs.uncertainty is None
    assert cs.iteration_budget is None
    assert cs.strategy_weights is None


def test_control_signal_metacog_fields_populated():
    """ControlSignal should store metacog fields when provided."""
    cs = ControlSignal(
        frequency_gate=torch.randn(8),
        spatial_focus=torch.randn(20),
        confidence_weights=torch.randn(20),
        diffusion_time=torch.tensor(1.0),
        wave_damping=torch.tensor(0.5),
        text_gate=torch.tensor(0.8),
        structure_gate=torch.tensor(0.3),
        uncertainty=torch.tensor(0.2),
        iteration_budget=torch.tensor(3.0),
        strategy_weights=torch.softmax(torch.randn(4), dim=0),
    )
    assert cs.text_gate.item() == 0.8
    assert cs.structure_gate.item() == 0.3
    assert cs.uncertainty.item() == 0.2
    assert cs.iteration_budget.item() == 3.0
    assert cs.strategy_weights.shape == (4,)
    assert abs(cs.strategy_weights.sum().item() - 1.0) < 1e-5
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_gnn_executive/test_metacog_fields.py -v`

Expected: FAIL — `ControlSignal.__init__() got unexpected keyword argument 'text_gate'`

**Step 3: Implement — add fields to ControlSignal**

In `src/gnn_executive/control_head.py`, replace lines 9-28:

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
        semantic_weight: scalar — soft weight for DSM attention bias [0,1].
        filter_weights: (num_filters,) — softmax weights over spectral filter ensemble.
        text_gate: scalar [0,1] — how much to weight text features in classifier.
        structure_gate: scalar [0,1] — how much to weight structural features.
        uncertainty: scalar [0,1] — temperature-scaled calibrated uncertainty.
        iteration_budget: scalar > 0 — learned soft iteration limit.
        strategy_weights: (4,) softmax — weights over reasoning strategies.
    """
    frequency_gate: torch.Tensor
    spatial_focus: torch.Tensor
    confidence_weights: torch.Tensor
    diffusion_time: torch.Tensor
    wave_damping: torch.Tensor
    semantic_weight: torch.Tensor = None
    filter_weights: torch.Tensor = None
    text_gate: torch.Tensor = None
    structure_gate: torch.Tensor = None
    uncertainty: torch.Tensor = None
    iteration_budget: torch.Tensor = None
    strategy_weights: torch.Tensor = None
```

**Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_gnn_executive/test_metacog_fields.py tests/test_gnn_executive/test_control_head.py -v`

Expected: ALL PASS (new + existing)

**Step 5: Commit**

```bash
git add src/gnn_executive/control_head.py tests/test_gnn_executive/test_metacog_fields.py
git commit -m "feat: extend ControlSignal with metacognition fields (text_gate, uncertainty, etc.)"
```

---

### Task 2: Create MetaCognitiveController module

**Files:**
- Create: `src/gnn_executive/metacognitive_controller.py`
- Test: `tests/test_gnn_executive/test_metacog_fields.py` (add tests)

**Step 1: Write failing tests**

Append to `tests/test_gnn_executive/test_metacog_fields.py`:

```python
import pytest
from src.gnn_executive.metacognitive_controller import MetaCognitiveController


class TestMetaCognitiveController:
    """Tests for MetaCognitiveController module."""

    @pytest.fixture
    def controller(self):
        return MetaCognitiveController(
            embedding_dim=32, num_freqs=8, num_filters=4,
            num_tasks=19, task_embed_dim=128,
            use_topo_feedback=True, use_embedding_topo_feedback=True,
        )

    def test_output_is_control_signal(self, controller):
        """forward() should return a ControlSignal dataclass."""
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert isinstance(cs, ControlSignal)

    def test_existing_fields_present(self, controller):
        """All original ControlHead fields should still be produced."""
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert cs.frequency_gate.shape == (8,)
        assert cs.spatial_focus.shape == (20,)
        assert cs.confidence_weights.shape == (20,)
        assert cs.diffusion_time.dim() == 0
        assert cs.wave_damping.dim() == 0
        assert cs.semantic_weight.dim() == 0
        assert cs.filter_weights.shape == (4,)

    def test_metacog_fields_present(self, controller):
        """New metacognition fields should be produced."""
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert cs.text_gate is not None
        assert cs.structure_gate is not None
        assert cs.uncertainty is not None
        assert cs.iteration_budget is not None
        assert cs.strategy_weights is not None

    def test_text_gate_range(self, controller):
        """text_gate should be in [0, 1] (sigmoid output)."""
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert 0.0 <= cs.text_gate.item() <= 1.0
        assert 0.0 <= cs.structure_gate.item() <= 1.0

    def test_uncertainty_range(self, controller):
        """uncertainty should be in [0, 1] (temperature-scaled sigmoid)."""
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert 0.0 <= cs.uncertainty.item() <= 1.0

    def test_strategy_weights_sum_to_one(self, controller):
        """strategy_weights should be a valid probability distribution."""
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert cs.strategy_weights.shape == (4,)
        assert abs(cs.strategy_weights.sum().item() - 1.0) < 1e-5

    def test_iteration_budget_positive(self, controller):
        """iteration_budget should be positive (softplus output)."""
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        assert cs.iteration_budget.item() > 0

    def test_task_id_changes_output(self, controller):
        """Different task_ids should produce different control signals."""
        node_embs = torch.randn(20, 32)
        cs0 = controller(node_embs, task_id=0)
        cs5 = controller(node_embs, task_id=5)
        # Task embedding differs → trunk output differs → at least one head differs
        assert not torch.allclose(cs0.text_gate, cs5.text_gate) or \
               not torch.allclose(cs0.strategy_weights, cs5.strategy_weights)

    def test_iteration_context(self, controller):
        """Providing iteration_context should not crash and should affect output."""
        node_embs = torch.randn(20, 32)
        iter_ctx = torch.tensor([0.4, 0.7, 0.01])  # iter_ratio, prev_conf, prev_delta
        cs = controller(node_embs, task_id=0, iteration_context=iter_ctx)
        assert isinstance(cs, ControlSignal)

    def test_topo_features_passthrough(self, controller):
        """topo_features should be accepted and affect output."""
        node_embs = torch.randn(20, 32)
        topo = torch.randn(6)
        cs = controller(node_embs, task_id=0, topo_features=topo)
        assert isinstance(cs, ControlSignal)

    def test_gradient_flows_through_task_embedding(self, controller):
        """Gradients should flow through the task embedding to the trunk."""
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=3)
        loss = cs.text_gate + cs.uncertainty
        loss.backward()
        assert controller.task_embedding.weight.grad is not None
        assert controller.task_embedding.weight.grad[3].abs().sum() > 0

    def test_confidence_temperature_learnable(self, controller):
        """confidence_temperature should be a learnable parameter."""
        assert controller.confidence_temperature.requires_grad
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=0)
        cs.uncertainty.backward()
        assert controller.confidence_temperature.grad is not None

    def test_no_task_id_uses_zeros(self, controller):
        """When task_id=None, should use zero task embedding."""
        node_embs = torch.randn(20, 32)
        cs = controller(node_embs, task_id=None)
        assert isinstance(cs, ControlSignal)
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_gnn_executive/test_metacog_fields.py::TestMetaCognitiveController -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'src.gnn_executive.metacognitive_controller'`

**Step 3: Implement MetaCognitiveController**

Create `src/gnn_executive/metacognitive_controller.py`:

```python
"""MetaCognitive Controller: task-conditional gating, confidence, and strategy selection.

Extends ControlHead with:
  - Task embedding input (nn.Embedding) so the controller knows WHAT task is running
  - Iteration context input (iter_ratio, prev_confidence, prev_delta)
  - text_gate / structure_gate: per-task feature gating
  - uncertainty: temperature-scaled calibrated uncertainty
  - iteration_budget: soft learned iteration limit
  - strategy_weights: softmax over reasoning strategies
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.gnn_executive.control_head import ControlSignal


class MetaCognitiveController(nn.Module):
    """Produces ControlSignal with metacognitive fields from node embeddings + task identity.

    Compared to ControlHead, adds:
      - task_embedding: nn.Embedding(num_tasks, task_embed_dim) — task identity
      - iteration_context: (3,) — [iter/max_iter, prev_confidence, prev_delta]
      - 5 new heads: text_gate, structure_gate, uncertainty, iteration_budget, strategy_weights
      - Learned confidence_temperature for calibrated uncertainty
    """

    NUM_STRATEGIES = 4  # spectral_dominant, spatial_dominant, balanced, text_dominant

    def __init__(self, embedding_dim: int, num_freqs: int, num_filters: int = 0,
                 num_tasks: int = 19, task_embed_dim: int = 128,
                 use_topo_feedback: bool = False,
                 use_embedding_topo_feedback: bool = False):
        super().__init__()
        self.num_freqs = num_freqs
        self.num_filters = num_filters
        self.num_tasks = num_tasks
        self.task_embed_dim = task_embed_dim

        # Task embedding: separate from GraphFormerEncoder's embedding
        self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)

        # Topo feedback dimension (same logic as ControlHead)
        if use_embedding_topo_feedback and use_topo_feedback:
            topo_dim = 6
        elif use_topo_feedback:
            topo_dim = 3
        else:
            topo_dim = 0
        self._topo_dim = topo_dim

        # Trunk input: mean_pool(32) + harmonic(1) + logN(1) + topo(0-6) + task(128) + iter_ctx(3)
        trunk_input_dim = embedding_dim + 2 + topo_dim + task_embed_dim + 3
        self._trunk_input_dim = trunk_input_dim

        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        trunk_out_dim = 128

        # === Existing heads (same as ControlHead) ===
        self.freq_head = nn.Linear(trunk_out_dim, num_freqs)
        self.spatial_head = nn.Linear(trunk_out_dim, 1)
        self.confidence_head = nn.Linear(trunk_out_dim, 1)
        self.time_head = nn.Linear(trunk_out_dim, 1)
        self.damping_head = nn.Linear(trunk_out_dim, 1)
        self.semantic_weight_head = nn.Linear(trunk_out_dim, 1)
        nn.init.constant_(self.semantic_weight_head.bias, -3.0)

        if num_filters > 0:
            self.filter_weights_head = nn.Linear(trunk_out_dim, num_filters)

        # === NEW metacognition heads ===

        # Layer 1: Task-conditional gating
        self.text_gate_head = nn.Linear(trunk_out_dim, 1)
        self.structure_gate_head = nn.Linear(trunk_out_dim, 1)
        # Init structure_gate bias high (structural tasks dominate early training)
        nn.init.constant_(self.structure_gate_head.bias, 2.0)

        # Layer 2: Confidence-aware reasoning
        self.uncertainty_head = nn.Linear(trunk_out_dim, 1)
        self.iteration_budget_head = nn.Linear(trunk_out_dim, 1)
        self.confidence_temperature = nn.Parameter(torch.tensor(1.5))

        # Layer 3: Strategy selection
        self.strategy_head = nn.Linear(trunk_out_dim, self.NUM_STRATEGIES)

    def forward(self, node_embeddings: torch.Tensor,
                task_id: int | None = None,
                harmonic_energy: torch.Tensor | None = None,
                topo_features: torch.Tensor | None = None,
                iteration_context: torch.Tensor | None = None) -> ControlSignal:
        """Produce control signals with metacognition from node embeddings + task identity.

        Args:
            node_embeddings: (N, embedding_dim) fused GNN output.
            task_id: integer task index (0-18) or None for zero embedding.
            harmonic_energy: scalar tensor, optional harmonic component energy.
            topo_features: (3,) or (6,) topological feedback features.
            iteration_context: (3,) tensor [iter_ratio, prev_confidence, prev_delta].

        Returns:
            ControlSignal with all original + metacognition fields.
        """
        dev = node_embeddings.device
        dtype = node_embeddings.dtype

        # Mean pool over nodes
        pooled = node_embeddings.mean(dim=0)  # (embedding_dim,)

        # Harmonic energy
        if harmonic_energy is None:
            harmonic_energy = torch.zeros(1, device=dev, dtype=dtype)
        else:
            harmonic_energy = harmonic_energy.reshape(1)

        # log(N) size feature
        n_nodes = node_embeddings.shape[0]
        log_n = torch.tensor(
            [math.log(max(n_nodes, 1)) / math.log(100)],
            device=dev, dtype=dtype,
        )

        # Topo features
        if self._topo_dim > 0:
            if topo_features is not None:
                topo_features = topo_features.to(dev, dtype)
                if topo_features.dim() == 0:
                    topo_features = topo_features.unsqueeze(0)
                topo_feat = topo_features[:self._topo_dim]
            else:
                topo_feat = torch.zeros(self._topo_dim, device=dev, dtype=dtype)
        else:
            topo_feat = torch.zeros(0, device=dev, dtype=dtype)

        # Task embedding
        if task_id is not None:
            task_idx = torch.tensor(task_id, device=dev, dtype=torch.long)
            task_emb = self.task_embedding(task_idx)  # (task_embed_dim,)
        else:
            task_emb = torch.zeros(self.task_embed_dim, device=dev, dtype=dtype)

        # Iteration context
        if iteration_context is None:
            iter_ctx = torch.zeros(3, device=dev, dtype=dtype)
        else:
            iter_ctx = iteration_context.to(dev, dtype)

        # Build trunk input
        trunk_input = torch.cat([pooled, harmonic_energy, log_n,
                                  topo_feat, task_emb, iter_ctx])

        features = self.trunk(trunk_input)  # (128,)

        # === Existing heads ===
        frequency_gate = torch.sigmoid(self.freq_head(features))
        spatial_focus = torch.sigmoid(self.spatial_head(node_embeddings).squeeze(-1))
        confidence_weights = torch.sigmoid(self.confidence_head(node_embeddings).squeeze(-1))
        diffusion_time = F.softplus(self.time_head(features).squeeze())
        wave_damping = F.softplus(self.damping_head(features).squeeze())
        semantic_weight = torch.sigmoid(self.semantic_weight_head(features).squeeze())

        filter_weights = None
        if self.num_filters > 0:
            filter_weights = torch.softmax(self.filter_weights_head(features), dim=-1)

        # === NEW metacognition heads ===

        # Layer 1: Task-conditional gating
        text_gate = torch.sigmoid(self.text_gate_head(features).squeeze())
        structure_gate = torch.sigmoid(self.structure_gate_head(features).squeeze())

        # Layer 2: Confidence — temperature-scaled sigmoid
        raw_uncertainty = self.uncertainty_head(features).squeeze()
        temperature = self.confidence_temperature.clamp(min=0.1)
        uncertainty = torch.sigmoid(raw_uncertainty / temperature)

        # Layer 2: Iteration budget — positive via softplus
        iteration_budget = F.softplus(self.iteration_budget_head(features).squeeze())

        # Layer 3: Strategy selection — softmax over 4 strategies
        strategy_weights = torch.softmax(self.strategy_head(features), dim=-1)

        return ControlSignal(
            frequency_gate=frequency_gate,
            spatial_focus=spatial_focus,
            confidence_weights=confidence_weights,
            diffusion_time=diffusion_time,
            wave_damping=wave_damping,
            semantic_weight=semantic_weight,
            filter_weights=filter_weights,
            text_gate=text_gate,
            structure_gate=structure_gate,
            uncertainty=uncertainty,
            iteration_budget=iteration_budget,
            strategy_weights=strategy_weights,
        )
```

**Step 4: Run tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_gnn_executive/test_metacog_fields.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/gnn_executive/metacognitive_controller.py tests/test_gnn_executive/test_metacog_fields.py
git commit -m "feat: MetaCognitiveController with task embedding + metacognition heads"
```

---

### Task 3: Upgrade MultiHeadClassifier to use MLP TaskHeads

**Files:**
- Modify: `src/training/multi_head_classifier.py`
- Test: `tests/test_training/test_multi_head_classifier.py` (existing + new)

**Step 1: Write failing tests**

Create `tests/test_training/test_task_head_mlp.py`:

```python
"""Tests for MLP-based TaskHead in MultiHeadClassifier."""
import torch
from src.training.multi_head_classifier import MultiHeadClassifier


def test_task_head_is_mlp():
    """Each task head should be an MLP (Sequential), not a single Linear."""
    mc = MultiHeadClassifier(input_dim=233, task_classes={'bfs': 16, 'kg_relation': 10})
    # Should have multiple layers (not just one Linear)
    head = mc.heads['bfs']
    # MLP has at least 2 Linear layers
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) >= 2, f"Expected MLP with >=2 Linear layers, got {len(linears)}"


def test_task_head_hidden_dim():
    """MLP hidden dim should default to 128."""
    mc = MultiHeadClassifier(input_dim=233, task_classes={'bfs': 16})
    head = mc.heads['bfs']
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert linears[0].out_features == 128  # first hidden layer


def test_task_head_output_shape():
    """MLP output should match n_classes."""
    mc = MultiHeadClassifier(input_dim=233, task_classes={'bfs': 16, 'kg_relation': 10})
    x = torch.randn(1, 233)
    out_bfs = mc(x, 'bfs')
    out_kg = mc(x, 'kg_relation')
    assert out_bfs.shape == (1, 16)
    assert out_kg.shape == (1, 10)


def test_add_task_creates_mlp():
    """Dynamically added tasks should also use MLP heads."""
    mc = MultiHeadClassifier(input_dim=233, task_classes={'bfs': 16})
    mc.add_task('kg_concept', 9)
    head = mc.heads['kg_concept']
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) >= 2


def test_task_head_has_dropout():
    """MLP head should include dropout for regularization."""
    mc = MultiHeadClassifier(input_dim=233, task_classes={'bfs': 16})
    head = mc.heads['bfs']
    dropouts = [m for m in head.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(dropouts) >= 1


def test_gradient_flows_through_mlp():
    """Gradients should flow through MLP hidden layers."""
    mc = MultiHeadClassifier(input_dim=233, task_classes={'bfs': 16})
    x = torch.randn(1, 233, requires_grad=True)
    out = mc(x, 'bfs')
    out.sum().backward()
    assert x.grad is not None
    # Check internal Linear layers have gradients
    linears = [m for m in mc.heads['bfs'].modules() if isinstance(m, torch.nn.Linear)]
    for lin in linears:
        assert lin.weight.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_training/test_task_head_mlp.py -v`

Expected: FAIL — head is a bare `nn.Linear`, not an MLP Sequential

**Step 3: Implement MLP TaskHead**

Replace the contents of `src/training/multi_head_classifier.py`:

```python
"""Persistent per-task classifier heads (MLP-based).

Each task head is a small MLP: Linear→GELU→Dropout→Linear→GELU→Dropout→Linear.
Provides enough capacity to learn text×structure interactions (vs single Linear).
"""
import torch.nn as nn


def _make_task_head(input_dim: int, n_classes: int,
                    hidden_dim: int = 128, dropout: float = 0.2) -> nn.Sequential:
    """Create an MLP classifier head for one task."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, n_classes),
    )


class MultiHeadClassifier(nn.Module):
    """Dictionary of MLP classifier heads keyed by task name.

    Each head is a 3-layer MLP. Heads persist across task switches —
    no weight destruction.
    """

    def __init__(self, input_dim: int, task_classes: dict[str, int],
                 hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.heads = nn.ModuleDict({
            task: _make_task_head(input_dim, n_classes, hidden_dim, dropout)
            for task, n_classes in task_classes.items()
        })

    def forward(self, x, task: str):
        return self.heads[task](x)

    def add_task(self, task: str, n_classes: int):
        """Add a new task head (for tasks discovered at runtime)."""
        self.heads[task] = _make_task_head(
            self.input_dim, n_classes, self.hidden_dim, self.dropout,
        )
```

**Step 4: Run tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_training/test_task_head_mlp.py tests/test_training/test_multi_head_classifier.py -v`

Expected: ALL PASS (new tests + check existing tests still pass)

**Step 5: Commit**

```bash
git add src/training/multi_head_classifier.py tests/test_training/test_task_head_mlp.py
git commit -m "feat: upgrade MultiHeadClassifier to MLP TaskHeads (128→64→n_classes)"
```

---

### Task 4: Wire MetaCognitiveController into GNNExecutive

**Files:**
- Modify: `src/gnn_executive/executive.py:10-115`
- Modify: `src/reasoning_loop/executive_loop.py:38-110`
- Test: existing `tests/test_gnn_executive/test_executive.py` + new integration test

**Step 1: Write failing test**

Create `tests/test_gnn_executive/test_metacog_integration.py`:

```python
"""Integration test: MetaCognitiveController wired into GNNExecutive."""
import torch
from src.gnn_executive.executive import GNNExecutive
from src.gnn_executive.control_head import ControlSignal


def test_gnn_executive_metacog_mode():
    """GNNExecutive with use_metacog=True should return metacog control signals."""
    exec = GNNExecutive(
        embedding_dim=32, hidden_dim=64,
        num_spatial_layers=2, num_spectral_layers=2,
        max_freqs=8, produce_control_signals=True,
        num_filters=4, use_topo_feedback=True,
        use_embedding_topo_feedback=True,
        use_metacog=True, num_tasks=19,
    )
    # Build a simple cell complex
    from src.cell_complex.cell_complex import CellComplex
    cc = CellComplex(32)
    for i in range(10):
        cc.add_0_cell(torch.randn(32), f"node_{i}")
    for i in range(9):
        cc.add_1_cell(i, i + 1, torch.randn(32), "edge")

    fused, edge_out, control = exec.forward_with_control(
        cc, task_id=3,
    )
    assert fused.shape == (10, 32)
    assert control.text_gate is not None
    assert control.strategy_weights is not None
    assert control.strategy_weights.shape == (4,)


def test_gnn_executive_metacog_off_by_default():
    """GNNExecutive without use_metacog should NOT produce metacog fields."""
    exec = GNNExecutive(
        embedding_dim=32, hidden_dim=64,
        num_spatial_layers=2, num_spectral_layers=2,
        max_freqs=8, produce_control_signals=True,
    )
    from src.cell_complex.cell_complex import CellComplex
    cc = CellComplex(32)
    for i in range(10):
        cc.add_0_cell(torch.randn(32), f"node_{i}")
    for i in range(9):
        cc.add_1_cell(i, i + 1, torch.randn(32), "edge")

    fused, edge_out, control = exec.forward_with_control(cc)
    assert control.text_gate is None  # no metacog
    assert control.strategy_weights is None
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_gnn_executive/test_metacog_integration.py -v`

Expected: FAIL — `GNNExecutive.__init__() got unexpected keyword argument 'use_metacog'`

**Step 3: Implement — modify GNNExecutive**

In `src/gnn_executive/executive.py`, add `use_metacog` and `num_tasks` params to `__init__`:

```python
# In GNNExecutive.__init__, add parameters:
#   use_metacog: bool = False,
#   num_tasks: int = 19,

# After existing ControlHead creation block, add:
        self.use_metacog = use_metacog
        if produce_control_signals:
            if use_metacog:
                from src.gnn_executive.metacognitive_controller import MetaCognitiveController
                self.control_head = MetaCognitiveController(
                    embedding_dim=embedding_dim,
                    num_freqs=max_freqs,
                    num_filters=num_filters,
                    num_tasks=num_tasks,
                    use_topo_feedback=use_topo_feedback,
                    use_embedding_topo_feedback=use_embedding_topo_feedback,
                )
            else:
                self.control_head = ControlHead(
                    embedding_dim=embedding_dim,
                    num_freqs=max_freqs,
                    num_filters=num_filters,
                    use_topo_feedback=use_topo_feedback,
                    use_embedding_topo_feedback=use_embedding_topo_feedback,
                )
```

In `forward_with_control()`, pass `task_id` and `iteration_context` through when metacog is active:

```python
    def forward_with_control(
        self, cc: CellComplex, harmonic_energy: torch.Tensor | None = None,
        topo_features: torch.Tensor | None = None,
        task_id: int | None = None,
        iteration_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, ControlSignal]:
        if not self.produce_control_signals:
            raise RuntimeError("forward_with_control() requires produce_control_signals=True")
        fused, edge_out = self.forward(cc)
        if self.use_metacog:
            control = self.control_head(
                fused, task_id=task_id,
                harmonic_energy=harmonic_energy,
                topo_features=topo_features,
                iteration_context=iteration_context,
            )
        else:
            control = self.control_head(
                fused, harmonic_energy=harmonic_energy,
                topo_features=topo_features,
            )
        return fused, edge_out, control
```

**Step 4: Run tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_gnn_executive/ -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/gnn_executive/executive.py tests/test_gnn_executive/test_metacog_integration.py
git commit -m "feat: wire MetaCognitiveController into GNNExecutive via use_metacog flag"
```

---

### Task 5: Thread task_id through ExecutiveReasoningLoop + add iteration budget exit

**Files:**
- Modify: `src/reasoning_loop/executive_loop.py:169-285` (forward method)
- Modify: `src/reasoning_loop/executive_loop.py:287-391` (forward_batched method)

**Step 1: Write failing test**

Create `tests/test_reasoning_loop/test_metacog_loop.py`:

```python
"""Tests for metacognition integration in ExecutiveReasoningLoop."""
import torch
from src.cell_complex.cell_complex import CellComplex
from src.reasoning_loop.executive_loop import ExecutiveReasoningLoop


def _make_cc(n=10, dim=32):
    cc = CellComplex(dim)
    for i in range(n):
        cc.add_0_cell(torch.randn(dim), f"node_{i}")
    for i in range(n - 1):
        cc.add_1_cell(i, i + 1, torch.randn(dim), "edge")
    return cc


def test_loop_accepts_task_id():
    """ExecutiveReasoningLoop.forward() should accept task_id kwarg."""
    loop = ExecutiveReasoningLoop(
        embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
        gnn_spectral_layers=2, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_iterations=2, use_wave_dynamics=False,
        use_metacog=True, num_tasks=19,
    )
    cc = _make_cc()
    output, num_iters, diagnostics = loop(cc, task_id=5)
    assert output.shape == (10, 32)
    # Control signals should have metacog fields
    cs = diagnostics['control_signals'][-1]
    assert cs.text_gate is not None


def test_loop_metacog_off_compat():
    """Without use_metacog, loop should work exactly as before (no task_id)."""
    loop = ExecutiveReasoningLoop(
        embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
        gnn_spectral_layers=2, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_iterations=2, use_wave_dynamics=False,
    )
    cc = _make_cc()
    output, num_iters, diagnostics = loop(cc)
    assert output.shape == (10, 32)
    cs = diagnostics['control_signals'][-1]
    assert cs.text_gate is None  # no metacog
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_reasoning_loop/test_metacog_loop.py -v`

Expected: FAIL — `ExecutiveReasoningLoop.__init__() got unexpected keyword argument 'use_metacog'`

**Step 3: Implement**

In `src/reasoning_loop/executive_loop.py`:

1. Add `use_metacog=False` and `num_tasks=19` to `__init__`. Pass them through to `GNNExecutive`:
```python
        self.gnn_executive = GNNExecutive(
            # ... existing params ...
            use_metacog=use_metacog,
            num_tasks=num_tasks,
        )
```

2. Add `task_id: int | None = None` param to `forward()` signature (line ~169). Thread it into `gnn_executive.forward_with_control()`:
```python
    def forward(self, cc: CellComplex,
                topo_features: torch.Tensor | None = None,
                task_id: int | None = None,
                ) -> tuple[torch.Tensor, int, dict]:
```

And in the loop body (~line 213):
```python
            # Build iteration context for metacog controller
            iteration_context = None
            if getattr(self.gnn_executive, 'use_metacog', False):
                prev_conf = diagnostics['control_signals'][-1].uncertainty.item() if diagnostics['control_signals'] else 0.5
                prev_delta = diagnostics['convergence_deltas'][-1] if diagnostics['convergence_deltas'] else 0.0
                iteration_context = torch.tensor(
                    [i / self.max_iterations, prev_conf, prev_delta],
                    device=cc.device, dtype=torch.float32,
                )

            gnn_out, edge_out, control = self.gnn_executive.forward_with_control(
                cc, harmonic_energy=harmonic_energy_input,
                topo_features=topo_features,
                task_id=task_id,
                iteration_context=iteration_context,
            )
```

3. Add soft iteration budget exit after the harmonic convergence check (~line 275):
```python
            if delta < self.convergence_threshold:
                break

            # Soft learned iteration budget (metacog)
            if (control.iteration_budget is not None
                    and control.uncertainty is not None
                    and i > 0):
                budget_ratio = float(i) / max(float(control.iteration_budget.item()), 1.0)
                if budget_ratio > 1.5 and control.uncertainty.item() < 0.3:
                    break
```

4. In `forward_batched()`, add `task_id` param and thread it the same way. For batched, `task_id` is a single int (all graphs in a batch are the same task):
```python
    def forward_batched(
        self, ccs: list[CellComplex],
        topo_features_list: list[torch.Tensor] | None = None,
        task_id: int | None = None,
    ) -> list[tuple[torch.Tensor, int, dict]]:
```

And in the per-graph GNN call (~line 328):
```python
                gnn_out, edge_out, control = self.gnn_executive.forward_with_control(
                    ccs[g], harmonic_energy=harmonic_energy,
                    topo_features=topo_feat,
                    task_id=task_id,
                )
```

**Step 4: Run tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_reasoning_loop/ -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/reasoning_loop/executive_loop.py src/gnn_executive/executive.py tests/test_reasoning_loop/test_metacog_loop.py
git commit -m "feat: thread task_id through executive loop + iteration budget soft exit"
```

---

### Task 6: Gated feature composition in HierarchicalMultiHopModel

**Files:**
- Modify: `src/benchmarks/run_comparison.py:73-301`
- Test: existing integration tests + new gating test

**Step 1: Write failing test**

Create `tests/test_benchmarks/test_metacog_gating.py`:

```python
"""Tests for gated feature composition in HierarchicalMultiHopModel."""
import torch
from src.benchmarks.run_comparison import HierarchicalMultiHopModel
from src.cell_complex.cell_complex import CellComplex


def _make_model(use_metacog=True, use_llm=True, backend='qwen'):
    return HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
        gnn_spectral_layers=2, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=16, max_iterations=2, convergence_threshold=0.05,
        use_wave_dynamics=False, use_higher_order=False,
        use_llm=use_llm,
        llm_config={'backend': backend, 'llm_dim': 64, 'use_mock': True} if use_llm else None,
        use_multi_head_classifier=True,
        use_metacog=use_metacog,
    )


def _make_cc(n=10, dim=32, with_texts=True):
    cc = CellComplex(dim)
    for i in range(n):
        cc.add_0_cell(torch.randn(dim), f"concept_{i}")
    for i in range(n - 1):
        cc.add_1_cell(i, i + 1, torch.randn(dim), "edge")
    if with_texts:
        cc.node_texts = [f"concept_{i}" for i in range(n)]
    return cc


def test_classifier_input_dim_with_metacog():
    """Classifier input should be 233 with metacog + text features.

    96 (gated structural) + 36 (topological) + 96 (gated text) + 4 (strategy) + 1 (uncertainty) = 233
    """
    model = _make_model(use_metacog=True, use_llm=True, backend='qwen')
    assert model.classifier_input_dim == 233


def test_classifier_input_dim_without_metacog():
    """Without metacog, classifier input should be 228 (original Approach C)."""
    model = _make_model(use_metacog=False, use_llm=True, backend='qwen')
    assert model.classifier_input_dim == 228


def test_classifier_input_dim_no_text():
    """Without text features and without metacog, should be 132."""
    model = _make_model(use_metacog=False, use_llm=False)
    assert model.classifier_input_dim == 132


def test_classifier_input_dim_metacog_no_text():
    """With metacog but no text, should be 132 + 5 (strategy+uncertainty) = 137."""
    model = _make_model(use_metacog=True, use_llm=False)
    assert model.classifier_input_dim == 137


def test_forward_with_metacog():
    """Full forward pass with metacog should produce correct output shape."""
    model = _make_model(use_metacog=True, use_llm=True, backend='qwen')
    cc = _make_cc()
    logits = model(cc, 0, 5, task='bfs')
    assert logits.shape == (16,)  # bfs has 16 classes
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_benchmarks/test_metacog_gating.py -v`

Expected: FAIL — `HierarchicalMultiHopModel.__init__() got unexpected keyword argument 'use_metacog'`

**Step 3: Implement gated composition**

In `src/benchmarks/run_comparison.py`, modify `HierarchicalMultiHopModel`:

1. Add `use_metacog=False` param to `__init__`. Pass to executive loop:
```python
        self.use_metacog = use_metacog
        # ... in ExecutiveReasoningLoop constructor:
        use_metacog=use_metacog,
        num_tasks=lc.get('num_tasks', 19) if use_metacog else 0,
```

2. Update `classifier_input_dim` calculation (replace lines 170-176):
```python
        base_classifier_dim = 3 * embedding_dim + 3 + 1 + PERSISTENCE_FEATURES  # 132
        text_feat_dim = embedding_dim if (use_llm and backend_type == 'qwen') else 0
        self.text_feat_dim = text_feat_dim

        # Metacog adds strategy_weights(4) + uncertainty(1) to classifier input
        metacog_dim = 5 if use_metacog else 0
        self.metacog_dim = metacog_dim

        classifier_input_dim = base_classifier_dim + 3 * text_feat_dim + metacog_dim
        self.classifier_input_dim = classifier_input_dim
```

3. In `forward()`, after computing `combined`, apply gates and append metacog signals. Replace lines 284-294:
```python
        structural = torch.cat([query_emb, target_emb, diff_emb])
        topological = torch.cat([hodge_features, wave_energy, persistence_features])

        # Get last control signal for metacog gating
        last_control = diagnostics['control_signals'][-1] if diagnostics.get('control_signals') else None

        # Apply metacog gates if available
        if self.use_metacog and last_control is not None and last_control.text_gate is not None:
            gated_structural = last_control.structure_gate * structural
            gated_text_parts = []
            if text_features is not None and self.text_feat_dim > 0:
                text_q = text_features[query_node]
                text_t = text_features[target_node]
                text_combined = torch.cat([text_q, text_t, text_q - text_t])
                gated_text_parts.append(last_control.text_gate * text_combined)
            elif self.text_feat_dim > 0:
                gated_text_parts.append(torch.zeros(3 * self.text_feat_dim, device=dev))

            combined = torch.cat([
                gated_structural, topological,
                *gated_text_parts,
                last_control.strategy_weights,
                last_control.uncertainty.unsqueeze(0),
            ])
        else:
            # Non-metacog path (original)
            combined = torch.cat([structural, topological])
            if text_features is not None and self.text_feat_dim > 0:
                text_q = text_features[query_node]
                text_t = text_features[target_node]
                combined = torch.cat([combined, text_q, text_t, text_q - text_t])
            elif self.text_feat_dim > 0:
                combined = torch.cat([combined,
                                      torch.zeros(3 * self.text_feat_dim, device=dev)])
```

4. Thread `task_id` into executive loop call. The `task` parameter (string) needs mapping to integer task_id. Add a helper:
```python
        # Near top of forward(), map task name to integer ID for metacog
        task_id = None
        if self.use_metacog and task is not None:
            from src.benchmarks.benchmark_dataset import TASK_REGISTRY
            task_list = sorted(TASK_REGISTRY.keys())
            task_id = task_list.index(task) if task in task_list else None

        output, num_iters, diagnostics = self.executive_loop(
            cc, topo_features=topo_features, task_id=task_id,
        )
```

**Step 4: Run tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_benchmarks/test_metacog_gating.py tests/test_llm/test_qwen_text_features.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/benchmarks/run_comparison.py tests/test_benchmarks/test_metacog_gating.py
git commit -m "feat: gated feature composition with metacog signals in classifier input"
```

---

### Task 7: Mirror gated composition in batch_utils.py

**Files:**
- Modify: `src/training/batch_utils.py:30-168`

**Step 1: Write failing test**

Add to `tests/test_benchmarks/test_metacog_gating.py`:

```python
def test_batched_forward_with_metacog():
    """_forward_batch should handle metacog gated composition."""
    from src.training.batch_utils import _forward_batch
    model = _make_model(use_metacog=True, use_llm=True, backend='qwen')
    cc1 = _make_cc(n=10, with_texts=True)
    cc2 = _make_cc(n=12, with_texts=True)
    batch = [
        (cc1, 0, 5, 3, None),  # (cc, query, target, answer, metadata)
        (cc2, 1, 6, 7, None),
    ]
    results = _forward_batch(model, batch, torch.device('cpu'), task='bfs')
    assert len(results) == 2
    for logits, answer in results:
        assert logits.shape == (16,)  # bfs has 16 classes
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_benchmarks/test_metacog_gating.py::test_batched_forward_with_metacog -v`

Expected: FAIL — dimension mismatch (metacog changes classifier_input_dim but batch path not updated)

**Step 3: Implement**

In `src/training/batch_utils.py`, update `_forward_batch()`:

1. Thread `task_id` to `model.executive_loop.forward_batched()`:
```python
        # Map task name to integer ID for metacog
        task_id = None
        if getattr(model, 'use_metacog', False) and task is not None:
            from src.benchmarks.benchmark_dataset import TASK_REGISTRY
            task_list = sorted(TASK_REGISTRY.keys())
            task_id = task_list.index(task) if task in task_list else None

        loop_results = model.executive_loop.forward_batched(
            ccs, topo_features_list=topo_features_list,
            task_id=task_id,
        )
```

2. In the per-graph classifier section, apply the same gated composition as run_comparison.py:
```python
            # Get last control for metacog gating
            last_control = diagnostics.get('control_signals', [None])[-1]
            use_metacog = getattr(model, 'use_metacog', False)
            metacog_dim = getattr(model, 'metacog_dim', 0)

            structural = torch.cat([query_emb, target_emb, diff_emb])
            topological = torch.cat([hodge_features, wave_energy, persistence_features])

            if use_metacog and last_control is not None and last_control.text_gate is not None:
                gated_structural = last_control.structure_gate * structural
                gated_text_parts = []
                if text_features is not None and text_feat_dim > 0:
                    text_q = text_features[queries[g]]
                    text_t = text_features[targets[g]]
                    text_combined = torch.cat([text_q, text_t, text_q - text_t])
                    gated_text_parts.append(last_control.text_gate * text_combined)
                elif text_feat_dim > 0:
                    gated_text_parts.append(torch.zeros(3 * text_feat_dim, device=dev))

                combined = torch.cat([
                    gated_structural, topological,
                    *gated_text_parts,
                    last_control.strategy_weights,
                    last_control.uncertainty.unsqueeze(0),
                ])
            else:
                combined = torch.cat([structural, topological])
                if text_features is not None and text_feat_dim > 0:
                    text_q = text_features[queries[g]]
                    text_t = text_features[targets[g]]
                    combined = torch.cat([combined, text_q, text_t, text_q - text_t])
                elif text_feat_dim > 0:
                    combined = torch.cat([combined,
                                          torch.zeros(3 * text_feat_dim, device=dev)])
```

**Step 4: Run tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_benchmarks/test_metacog_gating.py tests/test_training/test_batch_utils.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/training/batch_utils.py tests/test_benchmarks/test_metacog_gating.py
git commit -m "feat: mirror gated feature composition in batched training path"
```

---

### Task 8: Auxiliary losses (calibration + efficiency + gating diversity)

**Files:**
- Create: `src/training/metacog_losses.py`
- Test: `tests/test_training/test_metacog_losses.py`

**Step 1: Write failing tests**

Create `tests/test_training/test_metacog_losses.py`:

```python
"""Tests for metacognition auxiliary losses."""
import torch
from src.gnn_executive.control_head import ControlSignal


def _make_control(text_gate=0.7, uncertainty=0.3):
    return ControlSignal(
        frequency_gate=torch.randn(8),
        spatial_focus=torch.randn(20),
        confidence_weights=torch.randn(20),
        diffusion_time=torch.tensor(1.0),
        wave_damping=torch.tensor(0.5),
        text_gate=torch.tensor(text_gate, requires_grad=True),
        structure_gate=torch.tensor(0.8, requires_grad=True),
        uncertainty=torch.tensor(uncertainty, requires_grad=True),
        iteration_budget=torch.tensor(3.0, requires_grad=True),
        strategy_weights=torch.softmax(torch.randn(4, requires_grad=True), dim=0),
    )


def test_calibration_loss():
    from src.training.metacog_losses import calibration_loss
    control = _make_control(uncertainty=0.3)
    is_correct = torch.tensor(1.0)
    temp = torch.tensor(1.5)
    loss = calibration_loss(control, is_correct, temp)
    assert loss.dim() == 0  # scalar
    assert loss.item() >= 0.0
    loss.backward()
    assert control.uncertainty.grad is not None


def test_calibration_loss_correct_direction():
    """When model is correct and confident, loss should be low."""
    from src.training.metacog_losses import calibration_loss
    temp = torch.tensor(1.5)
    # Correct and confident (low uncertainty)
    c_correct = _make_control(uncertainty=0.1)
    loss_correct = calibration_loss(c_correct, torch.tensor(1.0), temp)
    # Incorrect and confident (low uncertainty)
    c_wrong = _make_control(uncertainty=0.1)
    loss_wrong = calibration_loss(c_wrong, torch.tensor(0.0), temp)
    assert loss_correct.item() < loss_wrong.item()


def test_efficiency_loss():
    from src.training.metacog_losses import efficiency_loss
    control = _make_control(uncertainty=0.2)
    loss = efficiency_loss(control, num_iters_used=4)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_gating_diversity_loss():
    from src.training.metacog_losses import gating_diversity_loss
    controls = [_make_control(text_gate=tg) for tg in [0.1, 0.9, 0.5, 0.3]]
    loss = gating_diversity_loss(controls)
    assert loss.dim() == 0
    # Negative variance → should be negative (we minimize it to maximize variance)
    assert loss.item() <= 0.0


def test_gating_diversity_loss_uniform_is_worst():
    """If all text_gates are the same, diversity loss should be 0 (worst)."""
    from src.training.metacog_losses import gating_diversity_loss
    controls = [_make_control(text_gate=0.5) for _ in range(4)]
    loss = gating_diversity_loss(controls)
    assert abs(loss.item()) < 1e-5  # variance = 0
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_training/test_metacog_losses.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'src.training.metacog_losses'`

**Step 3: Implement**

Create `src/training/metacog_losses.py`:

```python
"""Auxiliary losses for MetaCognitive Controller training.

- calibration_loss: predicted confidence should match actual correctness
- efficiency_loss: penalize excess iterations when confident
- gating_diversity_loss: encourage different text_gate values across tasks
"""

import torch
import torch.nn.functional as F
from src.gnn_executive.control_head import ControlSignal


def calibration_loss(control: ControlSignal, is_correct: torch.Tensor,
                     confidence_temperature: torch.Tensor) -> torch.Tensor:
    """Confidence calibration: predicted confidence should match correctness.

    Args:
        control: ControlSignal with uncertainty field.
        is_correct: scalar 0.0 or 1.0.
        confidence_temperature: learned temperature parameter.

    Returns:
        Scalar calibration loss + temperature regularization.
    """
    if control.uncertainty is None:
        return torch.tensor(0.0, device=is_correct.device)

    predicted_confidence = 1.0 - control.uncertainty
    # Clamp to avoid log(0) in BCE
    predicted_confidence = predicted_confidence.clamp(1e-6, 1 - 1e-6)
    bce = F.binary_cross_entropy(predicted_confidence, is_correct)

    # Temperature regularization: prevent collapse to 0
    temp_reg = -0.01 * torch.log(confidence_temperature.clamp(min=0.1))

    return bce + temp_reg


def efficiency_loss(control: ControlSignal, num_iters_used: int) -> torch.Tensor:
    """Iteration efficiency: penalize excess iterations when confident.

    Args:
        control: ControlSignal with iteration_budget and uncertainty fields.
        num_iters_used: actual number of iterations executed.

    Returns:
        Scalar efficiency loss.
    """
    if control.iteration_budget is None or control.uncertainty is None:
        return torch.tensor(0.0)

    predicted_confidence = 1.0 - control.uncertainty
    excess = max(0.0, num_iters_used - control.iteration_budget.item())
    excess_t = torch.tensor(excess, device=control.iteration_budget.device)
    return excess_t.pow(2) * predicted_confidence


def gating_diversity_loss(controls: list[ControlSignal]) -> torch.Tensor:
    """Gating diversity: encourage variance in text_gate across tasks.

    Returns negative variance (minimize to maximize diversity).

    Args:
        controls: list of ControlSignal from different tasks in a batch.

    Returns:
        Scalar negative variance of text_gate values.
    """
    text_gates = []
    for c in controls:
        if c.text_gate is not None:
            text_gates.append(c.text_gate)

    if len(text_gates) < 2:
        return torch.tensor(0.0)

    stacked = torch.stack(text_gates)
    return -torch.var(stacked)
```

**Step 4: Run tests**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_training/test_metacog_losses.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/training/metacog_losses.py tests/test_training/test_metacog_losses.py
git commit -m "feat: auxiliary losses for metacognition (calibration, efficiency, gating diversity)"
```

---

### Task 9: Update training script — optimizer groups, Phase E, aux losses

**Files:**
- Modify: `scripts/run_dsm_curriculum.py`
- Modify: `config/v7_metacognition.yaml`

**Step 1: No TDD for script changes — verify manually**

This task modifies the training script and config. Changes are verified by running training, not unit tests.

**Step 2: Modify `_build_dsm_optimizers` to add metacog + classifier groups**

In `scripts/run_dsm_curriculum.py`, modify `_build_dsm_optimizers()` (lines 101-166):

Add two new parameter groups by extending the routing logic:

```python
    metacog_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in qwen_llm_ids:
            continue
        if 'metacog' in name.lower() or 'metacognitive' in name.lower():
            metacog_params.append(param)
        elif 'multi_head_classifier' in name or 'classifier' in name:
            classifier_params.append(param)
        elif 'dsm' in name.lower() or 'distilled' in name.lower():
            dsm_params.append(param)
        elif ('adapter' in name.lower() or 'graph_former' in name.lower()
              or 'graphformer' in name.lower()):
            dsm_params.append(param)
        elif 'topo_bridge' in name or 'bridge' in name:
            bridge_params.append(param)
        else:
            gnn_tat_params.append(param)

    main_groups = [
        {'params': gnn_tat_params, 'lr': tc['learning_rate']},
        {'params': dsm_params, 'lr': semantic_lr},
    ]
    if metacog_params:
        metacog_lr = tc.get('metacog_learning_rate', 5e-4)
        main_groups.append({'params': metacog_params, 'lr': metacog_lr})
    if classifier_params:
        classifier_lr = tc.get('classifier_learning_rate', tc['learning_rate'])
        main_groups.append({'params': classifier_params, 'lr': classifier_lr})
    if qwen_llm_params:
        qwen_lr = tc.get('qwen_learning_rate', 1e-5)
        main_groups.append({'params': qwen_llm_params, 'lr': qwen_lr})
```

**Step 3: Remove rebalancing for KG tasks**

In `_run_phase()` (~line 612), comment out or remove the rebalancing call:

```python
        # KG datasets: use focal loss + class_weights instead of rebalancing
        # (rebalancing was too aggressive: 5000 → 738 samples)
        # if task.startswith("kg_"):
        #     _rebalance_dataset(train_ds)
```

**Step 4: Add Phase E support**

Add at module level:
```python
PHASE_E_TASKS = sorted(PHASE_A_TASKS + PHASE_B_TASKS + PHASE_C_TASKS + PHASE_D_TASKS)
```

In the main function where phases are dispatched, add Phase E after Phase D:

```python
    # Phase E: Metacognitive Consolidation (mixed tasks)
    if resume_phase in (None, 'a', 'b', 'c', 'd', 'e'):
        if resume_phase != 'e':
            # ... load Phase D checkpoint ...
            pass
        print("\n=== Phase E: Metacognitive Consolidation ===")
        # ... run Phase E with all 19 tasks mixed ...
```

**Step 5: Add aux losses to training loop**

In `train_epoch_batched()` or inline in `_run_phase()`, add after the main loss computation:

```python
        # Metacognition auxiliary losses
        if getattr(model, 'use_metacog', False) and hasattr(model, 'executive_loop'):
            from src.training.metacog_losses import calibration_loss, efficiency_loss
            # Get last control signal from diagnostics
            # (stored in model._last_control after forward pass)
            control = getattr(model, '_last_control', None)
            if control is not None and control.uncertainty is not None:
                # Calibration loss
                is_correct_t = torch.tensor(
                    float(logits.argmax().item() == answer),
                    device=device,
                )
                metacog_controller = model.executive_loop.gnn_executive.control_head
                cal_loss = calibration_loss(
                    control, is_correct_t,
                    metacog_controller.confidence_temperature,
                )
                batch_loss = batch_loss + 0.1 * cal_loss
```

**Step 6: Update config**

In `config/v7_metacognition.yaml`, add:

```yaml
model:
  use_metacog: true    # Enable MetaCognitiveController

training:
  metacog_learning_rate: 0.0005
  classifier_learning_rate: 0.001
  # bridge_learning_rate stays at 0.0001 BUT text_extractor.proj
  # is now captured by bridge group and gets 0.001 via separate group

  # Phase E
  epochs_phase_e: 30

phase_e:
  tasks: all  # all 19 tasks, uniform sampling
  calibration_loss_weight: 0.1
  efficiency_loss_weight: 0.01
  gating_diversity_loss_weight: 0.05
```

**Step 7: Commit**

```bash
git add scripts/run_dsm_curriculum.py config/v7_metacognition.yaml
git commit -m "feat: Phase E + metacog optimizer groups + aux losses + remove rebalancing"
```

---

### Task 10: Fix node_texts bug on structural replay samples

**Files:**
- Modify: `src/benchmarks/run_comparison.py` (defensive getattr already present)
- Modify: `src/training/batch_utils.py` (defensive getattr already present)
- Verify: `src/cell_complex/cell_complex.py` (node_texts already in `__init__` and `clone()`)

**Step 1: Write test to confirm the bug is reproducible**

Create `tests/test_training/test_node_texts_replay.py`:

```python
"""Test that structural samples without node_texts don't crash during replay."""
import torch
from src.cell_complex.cell_complex import CellComplex
from src.benchmarks.run_comparison import HierarchicalMultiHopModel


def test_structural_sample_no_crash():
    """Forward pass on a structural sample (empty node_texts) should not crash."""
    model = HierarchicalMultiHopModel(
        embedding_dim=32, gnn_hidden=64, gnn_spatial_layers=2,
        gnn_spectral_layers=2, max_freqs=8, tat_layers=1,
        tat_spatial_heads=2, tat_spectral_heads=2, tat_ff_dim=64,
        max_classes=16, max_iterations=2, convergence_threshold=0.05,
        use_wave_dynamics=False, use_higher_order=False,
        use_llm=True,
        llm_config={'backend': 'qwen', 'llm_dim': 64, 'use_mock': True},
        use_multi_head_classifier=True,
        use_metacog=True,
    )
    # Structural sample: node_texts is empty list (not None, not missing)
    cc = CellComplex(32)
    for i in range(10):
        cc.add_0_cell(torch.randn(32), f"node_{i}")
    for i in range(9):
        cc.add_1_cell(i, i + 1, torch.randn(32), "edge")
    # node_texts initialized as [] by CellComplex.__init__

    # Should NOT crash — text_features will be None, zero-padded
    logits = model(cc, 0, 5, task='bfs')
    assert logits.shape == (16,)
```

**Step 2: Run test**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/test_training/test_node_texts_replay.py -v`

Expected: PASS (if the `getattr(cc, 'node_texts', None) or None` pattern works with empty lists)

If it crashes, fix by ensuring empty list → None in the check:
```python
node_texts = getattr(cc, 'node_texts', None)
if not node_texts:  # None or empty list
    node_texts = None
```

**Step 3: Commit**

```bash
git add tests/test_training/test_node_texts_replay.py
git commit -m "test: verify structural replay samples don't crash with empty node_texts"
```

---

### Task 11: Full integration test + run existing test suite

**Files:**
- Test: run full test suite

**Step 1: Run full test suite**

Run: `cd /mnt/c/Users/gspea/source/repos/transformer-topology && python -m pytest tests/ -x -k "not test_spectral_gap_variable_sizes and not test_gradient_reaches_gnn_and_tat and not test_missing_class_gets_max_weight" --timeout=120 -q`

Expected: All tests pass (existing + new). Known flaky tests excluded.

**Step 2: Fix any failures**

If failures occur, fix them before proceeding. Common issues:
- Dimension mismatches from classifier_input_dim changes
- Import errors from new modules
- `_load_state_filtered()` not handling new param keys

**Step 3: Final commit**

```bash
git add -A
git commit -m "test: full integration — all tests pass with MetaCognitive Controller"
```

---

### Task 12: Build model config and dry-run smoke test

**Step 1: Smoke test model construction with metacog enabled**

Run:
```bash
cd /mnt/c/Users/gspea/source/repos/transformer-topology
python -c "
from src.benchmarks.run_benchmark_suite import _build_model
import yaml
with open('config/v7_metacognition.yaml') as f:
    config = yaml.safe_load(f)
config['llm']['use_mock'] = True  # mock mode for local testing
model = _build_model(config)
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
print(f'Classifier input dim: {model.classifier_input_dim}')
print(f'Text feat dim: {model.text_feat_dim}')
print(f'Metacog dim: {model.metacog_dim}')
print(f'Use metacog: {model.use_metacog}')
# Check metacog controller is wired
ctrl = model.executive_loop.gnn_executive.control_head
print(f'Controller type: {type(ctrl).__name__}')
print(f'Task embedding: {ctrl.task_embedding.weight.shape}')
print(f'Confidence temp: {ctrl.confidence_temperature.item():.2f}')
print('SUCCESS')
"
```

Expected: SUCCESS with MetaCognitiveController type, (19, 128) task embedding, temp=1.50

**Step 2: Commit any final fixes**

```bash
git add -A
git commit -m "chore: smoke test passes — MetaCognitive Controller fully wired"
```

---

## Dependency Graph

```
Task 1 (ControlSignal fields)
  └─→ Task 2 (MetaCognitiveController)
       └─→ Task 4 (Wire into GNNExecutive)
            └─→ Task 5 (Thread through ExecutiveLoop)
                 ├─→ Task 6 (Gated composition in run_comparison)
                 │    └─→ Task 7 (Mirror in batch_utils)
                 └─→ Task 8 (Auxiliary losses)
                      └─→ Task 9 (Training script + config)
                           └─→ Task 10 (Bug fix)
                                └─→ Task 11 (Full test suite)
                                     └─→ Task 12 (Smoke test)

Task 3 (MLP TaskHeads) — independent, can run in parallel with Tasks 2-5
```

## Estimated Effort

- Tasks 1-3: Foundation (~30 min)
- Tasks 4-7: Wiring (~45 min)
- Tasks 8-9: Training infra (~30 min)
- Tasks 10-12: Verification (~15 min)

**Total: ~2 hours implementation**

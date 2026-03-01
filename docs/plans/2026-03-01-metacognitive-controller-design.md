# MetaCognitive Controller Design

**Date**: 2026-03-01
**Phase**: 7+ (extends Phase 7 KG tasks with metacognitive architecture)
**Branch**: phase7-computation-graph-topology

## Problem Statement

Phase D training (5 KG tasks with Approach C text features) showed proof-of-life but poor absolute performance:

| Task | Classes | Best Bal Acc | Random | Lift |
|------|---------|-------------|--------|------|
| kg_relation | 10 | 24.2% | 10% | +14.2pp |
| kg_concept | 9 | 19.3% | 11% | +8.3pp |
| kg_pathvalid | 2 | 52.0% | 50% | +2pp |
| kg_analogy | 3 | 38.9% | 33% | +5.9pp |
| kg_cluster | 6 | 36.6% | 17% | +19.6pp |

Root causes:
1. **Task identity doesn't reach ControlHead** — model can't learn task-conditional behavior
2. **Classifier too shallow** — single Linear(228, N) can't learn text-structure interactions
3. **text_extractor.proj LR too slow** — 0.0001 when 0.001 is needed
4. **Data starvation** — rebalancing cuts 5000 to 738 samples
5. **No metacognitive signals** — no gating, confidence, or strategy selection

## Solution: MetaCognitive Controller (Approach A)

Three layered capabilities, built incrementally:
1. **Task-conditional gating**: Model learns per-task text vs structure weighting
2. **Confidence-aware reasoning**: Calibrated uncertainty, learned iteration budget
3. **Strategy selection**: Softmax weights over reasoning strategies

## Architecture

### MetaCognitiveController

Extends ControlHead with task conditioning and metacognition heads.

```
Inputs:
  node_embeddings (N, 32)     — mean-pooled to (32,)
  harmonic_energy (1,)         — convergence signal
  log(N) (1,)                  — graph size normalization
  topo_features (6,)           — topological feedback
  task_embedding (128,)        — NEW: from learned nn.Embedding(19, 128)
  iteration_context (3,)       — NEW: [iter/max_iter, prev_confidence, prev_delta]

Trunk (shared MLP):
  Input dim: 32 + 1 + 1 + 6 + 128 + 3 = 171
  Linear(171, 256) + GELU + Linear(256, 128)

Existing heads (unchanged):
  frequency_gate:     Linear(128, num_freqs) + sigmoid
  spatial_focus:      Linear(128, 1) + sigmoid (per-node)
  confidence_weights: Linear(128, 1) + sigmoid (per-node)
  diffusion_time:     Linear(128, 1) + softplus
  wave_damping:       Linear(128, 1) + softplus
  semantic_weight:    Linear(128, 1) + sigmoid [bias=-3.0]
  filter_weights:     Linear(128, num_filters) + softmax

New metacognition heads:
  Layer 1 — Task-Conditional Gating:
    text_gate:       Linear(128, 1) + sigmoid → scalar [0,1]
    structure_gate:  Linear(128, 1) + sigmoid → scalar [0,1]

  Layer 2 — Confidence-Aware Reasoning:
    uncertainty:     Linear(128, 1) → raw logit, then temperature-scaled sigmoid
    iteration_budget: Linear(128, 1) + softplus → scalar > 0
    confidence_temperature: nn.Parameter(1.5) — learned, clamped ≥ 0.1

  Layer 3 — Strategy Selection:
    strategy_weights: Linear(128, 4) + softmax
      [spectral_dominant, spatial_dominant, balanced, text_dominant]
```

Parameter cost: ~25K new params. Negligible vs 214M model.

Task embedding: separate `nn.Embedding(19, 128)` owned by controller, NOT shared with GraphFormerEncoder's `nn.Embedding(19, 2048)`.

### ControlSignal Extensions

```python
@dataclass
class ControlSignal:
    # Existing fields
    frequency_gate: Tensor
    spatial_focus: Tensor
    confidence_weights: Tensor
    diffusion_time: Tensor
    wave_damping: Tensor
    semantic_weight: Tensor | None = None
    filter_weights: Tensor | None = None

    # NEW metacognition fields
    text_gate: Tensor | None = None         # scalar [0,1]
    structure_gate: Tensor | None = None    # scalar [0,1]
    uncertainty: Tensor | None = None       # scalar [0,1], temperature-scaled
    iteration_budget: Tensor | None = None  # scalar > 0
    strategy_weights: Tensor | None = None  # (4,) softmax
```

## Gated Feature Composition

### Classifier Input (before)

```
combined = [query_emb(32), target_emb(32), diff_emb(32),     # 96 structural
            hodge(3), wave_energy(1), persistence(32),         # 36 topological
            text_q(32), text_t(32), text_q-text_t(32)]         # 96 text
```

### Classifier Input (after)

```python
structural = torch.cat([query_emb, target_emb, diff_emb])  # (96,)
topological = torch.cat([hodge, wave_energy, persistence])  # (36,)
text = torch.cat([text_q, text_t, text_q - text_t])         # (96,) or zeros

gated_structural = control.structure_gate * structural
gated_text = control.text_gate * text

combined = torch.cat([
    gated_structural,       # (96,) gated
    topological,            # (36,) always present
    gated_text,             # (96,) gated
    strategy_weights,       # (4,)  strategy context
    uncertainty,            # (1,)  self-assessed
])  # Total: 233
```

### Deeper Classifier (MLP)

```python
class TaskHead(nn.Module):
    def __init__(self, input_dim=233, hidden_dim=128, n_classes=10, dropout=0.2):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )
```

## Confidence Calibration

### Temperature-Scaled Uncertainty

```python
raw_logit = self.uncertainty_head(trunk_output)
temperature = self.confidence_temperature.clamp(min=0.1)
uncertainty = torch.sigmoid(raw_logit / temperature)
```

### Calibration Loss

```python
predicted_confidence = 1.0 - control.uncertainty
is_correct = (pred == answer).float()
calibration_loss = F.binary_cross_entropy(predicted_confidence, is_correct)
temp_reg = -0.01 * torch.log(controller.confidence_temperature.clamp(min=0.1))
total_calibration = 0.1 * calibration_loss + temp_reg
```

### Iteration Budget (soft exit)

```python
# In executive loop:
if t > 0 and control.iteration_budget is not None:
    budget_ratio = float(t) / max(float(control.iteration_budget), 1.0)
    if budget_ratio > 1.5 and control.uncertainty < 0.3:
        break  # confident + over budget → exit
```

### Optional Post-hoc Platt Scaling

After training, fit `sigmoid(a * raw_logit + b)` on calibration set for deployment.

## Training Curriculum

### Phase A: Structural Foundation (unchanged)
- bypass_llm = True, DSM frozen
- Tasks: diverse, bfs, hodge_class, spectral_gap, etc.
- MetaCog active: text_gate → 0, structure_gate → 1
- 30 epochs, LR 0.001

### Phase B: Semantic Introduction (unchanged)
- bypass_llm = False, DSM unfrozen
- Tasks: graph_completion, labeled_reasoning + 20% replay
- 20 epochs, LR 0.0005

### Phase C: Language Tasks (unchanged)
- Tasks: analogical_transfer + 20% replay
- 15 epochs, LR 0.0003

### Phase D: Knowledge Graph (REVISED)
- Tasks: kg_relation, kg_concept, kg_pathvalid, kg_analogy, kg_cluster
- **25k samples per task** (up from 5000)
- **20 epochs per task** (up from 10)
- Text features active (Approach C)
- Metacognition heads training
- Replay: 20% structural + 10% Phase B/C
- **No rebalancing** — use focal loss + class_weights instead

### Phase E: Metacognitive Consolidation (NEW)
- **Mixed tasks**: all 19 tasks, uniform sampling
- Purpose: metacognition heads learn to discriminate ACROSS tasks
- 30 epochs, LR 0.0001
- All auxiliary losses active
- Success: text_gate variance across tasks > 0.3

## Loss Function

```python
total_loss = (
    task_loss                           # CE or focal (primary)
    + 0.1 * contrastive_loss            # semantic contrastive
    + 0.1 * calibration_loss            # confidence calibration
    + 0.01 * efficiency_loss            # iteration budget
    + 0.05 * gating_diversity_loss      # Phase E only
)
```

### Gating Diversity Loss (Phase E only)

```python
text_gates = [control.text_gate for control in batch_controls]
gating_diversity_loss = -torch.var(torch.stack(text_gates))
```

## Optimizer Groups

| Group | Components | LR | Notes |
|-------|-----------|-----|-------|
| 1 | GNN + TAT + executive | 0.001 | Main backbone |
| 2 | DSM | 0.0001 | Frozen in Phase A |
| 3 | TopoBridge + text_extractor.proj | **0.001** | Up from 0.0001 |
| 4 | MetaCognitive Controller | 0.0005 | New metacog heads + task embedding |
| 5 | Classifier MLP heads | 0.001 | Rebuilt per phase |

## Bug Fixes

1. **node_texts AttributeError**: Set `cc.node_texts = []` in structural task generators, preserve in `CellComplex.clone()`
2. **Rebalancing too aggressive**: Remove rebalancing, use focal loss + class_weights instead
3. **text_extractor.proj LR**: Separate optimizer group at 0.001

## Files to Modify

| Order | File | Action |
|-------|------|--------|
| 1 | `src/gnn_executive/metacognitive_controller.py` | Create (~200 lines) |
| 2 | `src/gnn_executive/control_head.py` | Modify: new ControlSignal fields |
| 3 | `src/training/multi_head_classifier.py` | Modify: TaskHead MLP |
| 4 | `src/reasoning_loop/executive_loop.py` | Modify: iteration budget, task_id |
| 5 | `src/benchmarks/run_comparison.py` | Modify: gated composition |
| 6 | `src/training/batch_utils.py` | Modify: gated composition (batched) |
| 7 | `scripts/run_dsm_curriculum.py` | Modify: Phase E, optimizer groups, aux losses |
| 8 | `config/v7_metacognition.yaml` | Modify: Phase E config |
| 9 | `src/data/cell_complex.py` | Modify: node_texts in clone() |
| 10 | Tests | Create/modify: MetaCognitiveController, integration |

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| kg_relation bal_acc | 24.2% | >50% |
| kg_concept bal_acc | 19.3% | >40% |
| kg_pathvalid bal_acc | 52.0% | >75% |
| text_gate variance | N/A | >0.3 |
| structural retention (bfs) | 99.8% | >95% |
| confidence ECE | N/A | <0.15 |

## VRAM Budget (24GB RTX 4090)

| Component | Estimate |
|-----------|----------|
| Qwen embed_tokens | ~0.6GB |
| GNN/TAT/executive | ~1GB |
| DSM (205M params) | ~0.8GB |
| MetaCog controller | ~0.001GB |
| Classifier MLP heads | ~0.001GB |
| Activations + optimizer | ~4GB |
| **Total** | **~6.4GB** |

Headroom: ~17.6GB free for larger batches or future expansion.

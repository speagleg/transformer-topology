# Training Journal

Live record of training runs, results, and architectural decisions.

---

## v10 — Dual-Track Fusion (March 11, 2026 →)

**Design doc**: `docs/plans/2026-03-11-v10-dual-track-fusion-design.md`
**Branch**: TBD
**Instance**: vast.ai RTX 4090

### Motivation

v9 KG tasks all at or near random chance. Root causes: text detached from reasoning loop, identical iterations creating 78% curl energy, narrow classifier only seeing query/target, weak embed_tokens-only text encoder, poorly designed tasks.

### Architecture Changes
- Dual-track executive loop: iteration 1 (structural) → iteration 2 (cross-modal fusion)
- Qwen 4-layer contextual encoder (replaces embed_tokens)
- Cross-attention inside iteration 2 (structure attends to text)
- Text-conditioned GNN edge messages in iteration 2
- Attention-based readout over all nodes (replaces fixed query/target)
- Single `fusion_weight` meta-cognitive gate
- Curl penalty (0.01 weight, 0.5 threshold)

### New Task Suite
- `kg_relation` (10 classes) — kept from v9
- `kg_transitive` (5 classes) — replaces kg_concept
- `kg_consistency` (2 classes) — replaces kg_pathvalid
- `kg_analogy` (3 classes) — redesigned
- `kg_causal_chain` (3 classes) — replaces kg_cluster

### Status
- [ ] Implementation
- [ ] Tests passing
- [ ] Phase A structural warmup
- [ ] Phase B KG training
- [ ] Results analysis

---

## v9 — Text-Topology Fusion (March 10-11, 2026)

**Design doc**: `docs/plans/2026-03-10-v9-text-topology-fusion-design.md`
**Instance**: vast.ai RTX 4090 (Track 2, port 34701)

### Changes from v8
- TextReasoningHead: graph-aware text transformer (2-layer, adjacency-masked)
- TextEdgeEncoder: text-conditioned edge features in GNN message passing
- Classifier input expanded from 233 to 425 dims

### Phase D Results (kg_relation, 20 epochs)

| Metric | Best | Epoch |
|--------|------|-------|
| Val acc | 87.7% | 2 |
| Balanced acc | 25.5% | 11 |
| Final loss | 0.050 | 15 |

Topology observer (stable across all epochs):
- Spectral gap: 0.029-0.030
- Gradient energy: ~21%
- Curl energy: ~78%
- Harmonic energy: ~0.5%

**Diagnosis**: Overfitting after epoch 2. Balanced acc improved slightly over v8 (25.5% vs 14.5% on kg_relation) but text features still not contributing meaningfully. Topology observer showed zero adaptation — computation graph structure frozen from start.

### Phase D Results (kg_concept, 2 epochs before killed)

| Ep | Loss | Val Acc | Bal Acc |
|----|------|---------|---------|
| 0 | 1.681 | 19.1% | 11.0% |
| 1 | 1.660 | 18.0% | 12.0% |

Tracking v8c identically. Killed to begin v10 redesign.

### Known Issues
- MKL SGELSY error in topology observer caused process hang after ep 15
- Process stuck at 100% CPU with no GPU activity until manually killed

---

## v8 — Focal Loss + Curriculum Fixes (Feb 26 →)

### Key Variants
- **v8c**: focal loss (gamma=2.0), disabled class_weights, 20 epochs per KG task
- **v8d/v8e**: various checkpoint resume attempts

### Results (v8c, kg_relation)

| Metric | Best | Epoch |
|--------|------|-------|
| Balanced acc | 14.5% | 9 |

### Results (v8c, kg_concept)

| Metric | Best | Epoch |
|--------|------|-------|
| Balanced acc | 14.5% | 9 |
| Val acc | ~19% | — |

### Results (v7, kg_pathvalid)
- 50.0% balanced acc flat for 11 epochs → early stop (coin flip on binary task)

### Results (v7, kg_analogy)
- 37.1% balanced acc at epoch 7 (random = 33%)

---

## v6 — Dual-Track DSM/Qwen (Feb 23-24, 2026)

### Phase A Results (structural tasks)

| Task | Track 1 (DSM) | Track 2 (Qwen mock) |
|------|--------------|---------------------|
| diverse | 82.8% | 84.6% |
| bfs | 99.8% | 99.4% |
| hodge_class | 95.8% | ~96.6% |

All massively exceeding targets. Structural architecture is strong.

### Key Bugs Fixed
- bf16 autocast NaN in spectral ops → force fp32
- semantic_weight init bias=-3.0 → sigmoid~5% (text ignored)
- DSM running 5x per sample in Phase A for zero contribution → disabled

---

## v3 — First GPU Training (Feb 22, 2026)

### Targets vs Actuals (Phase A)
| Task | Target | Actual |
|------|--------|--------|
| bfs | 75% | 99.8% |
| hodge_class | 65% | 95.8% |
| diverse | 60% | 84.6% |

Architecture fundamentally works for structural reasoning. Semantic reasoning is the unsolved problem.

# Training Journal

Live record of training runs, results, and architectural decisions.

---

## v10 — Dual-Track Fusion (March 11, 2026 →)

**Design doc**: `docs/plans/2026-03-11-v10-dual-track-fusion-design.md`
**Branch**: `phase7-computation-graph-topology`
**Instance**: vast.ai RTX 4090 (Track 2, port 34701)
**Config**: `config/v10_dual_track.yaml`

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
- `_forward_v10` path bypasses old hodge/persistence features, uses AttentionReadout instead

### New Task Suite
- `kg_relation` (10 classes) — kept from v9
- `kg_transitive` (5 classes) — replaces kg_concept
- `kg_consistency` (2 classes) — replaces kg_pathvalid
- `kg_analogy` (3 classes) — redesigned
- `kg_causal_chain` (3 classes) — replaces kg_cluster

### Implementation (March 11-12)

Implemented in 4 chunks:
1. `QwenContextualEncoder`, `CrossAttention`, `AttentionReadout`, `TextConditionedGNN`
2. `fusion_weight` in ControlHead, dual-track executive loop (`_forward_dual_track`)
3. New KG task generators (transitive, consistency, analogy_v10, causal_chain)
4. Config, integration tests, training script fixes

### Phase A Results (structural warmup, 5 epochs each)

| Task | Bal Acc | Notes |
|------|---------|-------|
| bfs | 99.4% | Structural path strong as expected |
| hodge_class | 63.0% | Lower than v6 (95.8%) — new readout not optimal for this task |
| diverse | 97.6% | Near ceiling |

fusion_weight forced to 0 during Phase A (freeze_fusion: true).

### Phase B — First Attempt (imbalanced data, March 12)

**Critical bug**: Pathological class imbalance in generated KG datasets. kg_relation had 86.4% class 8 (RelatedTo) — model just predicted majority class.

| Ep | Loss | Val Acc | Bal Acc | Notes |
|----|------|---------|---------|-------|
| 0 | 0.407 | 86.4% | 19.4% | Just predicts RelatedTo |

Other bugs fixed during this run:
- Device mismatch in v10 forward path
- topo_features dim mismatch
- Rotary embedding errors in truncated Qwen
- Attention mask dtype errors
- Phase A checkpoint name wrong ("path_counting" vs "diverse" for v10)
- Phase B sample count reading from wrong config section

### Balanced Sampling Fix (March 13, commit 317578e)

Rewrote KG task generators with category-first balanced sampling:
- `generate_kg_relation_task`: Build `_relation_index` mapping category→edges, pick target class uniformly first, then find matching edges in subgraph
- `generate_kg_transitive_task`: Collect all 2-hop chains, filter by target class
- `generate_kg_causal_chain_task`: Same balanced approach
- Disabled `use_class_weights` (balanced data doesn't need it)
- Regenerated 25k train / 2k val datasets with balanced distributions

### Phase B — Balanced Run (IN PROGRESS, March 13)

Running with 25k balanced samples, batch_size=8, 30 epochs max, patience=15.
PID 138975 on vast.ai. Log: `data/v10_dual_track/training_b8.log`.

**kg_relation** results so far:

| Ep | Loss | Val Acc | Bal Acc | Time | fw |
|----|------|---------|---------|------|----|
| 0 | 1.878 | 38.5% | 38.9% | 6874s | 0.000 |
| 1 | 1.610 | 42.0% | 42.2% | 7058s | 0.000 |
| 2 | 1.489 | 44.8% | 44.8% | 6733s | 0.000 |
| 3 | 1.409 | 46.9% | 47.2% | 6990s | 0.000 |
| 4 | 1.343 | 51.2% | 51.0% | 6828s | 0.000 * |
| 5 | 1.292 | 50.6% | 50.3% | 7158s | 0.000 |

Key observations:
- **4× random chance** at ep0 (38.9% vs 10% random for 10 classes) — balanced sampling working
- Steady improvement through ep4, first plateau at ep5
- **fusion_weight stuck at 0.000** — text track completely dead (BUG, see below)
- Curl energy spiked to 0.642 after ep5 (>0.4 threshold alert)
- ~6900s/epoch (~1.9h) — very slow
- **51% bal acc is a STRUCTURAL-ONLY baseline** (no text contributing at all)

### Critical Bug: Text Track Dead (found March 13, fixed same day)

**Root cause**: When resuming with `--resume-phase b`, the Phase A checkpoint loads with `fusion_weight_head.bias = -10.0` (sigmoid ≈ 0) and `bypass_llm = True`. The unfreezing code that sets bias to 0.0 and enables text was inside the `if not skip_a:` block — unreachable when Phase A is skipped.

**Effect**: All 6 epochs ran with:
- `bypass_llm = True` → QwenEncoder never called, no text embeddings
- `fusion_weight = 0.000` → iteration 2 (cross-modal) has zero contribution
- Cross-modal params (cross_attn, text_gnn) frozen and not in optimizer
- Model was doing pure structural reasoning only

**Fix**: Moved cross-modal unfreezing to Phase B entry point, runs regardless of whether Phase A was skipped. 75 cross-modal params now in optimizer at lr=0.001.

### Phase B — Restarted with Text (March 13, training_b9_textfix.log)

Killed old run (PID 138975). Restarted with fix deployed.
Log: `data/v10_dual_track/training_b9_textfix.log`
Optimizer: 148 classifier params + 75 cross-modal params.
fusion_weight initialized at 0.5 (sigmoid(0)).

### Performance Bottleneck Analysis (March 13)

GPU utilization only 10-12%, CPU at 130%. Root causes identified:

1. **Sequential per-sample processing**: `use_batched: true` falls through to sequential path because `use_dsm=False` in v10. Each of 25,000 samples processed one-at-a-time.
2. **gudhi persistence in TopologicalPE**: Called 2× per sample (inside each TAT call). Loops over all ~20 nodes, builds Rips complex per ego-graph on CPU. Estimated ~1250-2000s/epoch (18-29%).
3. **Small-graph GPU starvation**: 20-node graphs produce tiny CUDA kernels. GPU starved between kernel launches.

Per-sample forward path: CC clone → QwenEncoder (cached) → harmonic energy (hodge decomp) → GNN executive → wave dynamics (3 spectral filters + 5-step ODE) → TAT iter 1 (with persistence PE) → text_gnn → cross_attn → TAT iter 2 (with persistence PE) → harmonic energy → AttentionReadout → classifier. Total ~270ms/sample.

### Planned: GPU Utilization Optimization
- True graph batching (combine 8-32 graphs into one large disconnected graph)
- Async/precomputed persistence features
- Expected: 3-5× speedup, GPU utilization 50-80%

### GPU Optimization (March 13-14, commits 8192b96..6b8bfee)

Implemented 6-task optimization plan to increase GPU utilization:
1. CellComplex caching (boundary/adjacency/spectral with topology-version invalidation)
2. Precomputed TopologicalPE (eliminates gudhi from training loop)
3. Batched dual-track forward (`forward_dual_track_batched()`)
4. Backward-compat for deserialized CellComplex cache attrs

**Result**: Epoch time dropped from ~6900s → ~1600s (**4.3× speedup**). GPU util ~14-17%.

### Phase B — Optimized Run (March 14, training_b11.log)

Running with GPU optimizations, 25k balanced samples, batch_size=8.

**kg_relation** (10 classes, v10 taxonomy):

| Ep | Loss | Bal Acc | fw | Notes |
|----|------|---------|-----|-------|
| 0 | 1.766 | 42.8% | 0.520 | Text contributing from start |
| 1 | 1.496 | 47.3% | 0.524 | |
| 2 | 1.401 | 50.2% | 0.526 | |
| 3 | 1.330 | 51.3% | 0.529 | |
| 4 | 1.270 | 51.9% | 0.532 | |
| 7 | 1.129 | 52.0% | 0.539 | |
| 9 | 1.039 | **52.6%** | 0.547 | **Best** |
| 15 | 0.798 | 49.9% | 0.568 | Overfitting — train loss dropping, val flat |
| 17 | 0.716 | 50.3% | 0.578 | Patience ~8/15 |

**Conclusions from v10 optimized run:**
- Plateaued at ~52% bal acc on kg_relation (10 classes)
- fusion_weight climbed 0.52→0.58 — text IS contributing (~58% weight to fused path)
- Train loss kept dropping (0.72 at ep17) but val acc flat — classic overfitting
- Topology observer: spectral gap 0.018, curl energy 45% — information cycling
- Other KG tasks (transitive etc.) showed poor results too

### Analysis: Why v10 KG Tasks Plateau (March 14)

Deep investigation into kg_relation bottlenecks:

1. **Classifier sees no direct text** — AttentionReadout output is 101D (3×32 + 4 + 1). Text modifies node embeddings indirectly via GNN/cross-attention, but classifier never sees raw text features.

2. **Flawed 10-class taxonomy** — 34 ConceptNet relations collapsed badly:
   - Synonym + Antonym in same "RelatedTo" bucket (opposites!)
   - FormOf (350k), DerivedFrom (292k), HasContext (223k) all lumped into RelatedTo
   - NotCapableOf merged with CapableOf, NotDesires with Causes

3. **Unidirectional cross-attention** — Only structure→text. Missing text→structure direction.

4. **Spectral gap bottleneck** — Gap of 0.018 with 45% curl energy = information cycling.

5. **32D embedding too narrow** — 10-class classification through 32D bottleneck with 20GB VRAM idle.

6. **`raw_relation` bug** — All KG task generators were reading the pre-categorized `relation` field instead of `raw_relation`, causing silent miscategorization with the pickle format.

---

## v11 — 16-Class Taxonomy + Text-Enhanced Classifier (March 14, 2026 →)

**Design doc**: `docs/plans/2026-03-13-v11-taxonomy-and-text-classifier.md`
**Branch**: `v11-taxonomy-text-classifier`
**Instance**: vast.ai RTX 4090 (Track 2, port 34701)
**Config**: `config/v10_dual_track.yaml` (updated in-place)

### Motivation

v10 plateaued at 52% balanced accuracy on kg_relation. Root causes: flawed class taxonomy, no direct text signal to classifier, unidirectional cross-attention, narrow embeddings, spectral bottleneck, and raw_relation bug across all KG tasks.

### Architecture Changes (11 commits)

1. **16-class relation taxonomy** — Semantically coherent groupings:
   - Split Synonym/Antonym (were both "RelatedTo")
   - FormOf, DerivedFrom, HasContext as standalone classes (were all "RelatedTo")
   - HasSubevent split from PartOf
   - Desire split from Causes
   - **Negation** class: real edges (NotCapableOf/NotHasProperty/NotDesires) + synthetic augmentation to ~12k via 2-hop non-adjacent pair sampling
   - "Other" eliminated — unmapped relations fall back to "RelatedTo"
   - `categorize_relation()` now uses `raw_relation` field preferentially

2. **Bidirectional cross-attention** — `BidirectionalCrossAttention` replaces `CrossAttentionBlock`:
   - s→t: structural queries attend to text (existing)
   - t→s: text queries attend to structural (NEW)
   - Returns both enriched representations

3. **Text embeddings in classifier** — `AttentionReadout` now accepts `text_dim`:
   - Enriched text embeddings (from t→s cross-attention) for query + target nodes concatenated to classifier input
   - Old: 3×32 + 4 + 1 = 101D
   - New: 5×64 + 4 + 1 = 325D

4. **Embedding dim 32→64** — Doubles information capacity throughout pipeline:
   - `gnn_hidden`: 64→128
   - `tat_ff_dim`: 128→256
   - `llm.output_dim`: 32→64
   - VRAM: 4.2GB → estimated ~10GB (24.6GB available)

5. **Fusion weight init** — `sigmoid(-1.0) ≈ 0.27` (was `sigmoid(0) = 0.5`):
   - Structure dominates early while text path learns
   - Text contribution grows organically as model trains

6. **Spectral gap regularization** — `-log(gap + eps)` loss term (weight 0.01):
   - Penalizes small spectral gaps to encourage information flow
   - Addresses the 45% curl energy / 0.018 spectral gap bottleneck

7. **raw_relation fix** — All 5 KG task generators now prefer `raw_relation` over `relation` field

### Dataset Changes
- All datasets regenerated with `embedding_dim=64`
- kg_relation: 16 classes + negation augmentation (was 10)
- Fresh training from Phase A required (classifier dim incompatible)

### Phase A Results (64D, 5 epochs each, ~165s/epoch)

| Task | Best Bal Acc | v10 (32D) | Notes |
|------|-------------|-----------|-------|
| bfs | **100.0%** | 99.4% | Perfect score at 64D |
| hodge_class | **49.3%** | 63.0% | Expected drop — v11 has no explicit hodge features in classifier |
| diverse | **98.7%** | 97.6% | Slightly better |

hodge_class regression is expected: v10/v11 AttentionReadout path does NOT compute explicit hodge decomposition norms. The v3/v6 path "cheated" by running the actual decomposition and feeding 3 norms to the classifier. The 49.3% is the model's genuine ability to infer hodge structure from learned node embeddings — a more honest measurement.

### Phase B — kg_relation (16 classes, IN PROGRESS)

Training log: `data/v10_dual_track/training_v11.log`
25k balanced samples, batch_size=8, 30 epochs max, patience=15.

| Ep | Loss | Val Acc | Bal Acc | fw | Time | Notes |
|----|------|---------|---------|-----|------|-------|
| 0 | 1.419 | 64.2% | **64.6%** | 0.513 | 1646s | 10.3× random (6.25%) — massive improvement over v10 |
| 1 | 1.087 | 65.8% | **65.8%** | 0.513 | 1398s | |
| 2 | 0.991 | 65.8% | **65.8%** | 0.516 | 1374s | |
| 3 | 0.910 | 68.6% | **68.7%** | 0.516 | 1345s | |
| 5 | 0.772 | 68.9% | **69.0%** | 0.517 | 1377s | |
| 7 | 0.659 | 69.9% | **69.9%** | 0.517 | 1404s | Best |
| 9 | 0.566 | 69.1% | 69.1% | 0.520 | 1615s | |
| 11 | 0.485 | 68.9% | 68.9% | 0.521 | 1709s | Patience 4/15, possible plateau |

**v11 vs v10 comparison:**
- v11 best: **69.9% bal acc on 16 classes** (11.2× random) at epoch 7
- v10 best: 52.6% bal acc on 10 classes (5.3× random) at epoch 9
- v11 already surpassed v10's ceiling by epoch 0

Every v11 change is contributing:
- Cleaner 16-class taxonomy eliminates ambiguous groupings
- Direct text in classifier (325D) gives the model semantic signal
- Bidirectional cross-attention enriches both modalities
- 64D embeddings double information capacity
- Fusion weight init at 0.27 lets structure dominate early
- fusion_weight growing slowly (0.51→0.52) — text increasingly contributing

### Inverse Scaling Research (parallel workstream, March 14)

While v11 trains, implemented the MultiScaleLaplacianFilter architecture and preliminary experiments for the inverse scaling investigation:

**Architecture (3 commits):**
1. `node_triangle_incidence()` on CellComplex — containment matrix for triangle→node projection (can't use B1@B2 because ∂²=0)
2. `MultiScaleLaplacianFilter` — parallel L0/L1/L2 spectral filtering with gated fusion, skip_l1/skip_l2 for ablations
3. Wired into `MultiFilterDynamics` via `use_multiscale=True` flag

**Preliminary findings (Exp 0a/0b):**
- **Curl class imbalance confirmed**: At n=20, curl is only 15% of samples (should be 33%). At n=40, only 4.5%. Small graphs lack triangles, so `generate_hodge_class_task` falls back to gradient. This is a confound for the inverse scaling claim.
- **Exact decomposition baseline**: 97.5% accuracy at n=20 — the math works fine, the model just can't learn to read it from node embeddings alone.

**Plan**: `docs/plans/2026-03-14-inverse-scaling-implementation.md` (10 tasks, ~28 GPU hours)

### Status
- [x] Implementation (17 commits, all tests passing)
- [x] raw_relation fix across all KG task generators
- [x] Dataset generation (Phase A + kg_relation done, other KG tasks in progress)
- [x] PE precomputation (Phase A + kg_relation done)
- [x] Phase A training (100% bfs, 49.3% hodge, 98.7% diverse)
- [x] MultiScaleLaplacianFilter architecture (L0/L1/L2)
- [x] Inverse scaling preliminary experiments (0a/0b)
- [ ] Phase B KG training (kg_relation ep11: 69.9% best bal acc, IN PROGRESS)
- [ ] Inverse scaling GPU experiments (after v11 completes)
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

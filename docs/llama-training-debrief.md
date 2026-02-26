# Llama 3.2 1B Curriculum Training: Full Debrief

**Date**: 2026-02-19
**Hardware**: vast.ai RTX 4090 (24 GB VRAM), ~$0.30/hr
**Model**: Hierarchical GNN Executive + TAT + Llama 3.2 1B (LoRA rank 16)
**Total params**: 1.287B (50.8M trainable: LoRA + TopoBridge + cross-attn)

---

## 1. Training Summary

Three-phase curriculum with `config/llama_training.yaml`:

### Phase A: Regression Lock (bypass_llm=True)
GNN/TAT core training, no LLM interference. Gate penalty = 1.0.

| Task | Classes | Best Val | Epochs |
|------|---------|----------|--------|
| diverse | 11 | 71.0% | 30 (patience) |
| bfs | 16 | 100% | 9 (perfect) |
| hodge_class | 3 | 82.0% | 25 (patience) |
| spectral_gap | 8 | 73.0% | 17 (patience) |

### Phase B: Graph Completion (LLM unfrozen, half LR)

| Task | Classes | Best Val | Epochs |
|------|---------|----------|--------|
| graph_completion | 2 | 80.5% | 18 (patience) |
| path_counting | 5 | 98.0% | 21 (patience) |

### Phase C: Language + Analogy (fine-tuning LR)

| Task | Classes | Best Val | Epochs | Notes |
|------|---------|----------|--------|-------|
| labeled_reasoning | 3 | 76.5% | 20 (patience) | Usable |
| analogical_transfer | 5 | 18.5% | ~10 | Killed (near random) |
| graph_completion | 2 | — | — | Instance terminated |

**Total GPU time**: ~48 hrs estimated

---

## 2. Generalization Results

Evaluated 5 checkpoints across 4 axes (500 test samples each).

**Note**: Phase A checkpoint was overwritten during `--resume-phase c` (see checkpoint management section). Phase A numbers below reflect a Phase C model evaluated on spectral_gap — they are NOT valid Phase A results.

### Accuracy Matrix

| Checkpoint | Task | ID (n=20) | OOD (n=48) | OOD (n=80) | Topo Transfer |
|------------|------|-----------|------------|------------|---------------|
| phase_a | spectral_gap | ~~3.6%~~ | ~~13.4%~~ | ~~19.8%~~ | ~~11.2%~~ |
| **phase_b_gc** | **graph_completion** | **75.2%** | **77.6%** | **69.8%** | **87.2%** |
| **phase_b_pc** | **path_counting** | **96.2%** | **98.4%** | **95.6%** | **98.6%** |
| **phase_c_lr** | **labeled_reasoning** | **68.4%** | **75.0%** | **76.6%** | **72.4%** |
| phase_c_gc | graph_completion | 66.4% | 73.4% | 74.0% | 75.6% |

### Key Observations

1. **Size generalization is strong**: Path counting barely degrades (96.2% → 95.6% at 4x size). Graph completion and labeled_reasoning actually *improve* slightly at larger sizes — surprising.
2. **Topology transfer works**: Path counting hits 98.6% on held-out topologies. Graph completion jumps from 75.2% (ID) to 87.2% (topo transfer) — the model learned generalizable edge-presence features rather than topology-specific patterns.
3. **Graph completion regresses after Phase C**: Phase B best = 75.2% → Phase C re-trained = 66.4% (8.8% drop on ID). Phase C fine-tuning on labeled_reasoning damaged the graph_completion capability.
4. **Phase A checkpoint was lost**: Overwritten during resume. The 5.1 GB local copy (fp32) is the only surviving original.

---

## 3. Per-Class Analysis

### Path Counting (5 classes) — Phase B checkpoint

| Class | ID Acc | Count | Notes |
|-------|--------|-------|-------|
| 0 | — | 0 | No samples (paths with 0 hops are trivial) |
| 1 | 96.5% | 115 | 1-hop paths |
| 2 | 95.8% | 166 | 2-hop paths |
| 3 | 92.6% | 94 | 3-hop paths — hardest |
| 4 | **100%** | 125 | 4+ hop paths |

Near-perfect performance. Class 3 (3-hop) has the most errors but still >92%.

### Labeled Reasoning (3 classes) — Phase C checkpoint

| Class | Meaning | ID Acc | Count | Notes |
|-------|---------|--------|-------|-------|
| 0 | causal_chain | **39.4%** | 175 | Severely undertrained |
| 1 | blocked | **84.0%** | 325 | Dominant prediction |
| 2 | independent | — | 0 | **Missing from test set** |

**Critical issue**: Class 2 (independent) has zero samples in ID test. The model defaults to predicting "blocked" — it learned a class-imbalanced shortcut. The 68.4% overall accuracy is inflated by class 1 dominance.

### Graph Completion (2 classes) — Phase B checkpoint

| Class | Meaning | ID Acc | Count |
|-------|---------|--------|-------|
| 0 | edge missing | 78.1% | 247 |
| 1 | edge present | 73.1% | 253 |

Balanced classes, roughly symmetric performance. The model learns genuine edge-presence signals.

---

## 4. Analogical Transfer Post-Mortem

### What Happened
- Best validation: 18.5% (random baseline: 20% for 5 classes)
- Training killed after ~10 epochs with no improvement

### Root Cause Analysis

**Fundamental problem**: Roles are **randomly assigned** to nodes (line 237 of `llm_tasks.py`):
```python
role_nodes = random.sample(nodes, num_roles)
```

This means:
1. **Zero structural signal** for GNN/TAT — role nodes have no distinguishing graph properties
2. The ONLY signal is in metadata `task_prompt`: `"query_role=producer"`
3. TopoBridge must: parse role name from text → map to index → encode in per-node embeddings
4. This is a **pure text comprehension task** requiring the LLM pathway

### Why It Still Failed

1. **LLM gate stays near zero** (0.013-0.089 across all checkpoints) — the gate penalty in Phase A permanently suppressed LLM usage, and Phase B/C didn't recover it
2. **Capacity**: Rank-16 LoRA + 8 prefix tokens may be insufficient for role-name parsing
3. **Combinatorics**: 6 domain pairs x 5 roles x 8 topologies = high diversity for ~1000 samples
4. **Architecture gap**: CrossAttention decoder produces uniform node embeddings — but role info needs to reach the specific query node embedding, not all nodes equally
5. **Training signal**: With `max_classes=16` (config override), 11 unused output classes dilute gradients

### Task Property Analysis (500 samples)

- **Class distribution**: Roughly balanced (18-22% per class)
- **Domain pairs**: 6 unique pairs, uniformly sampled (74-98 each)
- **Role node degree vs average**: role 0=3.35, role 1=3.36, role 2=3.52, role 3=3.84, role 4=3.27 — **no structural difference** (confirms random assignment)
- **Random baseline**: 20%, **majority class baseline**: ~22%

### Recommendations

1. **Add structural role signal**: Assign roles to degree-centrality-ordered nodes instead of random
2. **Reduce combinatorics**: Start with 1-2 domain pairs, 3 roles
3. **Fix max_classes**: Use `get_max_classes(task)` instead of config override
4. **Increase data**: 5000+ samples (currently ~1000)
5. **Address gate suppression**: Reduce Phase A gate penalty weight or add Phase B gate-opening incentive

---

## 5. Executive Behavior (Diagnostics)

### Control Signal Statistics

| Signal | Phase B GC | Phase B PC | Phase C LR | Phase C GC |
|--------|------------|------------|------------|------------|
| iterations (mean) | 2.0 | 2.1 | 2.1 | 1.9 |
| freq_gate (mean) | 0.681 | 0.633 | 0.632 | 0.617 |
| spatial_focus (gini) | 0.372 | 0.290 | 0.272 | 0.181 |
| confidence (mean) | 0.286 | 0.447 | 0.514 | 0.581 |
| diffusion_time | 1.955 | 3.786 | 3.870 | 2.083 |
| llm_gate (mean) | **0.075** | 0.021 | 0.014 | **0.089** |
| filter_weights | cheb 46%, id 45% | id 81%, cheb 19% | id 100% | id 89%, cheb 11% |

### Interpretation

1. **LLM gate is effectively dead**: Mean gate values of 0.014-0.089 mean the TopoBridge pathway contributes <9% to output embeddings. Phase A's gate penalty (1.0) successfully suppressed the gate, but Phase B/C never recovered it. **The LLM is barely being used.**

2. **Filter specialization collapsed to identity**: By Phase C, the ensemble weights are [0.00, 0.00, 0.00, 1.00, 0.00] — pure identity filter. The model learned to bypass all spectral processing. Only graph_completion (Phase B) maintains a chebyshev+identity mix (46%/45%).

3. **Convergence is fast**: 1.9-2.1 iterations on average (max 5). The executive loop converges quickly, suggesting the problems it's solving don't require deep iterative reasoning.

4. **Confidence increases through curriculum**: 0.286 (Phase B GC) → 0.581 (Phase C GC). Later training phases produce more confident models.

5. **Spatial focus decreases**: Gini drops from 0.372 → 0.181 through phases, meaning attention becomes more uniform (less focused on specific nodes). This could indicate the model is learning global rather than local features.

---

## 6. Key Findings

### What Works

1. **Path counting is nearly solved**: 96-99% across all evaluation axes — size OOD, topology transfer, everything. The GNN/TAT backbone excels at structural counting tasks.
2. **Graph completion generalizes well**: 75-87% with strong topology transfer (87.2% on unseen topologies vs 75.2% ID — it actually performs better on new graph types).
3. **Size generalization is real**: Mixed-size training (n=16-48) generalizes to n=80 with minimal degradation. Path counting drops only 0.6% at 4x training size.
4. **The GNN/TAT backbone is the workhorse**: With LLM gate <9%, essentially all reasoning is done by the GNN executive + topology-aware transformer.

### What Doesn't

1. **Analogical transfer (18.5%)**: Below random — fundamentally broken task design (no structural signal)
2. **LLM integration provides no measurable benefit**: Gate stays near zero, ensemble collapses to identity. The 1.24B frozen Llama parameters are dead weight.
3. **Phase C causes regression**: Graph completion dropped 8.8% after labeled_reasoning training. Sequential task training is destructive.
4. **Labeled reasoning class imbalance**: Class 2 (independent) absent from test data; class 0 (causal) at only 39.4%. The 68.4% headline number is misleading.
5. **Checkpoint management was lossy**: Phase A original overwritten during resume.

### Architecture Insights

1. **Ensemble wave mode collapses**: The model learns to use only the identity filter, bypassing all spectral dynamics (heat, chebyshev, wave_cosine, sheaf). This suggests either (a) the spectral filters don't help for these tasks, or (b) the ensemble training is unstable and defaults to the simplest option.
2. **TopoBridge round-trip exists but is unused**: The architecture correctly routes topo→LLM→topo, but the gate mechanism prevents information flow. The 0.075 gate for graph_completion is the highest — there's a faint signal that the LLM helps for edge-presence tasks.
3. **Executive loop converges in 2 iterations**: The max_iterations=5 setting is rarely reached. The harmonic energy convergence criterion works but the model doesn't need deep iteration for these tasks.

---

## 7. Recommended Next Steps

### Critical Fixes (do before next training run)

1. **Fix max_classes per task** — Use `get_max_classes(task)` instead of global 16. This wastes 60%+ of classifier capacity on unused outputs.
2. **Fix labeled_reasoning class balance** — Ensure all 3 classes appear in train/test data. Current `independent` class is missing.
3. **Fix checkpoint management** — Save Phase A original separately; never overwrite for resume.
4. **Reduce gate penalty or add gate-opening phase** — Phase A's penalty=1.0 permanently kills the LLM pathway. Try penalty=0.1, or add an explicit "gate warm-up" phase between A and B.

### Architecture Changes

5. **Ablate the LLM entirely** — Run the same curriculum with `use_llm=False`. If accuracy is the same, the 1.24B Llama params are pure overhead.
6. **Fix analogical_transfer** — Assign roles by degree centrality, reduce to 3 roles and 2 domain pairs, add structural role features to node embeddings.
7. **Try single filter instead of ensemble** — If the model always picks identity, the ensemble adds complexity with no benefit. Try chebyshev-only or identity-only baselines.
8. **Multi-task training** — Train all Phase B/C tasks simultaneously (mixed batches) instead of sequentially, to prevent regression.

### Training Improvements

9. **Increase data** — 5000+ train samples per task (currently 1000-2000)
10. **Increase Phase A budget** — 50+ epochs (benchmark suite needed 22 epochs for simpler model)
11. **Cosine annealing** — Replace ReduceLROnPlateau with cosine schedule
12. **EWC or rehearsal** — Add elastic weight consolidation or task rehearsal to prevent catastrophic forgetting between phases

---

## Appendix: Checkpoint Details

| File | Size | Contents | Notes |
|------|------|----------|-------|
| `phase_a_spectral_gap.pt` (vast.ai) | 2.5 GB | Phase C weights (overwritten) | NOT valid Phase A |
| `phase_a_spectral_gap_original.pt` (vast.ai) | 2.5 GB | Same as above | Also overwritten |
| `phase_a_spectral_gap.pt` (local) | 5.1 GB | **Real Phase A** (fp32) | Only surviving copy |
| `phase_b_graph_completion_best.pt` | 2.5 GB | Phase B graph_completion best | Valid |
| `phase_b_path_counting_best.pt` | 2.5 GB | Phase B path_counting best | Valid |
| `phase_c_labeled_reasoning_best.pt` | 2.5 GB | Phase C labeled_reasoning best | Valid |
| `phase_c_graph_completion_best.pt` | 2.5 GB | Phase C graph_completion best | Valid |

## Appendix: Files

| File | Purpose |
|------|---------|
| `config/llama_training.yaml` | Training config |
| `scripts/run_phase4c_curriculum.py` | Curriculum training script |
| `scripts/run_full_evaluation.py` | Evaluation script |
| `scripts/generate_llama_datasets.py` | Dataset generation |
| `docs/gpu-training-reference.md` | Quick reference for GPU deployment |
| `data/llama_checkpoints/eval_results.json` | Raw evaluation metrics |

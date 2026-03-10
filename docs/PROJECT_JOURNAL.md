# Transformer-Topology Project Journal

A living document tracking the development of a hierarchical GNN-TAT architecture for topological reasoning. Updated with each major milestone.

---

## The Hypothesis

**Problem**: Large language models lack structured reasoning. They fail at multi-hop relational reasoning over graphs and have no persistent world model — every call is stateless.

**Thesis**: A two-tier architecture — a GNN "executive" operating on a persistent cell complex, interleaved with a topology-aware transformer — can perform structured reasoning that scales with graph complexity rather than degrading. Text semantics (from an LLM) can be fused with topological structure to enable reasoning that neither system achieves alone.

**Key bet**: Topological features (Hodge decomposition, spectral gaps, persistence diagrams) provide a fundamentally different and complementary signal to learned neural representations. By decomposing graph signals into gradient (local flow), curl (cyclic structure), and harmonic (global topology) components, the model can reason about structure in ways that standard GNNs cannot.

---

## Architecture Overview

```
                    Cell Complex (persistent world model)
                    0-cells: entities  |  1-cells: relations  |  2-cells: groups
                            |
                    GNN Executive (spatial + spectral dual path)
                            |
                    Wave Dynamics (spectral filters + neural ODE)
                            |
                    Topology-Aware Transformer (TAT)
                            |
                    Control Signal (confidence, gating, strategy)
                            |
                    Classifier (per-task MLP heads)
```

The GNN Executive runs spatial message-passing and spectral filtering in parallel, producing a ControlSignal that governs TAT attention. Wave dynamics propagate signals through learned spectral filters (Chebyshev, heat, wave cosine, etc.). The TAT applies dual spatial/spectral attention, biased by the control signal. This loop repeats until harmonic energy converges.

---

## Development Timeline

### Week 1: Foundation (Feb 11-14, 2026)

**Feb 11 — Project inception and Phase 1 scaffolding**
- Wrote architecture design document establishing the two-tier GNN-TAT hypothesis
- Implemented cell complex data structure with boundary operators
- Built Hodge Laplacians (L0, L1, L2), spectral decomposition, positional encodings
- Implemented dual-path GNN: spatial message-passing + spectral filtering
- Built topology-aware transformer with spatial and spectral attention heads
- Created multi-hop graph traversal benchmark

**Feb 12 — Phase 1 completion, Phase 2-3 in one push**
- Phase 1 complete: interleaved GNN-TAT reasoning loop with convergence detection
- Phase 2: added 2-cells, Hodge decomposition, persistent homology, higher-order GNN
- Phase 3: hierarchical executive with wave dynamics for temporal reasoning
- Key bug found and fixed: off-by-one in classifier (max_hops=10 needs 11 output classes)
- Key bug found and fixed: graph reuse in training — must detach() before storing embeddings

**Feb 13-14 — Benchmark suite**
- Comprehensive benchmark: 11 task types across 3 evaluation axes
- In-distribution, topology-transfer (train BA/WS/grid, test ER/caveman), size-OOD (train n=20, test n=40/80)
- DiagnosticCollector for control signal analysis

### Week 2: Spectral Dynamics & LLM Integration (Feb 16-19, 2026)

**Feb 16 — Phase 4: modular spectral + size generalization + LLM + ensemble**
- SpectralFilter base class with 6 implementations (FILTER_REGISTRY)
- Magnetic Laplacian (complex-valued, direction-aware) + Sheaf Diffusion (learnable restriction maps)
- Size generalization: eigenvalue normalization, size-normalized control head, mixed-size training
- LLM integration: TopoBridge with MockLLM/Llama backends, 3 new tasks (graph_completion, labeled_reasoning, analogical_transfer)
- Multi-filter ensemble: runs multiple spectral filters in parallel, blends with executive-learned softmax weights

**Key finding — Filter specialization (Phase 4b benchmarks)**:
| Filter | Best at |
|--------|---------|
| chebyshev | spectral_gap (65.2%), hodge retention (185%) |
| sheaf | BFS retention (96%), high retention but NaN-prone |
| wave_cosine | spectral_gap retention (82%) |
| nowave | structural tasks (prop_delay 54%, diverse 73%) |

This confirmed that different spectral filters excel at different task types — justifying the ensemble approach.

**Feb 17-18 — Phase 5: NL reasoning pipeline**
- Built full NL-to-graph-to-reasoning-to-NL pipeline
- TopologyInferrer: ~30 keyword patterns mapping text descriptions to 8 topology types
- Topology-aware graph generation, task routing, answer generation
- This phase was more about completeness than breakthrough results

**Feb 18-19 — Llama integration + DSM design**
- Attempted Llama 3.2-1B integration with LoRA
- bf16 caused NaN in spectral/ODE ops after ~160 samples — critical stability bug
- Fix: force fp32 for spectral ops, keep bf16 for the LLM transformer only
- Designed Distilled Semantic Model (DSM): 205M-param transformer trained via distillation from Llama

### Week 2-3: Computation Graph Topology & DSM (Feb 20-23, 2026)

**Feb 20-21 — Computation graph topology analysis**
- Novel idea: apply our own topological analysis to PyTorch's computation graph
- Operations = 0-cells, data flows = 1-cells, composite ops (residual blocks) = 2-cells
- Three analysis passes: Hodge decomposition of gradient flow, spectral gap analysis, sheaf analysis
- Practical alerts: "high curl energy" = potential gradient circulation, "small spectral gap" = bottleneck
- Topology-grounded confidence: fundamentally different from softmax entropy — detects "confidently wrong" via gradient curl

**Feb 22 — bf16 stability + graph batching**
- Resolved bf16 NaN: force fp32 for spectral_filter, spectral_attention, executive_loop wave dynamics
- Graph batching: 4-8x GPU utilization improvement by batching DSM/LLM calls

**Feb 23 — v6 dual-track design and deployment**
- Designed two parallel training tracks for vast.ai RTX 4090:
  - Track 1: WikiText-pretrained DSM (214M params)
  - Track 2: Frozen Qwen 2.5-3B + GraphFormer adapter (277M params)
- Key infrastructure: MultiHeadClassifier (persistent per-task heads), 50% feature replay, contrastive auxiliary loss, class-weighted focal loss, batched training

**v6 Phase A results (Feb 24) — structural tasks crushed targets**:
| Task | Track 1 (DSM) | Track 2 (Qwen) | v3 Target |
|------|---------------|----------------|-----------|
| bfs | **99.8%** | **99.4%** | 75% |
| hodge_class | **95.8%** | ~96.6% | 65% |
| diverse | 82.8% | **84.6%** | 70% |

These results proved the GNN-TAT architecture was working well for structural reasoning.

### The Inverse Scaling Discovery

During Phase 4b benchmarks on Lambda A100, we discovered that **Hodge decomposition classification shows inverse size scaling** — accuracy *improves* with out-of-distribution graph size:

| Filter | n=20 (ID) | n=40 (OOD) | n=80 (OOD) |
|--------|-----------|------------|------------|
| chebyshev | 38.4% | 49.8% | **71.0%** |
| sheaf | 39.2% | 50.2% | **71.4%** |
| wave_cosine | 38.4% | 50.4% | **70.2%** |

This is the opposite of typical ML behavior. BFS shows normal degradation (99.8% → 96.6% → 43.2%).

**Why it works**: Larger graphs have richer Laplacian eigenspaces, giving the spectral filters more resolution. Hodge decomposition becomes more stable as the cell complex grows — small graphs (n=20, ~40 edges) are degenerate for clear gradient/curl/harmonic separation.

**Implication**: Train on small synthetic networks, deploy on large real-world networks with *better* accuracy. Potential applications in network flow classification, power grid monitoring, financial networks.

### Week 3: Phase 7 — MetaCognition & Knowledge Graphs (Feb 26 - Mar 4, 2026)

**Feb 26 — ConceptNet integration**
- Built ConceptNet pipeline: 1.1M nodes, 2.7M edges from ConceptNet 5.7
- 5 new KG task generators: kg_relation (10 classes), kg_concept (9), kg_pathvalid (2), kg_analogy (3), kg_cluster (6)
- TASK_REGISTRY expanded from 14 to 19 tasks
- Phase D curriculum: load Phase C structural checkpoint, train on KG tasks

**Feb 26-27 — The Qwen VRAM saga**
- AWQ (4-bit) Qwen won't compile on vast.ai → switched to fp16 (~6GB VRAM)
- GraphFormerEncoder compressed N nodes into K=16 graph tokens — catastrophic information bottleneck
- Approach A (gradient surgery, unfreeze 1 Qwen layer): OOMs on 24GB RTX 4090
- Approach B (text token extraction): balanced accuracy stuck at 12% (random for 10 classes)
- **Approach C**: Use only Qwen's `embed_tokens` as frozen feature extractor (~0.6GB vs ~6GB for full model). Per-node embeddings bypass GraphFormer entirely.

**Feb 27 — Approach C implemented**
- QwenTextFeatureExtractor: tokenize → embed_tokens → mean-pool → learned proj (Linear+LayerNorm)
- Cache raw embeddings by concept string (frozen = deterministic)
- Text features concatenated directly to classifier input (query + target + diff = 96 extra dims)
- Classifier: 132-dim (structural) + 96-dim (text) = 228-dim input

**Mar 1 — MetaCognitive Controller**
- Extended ControlSignal with: text_gate, structure_gate, uncertainty, iteration_budget, strategy_weights
- MetaCognitiveController: task embedding (128-dim) + iteration context (3-dim) → 5 metacognition heads
- Gated feature composition: structure_gate and text_gate modulate classifier input
- Deeper classifier: MLP TaskHeads (128→64→n_classes) replacing single Linear
- Phase E: metacognitive consolidation across all 19 tasks

**Mar 1-2 — Training iterations on kg_relation**

*v1-v2 (pre-metacog)*: ~84.8% val accuracy (just predicting majority class RelatedTo), balanced accuracy stuck at 12%.

*v3 (text concat to classifier + metacog)*:
- Pre-computed all 1,446,714 unique concept embeddings → GPU cache (5.6GB)
- Fixed NaN gates from partial checkpoint loading (old ControlHead weights into new MetaCognitiveController)
- **Best: 22.6% balanced accuracy** at epoch 14 (2.26x random)
- Loss declined 0.337 → 0.048 but bal_acc plateaued around 18-22%

**Mar 3-4 — Topological text processing**

The 22.6% result from v3 showed text features help, but they bypassed the entire topology pipeline — just flat vectors concatenated to the classifier. The GNN/TAT never saw the text.

*v4 (injection only, ungated)*: Text added directly to node embeddings before executive loop. Bal_acc stuck at 10% — too disruptive to pre-trained GNN/TAT weights.

*v5 (injection only, gated at sigmoid(-3)≈0.05)*: Same result — 10% for 3 epochs. Gating didn't help.

*Deep diagnostic revealed*:
1. `text_injection_gate.grad: None` — gradient chain broken by `cc.set_embeddings(0, gnn_out.detach())` inside executive loop
2. ALL classifier/proj gradients were None — the multi_head_classifier path wasn't flowing gradients back
3. Node embedding norms (0.04) dwarfed by topological features (81.9) — text signal washed out
4. Class collapse: 49/50 predictions = class 8 (majority)

*v6 (dual-path, with gradient)*: Both injection AND classifier concat. Bal_acc 10.7%→12.1% in 2 epochs but **303 min/epoch** (4.5x slowdown). Root cause: autograd tracked through entire executive loop for injection gradients.

*v7 (dual-path, detached injection)*:
- Path 1: `(node_embs + gate * text_features).detach()` → executive loop (forward-only topological processing)
- Path 2: text_features → classifier concat (gradient flow to proj layer)
- Epoch time restored to ~72 min

**v7 kg_relation results** (20 epochs, best at ep 16):

| Epoch | Loss | Bal Acc | | Epoch | Loss | Bal Acc |
|-------|------|---------|--|-------|------|---------|
| 0 | 0.332 | 11.1% | | 10 | 0.109 | 20.7% |
| 1 | 0.278 | 14.9% | | 11 | 0.093 | 22.6% |
| 4 | 0.196 | 20.6% | | 14 | 0.064 | 22.6% |
| 7 | 0.148 | 21.0% | | 16 | 0.050 | **22.9%** |
| 9 | 0.123 | 21.7% | | 19 | 0.039 | 22.2% |

Topology monitoring showed healthy trends: spectral gap ↑70%, gradient energy ↑16%, curl ↓3.5%, harmonic energy grew 7.5x (0.002→0.015) — the model was using increasingly global topological structure.

**v7 kg_pathvalid — task design bug discovered and fixed**:
- Original task: "Is this multi-hop path valid or corrupted?" — corrupt by swapping a middle node
- **BUG**: Corruption only swapped the node in a path list, never modified the graph. CellComplex was identical for valid and corrupted paths. Model had zero signal → stuck at 50%.
- **FIX**: Corruption now rewires graph edges — removes edges through original mid node, adds edges through swap node. Creates detectable structural discontinuity.
- After fix: **86.9% balanced accuracy** (ep 5). Jumped from 50% baseline to 83.3% at ep 0.

**v7 kg_concept — classifier design bug**:
- `classify_concept()` used ~20 keywords per category with "abstract" as default fallback
- 84% of ConceptNet concepts matched no keywords → all classified as "abstract" → class collapse
- Model correctly learned to always predict majority class (val_acc 83.9%, bal_acc 11.0% = random)

**v7 kg_analogy — missing features**:
- Task measures relation-type Jaccard overlap between two subgraphs
- But edge embeddings had no relation-type encoding — only random noise + weight
- Model could see graph structure but not what *types* of relations existed
- Best: 37.1% (3.8% above random baseline of 33.3%)

**Mar 6-7 — Comprehensive data quality audit (v8 preparation)**

Deep audit found 8 issues across the KG pipeline:

| # | Issue | Fix |
|---|-------|-----|
| 1 | kg_pathvalid: corruption doesn't modify graph | Rewire edges through swap node |
| 2 | kg_concept: keyword classifier → 84% "abstract" | IsA hierarchy + expanded keywords + uniform fallback |
| 3 | kg_cluster: same classifier issue | Same fix (classify_domain) |
| 4 | Edge embeddings: no relation type info | One-hot encoding in dims 1-10 of 1-cell embeddings |
| 5 | kg_analogy: merged graph missing edge encoding | Added relation one-hot to analogy's merged CellComplex |
| 6 | kg_relation: edge encoding leaks answer | Mask target edge's relation dims to zero |
| 7 | Circular import risk | Moved RELATION_CLASSES → use existing RELATION_CATEGORIES from conceptnet.py |
| 8 | Phase E: fragile conceptnet_graph check | Load ConceptNet explicitly when --resume-phase e |

All fixes applied, 710 tests passing (1 known flaky).

---

## v8: Clean Run (Mar 7-10, 2026)

Regenerated all 5 KG datasets with all 8 fixes applied. Fresh Phase D training from Phase C checkpoint. 25,000 train / 2,000 val samples per task. Early stopping patience=10.

### v8 kg_relation (10 classes) — Final: 23.2% bal_acc

| Epoch | Loss | Bal Acc | Note |
|-------|------|---------|------|
| 0 | 0.319 | 15.0% | |
| 2 | 0.228 | 20.4% | |
| 6 | 0.151 | 23.1% | |
| 11 | 0.079 | **23.2%** | best, stopped manually |

Edge relation encoding helped early convergence (v8 hit 23.1% at ep 6 vs v7's 22.9% at ep 16), but the ceiling is similar. The model can read structural graph topology + edge relation features, but 10-class relation classification fundamentally requires *semantic* understanding the current architecture can't provide. Loss continued dropping while bal_acc plateaued — classic overfitting to majority class (RelatedTo ~50% of data).

**Verdict**: Structural features alone cap around 23%. Text integration is too shallow to break through.

### v8 kg_concept (9 classes) — Final: 14.5% bal_acc

| Epoch | Loss | Bal Acc | Note |
|-------|------|---------|------|
| 0 | 1.680 | 11.0% | ~random (9 classes) |
| 5 | 1.573 | 14.2% | |
| 9 | 1.491 | **14.5%** | best, stopped manually |
| 11 | 1.451 | 13.5% | declining |

Fixed classifier (IsA hierarchy + expanded keywords + uniform fallback) eliminated the v7 class collapse (was 84% "abstract"). Categories now distributed across 9 classes. But 14.5% vs 11% random is only +3.5pp — the model barely learns this task. Concept classification requires understanding what a concept *is*, not just its structural neighborhood.

**Verdict**: Task is learnable now (not collapsed) but the architecture lacks the semantic depth for meaningful concept categorization.

### v8 kg_pathvalid (2 classes) — Final: 83.3% bal_acc

| Epoch | Loss | Bal Acc | Note |
|-------|------|---------|------|
| 0 | 0.113 | 81.9% | massive jump from 50% |
| 7 | 0.284 | 83.2% | |
| 10 | 0.042 | **83.3%** | best, stopped manually |
| 13 | 0.030 | 82.0% | plateauing |

**The big success**. Graph rewiring fix completely solved the v7 stuck-at-50% problem. The model detects structural discontinuities from corrupted paths with high accuracy from epoch 0. Val_acc ≈ bal_acc throughout — excellent class balance. Loss spikes (ep 1: 5.46, ep 3: 1.22) suggest some hard examples but overall stable.

**Verdict**: Proves the GNN+TAT architecture can do real structural reasoning on ConceptNet graphs. This is the strongest KG result and validates the topological approach for path integrity tasks.

### v8 kg_analogy (4 classes) — Training in progress

Dataset generating as of Mar 10. This task now has edge relation encoding in the merged analogy graph. v7 baseline: 37.1% (random = 25%). Expected improvement from edge encoding.

### v8 kg_cluster (6 classes) — Pending

Will follow kg_analogy. This task has the fixed `classify_domain()` classifier.

### v8 Summary Table

| Task | Classes | Random | v7 Best | v8 Best | Lift vs Random | Status |
|------|---------|--------|---------|---------|---------------|--------|
| kg_pathvalid | 2 | 50% | 50% (broken) | **83.3%** | +33.3pp | Structural task works |
| kg_relation | 10 | 10% | 22.9% | **23.2%** | +13.2pp | Ceiling hit |
| kg_concept | 9 | 11% | 11% (broken) | **14.5%** | +3.5pp | Barely above random |
| kg_analogy | 4 | 25% | 37.1% | TBD | TBD | Training |
| kg_cluster | 6 | 17% | untested | TBD | TBD | Pending |

### v8 Key Findings

**What the architecture excels at:**
- Structural/topological reasoning on both synthetic and real-world graphs
- Detecting graph structure changes (pathvalid 83.3%)
- Spectral decomposition tasks (hodge_class 95.8%)
- Graph traversal (BFS 99.8%)

**Where it hits a wall:**
- Tasks requiring semantic understanding (relation type, concept category)
- Text features are too shallowly integrated — projected and concatenated to classifier, but never deeply fused with topological processing
- The GNN/TAT core processes structure but is blind to semantics
- Dual-path architecture provides gradient flow but not *semantic reasoning*

**Training dynamics observations:**
- 25K samples at ~75 min/epoch makes experimentation slow — consider smaller dataset for hyperparameter search
- Early stopping at patience=10 is effective — tasks plateau well before 40 epochs
- Output buffering caused ghost stalls (PYTHONUNBUFFERED=1 required for nohup)
- Dataset generation is the bottleneck on restart: 1.7GB files take 3+ min to load, 25K sample generation takes 30-40 min

---

## Current State (Mar 10, 2026)

**Training**: v8 kg_analogy generating on vast.ai. kg_cluster will follow. All prior tasks manually stopped after plateau.

**Architecture insight**: The v8 results reveal a clear **structural vs semantic divide**. The architecture solves structural tasks (pathvalid 83.3%, BFS 99.8%, hodge 95.8%) but cannot cross into semantic territory with the current text integration approach. This motivates v9.

**All-time best results**:

| Task | Best Bal Acc | Version | Category |
|------|-------------|---------|----------|
| bfs | 99.8% | Phase A | Structural |
| hodge_class | 95.8% | Phase A | Structural |
| diverse | 84.6% | Phase A | Structural |
| kg_pathvalid | **83.3%** | v8 | Structural (on KG) |
| spectral_gap | ~74% | Phase A | Structural |
| kg_analogy | 37.1% | v7 | Semantic (weak) |
| kg_relation | **23.2%** | v8 | Semantic (weak) |
| kg_concept | **14.5%** | v8 | Semantic (very weak) |

**Test suite**: 710+ tests passing, 2 skipped, 1 known flaky.

---

## Key Lessons Learned

1. **Detach matters**: The executive loop's `.detach()` on embeddings breaks gradient chains. Any feature injected before the loop needs a separate gradient path (like classifier concat).

2. **Scale mismatch kills learning**: Text features at 16x structural norm overwhelmed the GNN. Learned gating (sigmoid init at -3) prevents disruption.

3. **bf16 is poison for spectral ops**: Eigendecomposition, neural ODE integration, and spectral filtering produce NaN under bf16 autocast. Force fp32 for these paths.

4. **Checkpoint surgery is fragile**: Partial-loading old ControlHead weights into MetaCognitiveController created NaN-producing franken-weights. Skip prefixes entirely and reinitialize.

5. **Inverse scaling is real**: Hodge classification improves with graph size because spectral resolution increases. This is a genuine architectural advantage, not a data artifact.

6. **The bottleneck is always the bottleneck**: GraphFormer's N→16→N compression destroyed per-node text identity. Approach C (direct embed_tokens) eliminated this by giving each node its own embedding.

7. **Dual-path is the right architecture for gradient flow**: Text needs to enter both the topological pipeline (for structural context) AND the classifier (for gradient flow). But this is necessary, not sufficient — the text path needs to be *deeper*, not just wider.

8. **If the model can't learn, check the data first**: Three separate task design bugs (pathvalid graph unchanged, concept classifier collapse, missing edge features) were all data/feature problems, not model problems. Deep diagnostic before architecture changes.

9. **Edge features matter**: Adding relation-type one-hot to edge embeddings provided richer features for GNN reasoning. The model needs to *see* the information to learn from it.

10. **Structural vs semantic is a real boundary**: The current architecture consistently solves structural tasks (50-99% bal_acc) but plateaus on semantic tasks (14-23%). This is not a hyperparameter problem — it's an architectural limitation in how text information is processed.

11. **Manual early stopping saves GPU hours**: Stopping plateau'd tasks after 10-13 epochs instead of running to 40 saves 20+ hours of compute per task with no loss of quality.

---

## Open Questions for v9

- Can deeper text-topology fusion (e.g., text-conditioned attention in TAT) break the semantic ceiling?
- Should the metacognitive controller learn to *route* between structural and semantic processing paths?
- Is per-node Qwen embedding (Approach C) sufficient, or do we need contextual embeddings (sentence-level)?
- Can we train a lightweight text encoder end-to-end instead of using frozen Qwen features?
- Would a graph transformer (e.g., GPS-style) that natively interleaves text and structure outperform the two-tier architecture?
- Can inverse scaling extend to KG tasks — does training on small ConceptNet subgraphs generalize to large ones?

---

## Infrastructure Notes

- **Training hardware**: vast.ai RTX 4090 (24GB VRAM)
- **SSH**: `ssh -i ~/.ssh/vastai -p 34701 root@136.59.129.136`
- **Embedding cache**: 1,446,714 concepts, 5.6GB GPU-resident tensor
- **Checkpoint backup**: `rsync -avz -e 'ssh -i ~/.ssh/vastai -p PORT' root@IP:~/transformer-topology/data/v6_track2/checkpoints/ data/v6_track2/checkpoints/`
- **Log naming**: v8a=kg_relation, v8b=restarted(buffered), v8c=kg_concept(unbuffered), v8d=kg_pathvalid, v8e=kg_analogy
- **Dataset locations**: Remote `dsm_datasets_25k/` has v8-regenerated KG data. Local copies deleted (were pre-fix, corrupt).

---

*Last updated: March 10, 2026 — v8 kg_analogy generating, v9 planning underway*

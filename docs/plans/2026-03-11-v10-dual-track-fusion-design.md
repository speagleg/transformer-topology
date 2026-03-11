# v10 Dual-Track Fusion Design

**Date**: March 11, 2026
**Goal**: Redesign the executive loop and KG task suite so that topological reasoning over knowledge graphs genuinely outperforms probabilistic LLMs on tasks requiring structural + semantic fusion.

## Background & Motivation

v9 results revealed that shallow text integration (TextReasoningHead + TextEdgeEncoder) is insufficient. All 5 KG tasks plateau at or near random chance:

| Task | Classes | Best Bal Acc | Status |
|------|---------|-------------|--------|
| kg_relation | 10 | 25.5% | Only task showing learning above chance |
| kg_concept | 9 | 14.5% | Majority-class trap (class 8 = 86% of data) |
| kg_pathvalid | 2 | 50.0% | Coin flip — structural corruption doesn't need text |
| kg_analogy | 3 | 37.1% | Barely above random (33%) |
| kg_cluster | 6 | N/A | Never reached |

### Root Causes

1. **Text features detached from reasoning loop**: `.detach()` on Path 1 blocks gradients. Text never influences GNN/TAT learning.
2. **Narrow classifier input**: Only query/target node text reaches classifier. N-2 nodes invisible.
3. **Identical loop iterations**: Both iterations do the same thing → 78% curl energy (information cycling, not progressing).
4. **Near-zero spectral gap (0.03)**: Information bottleneck between structure and text modalities.
5. **Weak text encoder**: embed_tokens mean-pooling is contextless word2vec. "bank" near "river" = "bank" near "money".
6. **Poor task design**: kg_concept/kg_cluster have noisy keyword labels. kg_pathvalid corruption is structural not semantic. kg_analogy uses Jaccard overlap instead of true analogy.

### Computation Graph Topology Evidence

Topology observer readings were stable across all 16 kg_relation epochs:
- Spectral gap: 0.029-0.030 (near bottleneck)
- Gradient energy: ~21% (forward information flow)
- Curl energy: ~78% (cyclic information, wasted)
- Harmonic energy: ~0.5% (no dead subnetworks)
- Zero adaptation across training — topology never changed

## Architecture: Dual-Track Executive Loop

### Core Idea

Replace 2 identical iterations with 2 **asymmetric** iterations, each with a clear purpose:

- **Iteration 1 (Structural)**: Pure topology reasoning. GNN + wave dynamics + TAT. No text.
- **Iteration 2 (Fusion)**: Cross-modal reasoning. Text-conditioned GNN + cross-attention to Qwen embeddings + TAT with semantic bias.

The ControlHead learns `fusion_weight` — a meta-cognitive gate that decides how much iteration 2 overrides iteration 1. Structural tasks learn fusion_weight→0. KG tasks learn fusion_weight→1.

### Component 1: Qwen Contextual Encoder

Replace embed_tokens-only path with frozen 4-layer Qwen encoder.

```
concept strings → Qwen tokenizer (max 16 subwords)
                → embed_tokens (frozen)
                → 4 transformer layers (frozen)
                → mean-pool over subword positions → (N, 2048)
                → Linear(2048, 32) + LayerNorm → (N, 32) contextual text embeddings
```

- VRAM: ~3GB (embed_tokens 0.6GB + 4 layers ~2.4GB)
- Cached per-concept at dataset load time (zero Qwen cost during training)
- Provides contextual embeddings where "bank" differs based on surrounding subwords

### Component 2: Iteration 1 — Structural Encoding

Identical to current architecture:
1. GNN message passing with structural features only
2. Wave dynamics (ensemble: chebyshev, wave_cosine, heat, identity)
3. TAT spatial attention
4. Produces: `h_struct (N, 32)` + ControlSignal (including `fusion_weight`)

No text involved. This is the "conscious mind thinking about structure."

### Component 3: Iteration 2 — Cross-Modal Fusion

Three new sub-components:

**A. Text-conditioned GNN message passing:**
- Each edge message modulated by learned text similarity of source/target:
  ```
  text_sim = sigmoid(Linear([text_src, text_dst, text_src * text_dst]))  → scalar
  msg = standard_msg * (1 + text_sim)  # amplify messages between semantically related nodes
  ```
- Gradients flow through text_sim → back to projection layer
- ~2K parameters

**B. Cross-attention block (new):**
- Standard multi-head cross-attention (4 heads):
  ```
  Q = Linear(h_struct)     # (N, 32) from iteration 1
  K = Linear(h_text)       # (N, 32) cached Qwen contextual embeddings
  V = Linear(h_text)       # (N, 32)
  h_cross = MultiHeadAttn(Q, K, V)  → (N, 32)
  ```
- LayerNorm + residual: `h_fused = LayerNorm(h_struct + h_cross)`
- ~8K parameters

**C. TAT with semantic context:**
- TAT processes `h_fused` instead of raw GNN output
- Produces: `h_final (N, 32)`

**No wave dynamics in iteration 2.** Cross-modal fusion, not spectral processing.

### Component 4: Fusion Gating

```
h_out = (1 - fusion_weight) * h_struct + fusion_weight * h_final
```

- `fusion_weight`: sigmoid scalar from ControlHead, initialized at bias=0 (sigmoid=0.5)
- Structural tasks: model learns fusion_weight → 0 (skip iteration 2 entirely)
- KG tasks: model learns fusion_weight → 1 (use cross-modal fusion)
- This is the meta-cognitive decision

### Component 5: Attention-Based Readout

Replace fixed query/target extraction with learned attention over all nodes.

```
task_emb = task_embedding_table[task_id]           # (32,)
readout_query = Linear([task_emb, h_out[query], h_out[target]])  → (32,)

# Attend over all N node embeddings
Q = readout_query                                  # (1, 32)
K = V = h_out                                      # (N, 32)
context = softmax(Q @ K^T / sqrt(32)) @ V          → (32,)

# Classifier input
combined = [
    h_out[query],     # (32,) local query context
    h_out[target],    # (32,) local target context
    context,          # (32,) global attention-pooled context
    topo_features,    # hodge(3) + wave_energy(1) = (4,)
    fusion_weight,    # (1,) meta-cognitive signal
]
# Total: 101 dims
```

Classifier: 2-layer MLP (101 → 128 → 64 → num_classes) with LayerNorm/GELU/Dropout. Multi-head classifier (per-task output layer) retained.

**Removed from classifier input:** strategy_weights, uncertainty, text_gate, structure_gate, text_reasoning features, separate text features. All replaced by cleaner architecture.

### Curl Regularization

- Compute curl energy ratio after iteration 1
- Penalty: `curl_loss = max(0, curl_energy_ratio - 0.5) * 0.01`
- Threshold 0.5: some cycling is healthy, but majority should flow forward
- Added to total loss

## Reasoning Task Suite

### Design Principle

Each task targets a specific reasoning capability where topology + semantics together outperform either alone. Tasks reflect real-world reasoning patterns that probabilistic LLMs struggle with.

### Task 1: `kg_relation` (kept)
- **Reasoning type**: Semantic edge classification
- **Classes**: 10 (IsA, HasA, PartOf, UsedFor, CapableOf, AtLocation, Causes, HasProperty, RelatedTo, Other)
- **Structure signal**: Neighborhood context, degree patterns
- **Text signal**: Concept meaning disambiguates relations with identical structure
- **No changes** from v9

### Task 2: `kg_transitive` (new, replaces kg_concept)
- **Reasoning type**: Multi-hop transitive inference
- **What**: Given chain A→B→C with known relations, predict whether missing direct edge A→C holds and what relation it would have
- **Classes**: 5 (IsA, HasA, PartOf, Causes, none)
- **Examples**:
  - "dog IsA mammal" + "mammal IsA animal" → "dog IsA animal" (valid, class=IsA)
  - "dog AtLocation park" + "park PartOf city" → "dog PartOf city" (invalid, class=none)
- **Structure signal**: 2-hop path exists
- **Text signal**: Whether transitivity applies for this relation pair
- **Fusion requirement**: High — structure says "path exists", text says "this transfer is valid"

### Task 3: `kg_consistency` (new, replaces kg_pathvalid)
- **Reasoning type**: Logical consistency checking
- **What**: Given a subgraph, detect whether it contains a semantic contradiction
- **Classes**: 2 (consistent=1, contradiction=0)
- **Generation**: 50% corruption rate. Corrupted graphs replace one edge's target with semantically incompatible concept (different IsA branch) while preserving graph structure
- **Examples**:
  - Consistent: "robin IsA bird" + "bird CapableOf fly" + "robin HasA wings"
  - Contradiction: "robin IsA bird" + "bird CapableOf fly" + "robin IsA fish"
- **Structure signal**: Graph structure identical for both (same degrees, connectivity)
- **Text signal**: Semantic incompatibility reveals contradiction
- **Fusion requirement**: High — requires tracing implications through 2-3 hops

### Task 4: `kg_analogy` (redesigned)
- **Reasoning type**: Structural analogy via subgraph isomorphism + semantic mapping
- **What**: Given two small subgraphs (3-5 nodes each), classify whether they represent the same relational pattern with different content
- **Classes**: 3 (analogous, partial, not_analogous)
- **Generation**: "Analogous" pairs share same relation-type sequence along matched paths AND semantically compatible role filling
- **Examples**:
  - Analogous: "teacher→teaches→student→studies→subject" ~ "coach→trains→player→competes→sport"
  - Not analogous: "teacher→teaches→student" ~ "dog→IsA→animal"
- **Structure signal**: Subgraph isomorphism detection
- **Text signal**: Semantic role compatibility verification
- **Fusion requirement**: Very high — both structure AND semantics must align

### Task 5: `kg_causal_chain` (new, replaces kg_cluster)
- **Reasoning type**: Causal chain coherence
- **What**: Given a path of 3-5 nodes connected by Causes/UsedFor/CapableOf edges, classify coherence level
- **Classes**: 3 (coherent, broken, incoherent)
- **Examples**:
  - Coherent: "rain → Causes → flood → Causes → damage"
  - Broken: "rain → Causes → flood → Causes → happiness" (each local step plausible, global chain fails)
  - Incoherent: "rain → Causes → pencil → Causes → democracy"
- **Structure signal**: Path structure, edge types
- **Text signal**: Semantic plausibility of each causal step
- **Fusion requirement**: Very high — "broken" class requires multi-hop topological reasoning + semantic plausibility. LLMs often accept plausible-but-broken chains.

### Task Summary

| Task | Reasoning | Classes | Structure | Text | Fusion |
|------|-----------|---------|-----------|------|--------|
| kg_relation | Edge classification | 10 | Neighborhood | Concept meaning | Medium |
| kg_transitive | Transitive inference | 5 | Path existence | Relation transferability | High |
| kg_consistency | Contradiction detection | 2 | Graph validity | Semantic compatibility | High |
| kg_analogy | Structural analogy | 3 | Isomorphism | Role compatibility | Very high |
| kg_causal_chain | Causal coherence | 3 | Chain structure | Step plausibility | Very high |

## Training Strategy

### Two-Phase Curriculum

**Phase A — Structural warmup (5 epochs):**
- Train on BFS, hodge_class, diverse only
- `fusion_weight` forced to 0 (iteration 2 disabled)
- Verifies iteration 1 works from Phase C checkpoint
- Qwen layers loaded but unused

**Phase B — Joint KG training (30 epochs per task):**
- All 5 KG tasks trained sequentially
- Both iterations active, fusion_weight learned
- 20% structural replay
- Curl penalty active
- Early stopping: patience 15 on balanced accuracy

### Optimizer Groups

| Group | Components | Learning Rate |
|-------|-----------|---------------|
| 1 | GNN, TAT, wave dynamics | 0.0003 |
| 2 | Cross-attention, Qwen projection | 0.001 |
| 3 | Attention readout, classifier heads | 0.001 |

### Loss

`total_loss = focal_loss(gamma=2.0) + 0.01 * curl_penalty`

No contrastive loss, no class weights (focal handles imbalance).

### Checkpoint Strategy

- Load Phase C checkpoint for GNN/TAT/wave weights
- Fresh init: cross-attention, attention readout, Qwen projection, new classifier heads
- `_load_state_filtered()` handles mismatches

### Success Criteria

| Task | Target Bal Acc | Baseline |
|------|---------------|----------|
| kg_relation | > 35% | 25.5% (v9) |
| kg_transitive | > 40% | new task |
| kg_consistency | > 70% | new task (binary) |
| kg_analogy | > 40% | 37.1% (v7) |
| kg_causal_chain | > 45% | new task |

Meta-cognitive validation:
- `fusion_weight > 0.3` on KG tasks (subconscious consulted)
- `fusion_weight < 0.1` on structural tasks (subconscious ignored)

## VRAM Budget (24GB RTX 4090)

| Component | v9 | v10 |
|-----------|-----|-----|
| Qwen embed_tokens cache | 5.6GB | — (replaced) |
| Qwen embed_tokens + 4 layers | — | ~3.0GB |
| Contextual embedding cache | — | ~1.5GB (est.) |
| Model params (GNN/TAT/wave) | ~4.3MB | ~4.3MB |
| Cross-attention block | 0 | ~0.1MB |
| Attention readout | 0 | ~0.05MB |
| Activations + optimizer | ~3.7GB | ~4.5GB |
| **Total** | ~9.6GB | ~9.4GB |

Net neutral on VRAM. Qwen layers replace the 5.6GB flat embedding cache with a 3GB model + 1.5GB contextual cache.

## Files Changed

| Order | File | Action |
|-------|------|--------|
| 1 | `src/llm/qwen_contextual_encoder.py` | Create — 4-layer Qwen encoder + projection + caching |
| 2 | `src/reasoning_loop/cross_attention.py` | Create — cross-attention block for iteration 2 |
| 3 | `src/reasoning_loop/attention_readout.py` | Create — task-conditioned attention readout |
| 4 | `src/reasoning_loop/executive_loop.py` | Refactor — dual-track iterations, fusion gating |
| 5 | `src/gnn_executive/higher_order_gnn.py` | Add text-conditioned edge messages |
| 6 | `src/training/batch_utils.py` | Update forward path for new architecture |
| 7 | `src/benchmarks/conceptnet_tasks.py` | New task generators (transitive, consistency, causal_chain), redesign analogy |
| 8 | `src/benchmarks/benchmark_dataset.py` | Update TASK_REGISTRY |
| 9 | `src/computation_graph/diagnostics.py` | Add fusion_weight tracking, per-iteration energy |
| 10 | `scripts/run_dsm_curriculum.py` | Update Phase B training, optimizer groups |
| 11 | `config/v10_dual_track.yaml` | New config |
| 12 | Tests | All new components + integration |

**Removed**: `src/llm/text_reasoning_head.py`, `src/llm/text_edge_encoder.py` (v9 modules replaced by deeper integration). Note: `_load_state_filtered()` will skip their weights when loading v9 checkpoints.

## Clarifications (from spec review)

### C1: Qwen projection caching and gradient flow

Cache the **raw 2048-dim** Qwen outputs per-concept (frozen layers, deterministic). Apply the learned `Linear(2048, 32) + LayerNorm` projection **at forward time**. This means:
- Cache: `concept_string → (2048,)` tensor on GPU (~40MB for ~10K unique concepts per task)
- Forward: load (N, 2048) from cache, project to (N, 32) — gradients flow through projection layer
- VRAM: ~40MB cache (not 1.5GB as originally estimated), plus (N, 2048) activations per batch

### C2: No .detach() between iterations

Remove the `.detach()` between iteration 1 and iteration 2. Iteration 2's GNN receives live embeddings from iteration 1. This means:
- `fusion_weight` gradient flows back through both iterations
- The model can learn to adjust iteration 1's output to better serve iteration 2
- Risk: deeper gradient chain. Mitigate with gradient clipping (max_norm=5.0, already in config)

### C3: MetaCognitiveController replacement

Replace MetaCognitiveController with an extended ControlHead:
- **Keep**: frequency_gate, spatial_focus, confidence_weights, diffusion_time, wave_damping, filter_weights (for wave ensemble)
- **Add**: `fusion_weight` (sigmoid scalar, bias=0.0)
- **Remove**: text_gate, structure_gate, semantic_weight, uncertainty, strategy_weights, iteration_budget, task_embedding (128-dim)
- The attention readout has its **own** task embedding table (32-dim, separate from any controller embedding)
- ControlHead input: `embedding_dim + 2` (unchanged: +1 harmonic energy, +1 log(N)/log(100))

### C4: Phase A — fresh components frozen

During Phase A structural warmup:
- Cross-attention, attention readout, Qwen projection: **frozen** (excluded from optimizer)
- Only GNN/TAT/wave parameters in optimizer
- fusion_weight forced to 0 via direct override (not learned)
- This ensures Phase A validates iteration 1 in isolation

### C5: kg_analogy generation algorithm

Use NetworkX `is_isomorphic()` with edge-type matching for small subgraphs (3-5 nodes):
1. Extract subgraph A (3-5 nodes) from ConceptNet
2. Record its relation-type sequence along paths (the "pattern")
3. For **analogous**: find subgraph B with matching relation-type sequence via BFS from random seed, accept if `nx.is_isomorphic(A, B, edge_match=relation_type_match)`. Retry up to 20 times.
4. For **partial**: find subgraph B where relation-type overlap > 0.3 but isomorphism fails
5. For **not_analogous**: random subgraph B with different node count or no relation overlap
6. Fallback: if analogous pair not found in 20 attempts, skip this sample

### C6: kg_causal_chain "broken" class generation

Use ConceptNet edge weights as plausibility proxy:
1. **Coherent**: extract existing Causes/UsedFor chain of 3-5 nodes from ConceptNet (all edges actually exist, weight > 1.0)
2. **Broken**: take a coherent chain, replace the LAST node with a concept that is plausibly connected to the penultimate node (has a Causes/UsedFor edge from it) but semantically unrelated to the chain start. Plausibility of each local step verified by edge existence in ConceptNet.
3. **Incoherent**: replace 2+ nodes with random concepts (no ConceptNet edges between them)
4. Fallback: if chain not found, retry from different seed concept

### C7: Curl penalty may need higher weight

Start at 0.01. If after 5 epochs curl energy is still >60%, increase to 0.1. The asymmetric iteration structure should do most of the work; the penalty is a safety net. Log curl energy per epoch for tuning.

### C8: kg_consistency success target raised to 70%

Binary task with rich fusion signal — 65% was conservative. Raise to 70%.

## Comparison: v9 vs v10

| Aspect | v9 | v10 |
|--------|-----|-----|
| Text encoder | embed_tokens only (word2vec) | 4 Qwen layers (contextual) |
| Text in GNN | TextEdgeEncoder side channel | Text-conditioned edge messages |
| Text in TAT | None | Cross-attention to Qwen embeddings |
| Loop structure | 2 identical iterations | Asymmetric: structural → fusion |
| Gradient flow | Detached in Path 1 | Full flow through cross-attention |
| Classifier input | 425 dims (query/target + text + metacog) | 101 dims (attention readout + topo) |
| Classifier readout | Fixed query/target | Learned attention over all nodes |
| Fusion control | text_gate + structure_gate + semantic_weight | Single fusion_weight |
| Curl management | None | 0.01 * curl_penalty |
| KG tasks | 5 (noisy labels, structural corruption) | 5 (reasoning-focused, fusion-required) |

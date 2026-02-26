# Phase 7: Meta-Cognition via ConceptNet Grounded Reasoning

## Context

Track 2 (v6) proved the conscious executive works: 84.6% diverse, 99.4% BFS, 80.8% graph_completion.
But `semantic_weight` learned to be 0.000 on every task -- the subconscious (LLM) was never consulted.
Reason: all 14 existing tasks are solvable from graph structure alone. The LLM had nothing to contribute.

Phase 7 adds tasks that **require multi-modal fusion** -- the GNN sees graph structure but
cannot read text, while the LLM reads concept text but cannot see topology. Only fusion succeeds.

## Goal

Prove the meta-cognition loop works: the GNN executive (conscious mind) learns **when** to consult
the LLM (subconscious) by observing that semantic_weight rises on knowledge graph tasks while
staying near zero on structural tasks. This is the first genuine test of task-conditional
conscious/subconscious interaction.

## Architecture

### Data Flow

```
ConceptNet subgraph (20-80 nodes)
    |
    +--- Structural features --> GNN Executive --> ControlSignal
    |    (degree, clustering,       |                  |
    |     BFS distance, topology)   |          semantic_weight
    |                               |                  |
    |                               v                  v
    +--- Concept text ---------> TopoBridge --> Qwen 2.5-3B --> semantic_bias
         ("dog", "animal",     (prefix + hard     (frozen +       (NxN attention
          "running", ...)       text tokens)       LoRA)           bias matrix)
                                                      |
                                                      v
                                              TAT spatial attention
                                          (structural + semantic fused)
```

### Modality Separation

- **GNN sees:** structural node features only (degree, clustering coefficient, BFS distance,
  topology type indicators). Cannot read text.
- **LLM sees:** concept text via hard tokens concatenated with TopoBridge prefix tokens.
  Processes both structural context (from prefix) and semantic content (from text).
- **TopoBridge mediates:** prefix tokens encode structural context, text tokens provide
  semantic content. Qwen processes both. NodeExtractor attends to prefix positions.
- **ControlHead decides:** semantic_weight scalar [0,1] controls how much the LLM's
  semantic_bias influences TAT spatial attention.

### TopoBridge Text Channel (New)

Current flow (no text):
```
node_emb (N,32) -> NodeProjector -> topo_memory (N,2048)
                -> PrefixGenerator -> prefix_tokens (16,2048)
                -> QwenGraphBackend.forward(prefix, topo_memory, task_name)
                -> Qwen frozen -> hidden_states
                -> NodeExtractor -> semantic_out (N,32)
```

New flow (with text):
```
node_emb (N,32) -> NodeProjector -> topo_memory (N,2048)
                -> PrefixGenerator -> prefix_tokens (16,2048)

concept_texts -> Qwen tokenizer -> text_tokens (T tokens)
             -> Qwen embedding layer -> text_emb (T,2048)

input_sequence = [prefix_tokens, text_emb]  # (16+T, 2048)

-> QwenGraphBackend.forward(input_sequence, topo_memory, task_name)
-> Qwen frozen + LoRA -> hidden_states (16+T, 2048)
-> NodeExtractor attends to hidden_states[:16] (prefix positions only)
-> semantic_out (N,32)
```

Text prompt format: `"Graph nodes: 0=dog, 1=animal, 2=park, 3=running. Task: kg_relation"`
~60 tokens for a 30-node subgraph. Total sequence ~76 tokens (16 prefix + 60 text).

Mock backend receives `node_texts` but ignores them (returns random features as before).

## Data: ConceptNet Pipeline

### Source
ConceptNet 5.7 assertions CSV (~3GB). Filter to English concepts (`/c/en/`), weight >= 1.0.
Build NetworkX graph. ~500K nodes, ~1.5M edges.

### Relation Categories (10)
IsA, HasA, PartOf, UsedFor, CapableOf, AtLocation, Causes, HasProperty, RelatedTo, Other

### Subgraph Extraction
1. Seed: random concept with degree >= 5
2. BFS expansion to 20-80 nodes (training: 20-50, OOD test: 60-100)
3. Keep all internal edges
4. Require: connected, >= 3 distinct relation types, >= 1 valid task instance

### Dataset Scale
- 5000 training subgraphs, 500 validation, 500 test
- Each subgraph generates multiple task instances
- Estimated ~135K+ training samples total

### CellComplex Conversion
Reuse existing `graph_convert.py`. Node embeddings = structural features ONLY.
Concept text stored as `CellComplex.node_texts: list[str]` metadata attribute.

## Tasks (5 New Types, 18 Total)

### 1. Relation Type Prediction (`kg_relation`, 10 classes)
- **Input:** Subgraph + two highlighted nodes (query, target)
- **Output:** Relation type (IsA, HasA, PartOf, UsedFor, CapableOf, AtLocation, Causes,
  HasProperty, RelatedTo, Other)
- **Why fusion required:** "dog->animal" (IsA) and "dog->park" (AtLocation) have identical
  local structure but different relation types. GNN alone ~30-40%, LLM alone ~50-60%,
  fusion target 80%+.
- **Data:** Every internal edge is a sample. ~20 edges per 30-node graph. ~100K samples.

### 2. Masked Concept Category (`kg_concept`, 9 classes)
- **Input:** Subgraph with one node's text replaced by [MASK]. Others retain text.
- **Output:** Semantic category (animal, place, activity, object, emotion, person, food,
  body_part, abstract)
- **Why fusion required:** Same degree-5 hub topology, but neighbors "river, bridge, dam,
  lake, stream" = place vs "fur, paws, tail, bark, fetch" = animal. GNN sees identical
  structure. LLM reads neighbor text. Fusion combines both signals.
- **Data:** 1-3 masks per subgraph. ~10K samples.

### 3. Commonsense Path Validity (`kg_pathvalid`, 2 classes)
- **Input:** Subgraph + highlighted path (3-5 hops)
- **Output:** Binary -- valid reasoning chain or not
- **Why fusion required:** Valid: "rain->wet_ground->slippery->falling". Invalid:
  "rain->wet_ground->slippery->happiness" (swap one concept for random same-degree node).
  Path always exists structurally (GNN: 50%). LLM judges semantic coherence.
- **Data:** 2-5 paths per subgraph (half valid, half corrupted). ~15K samples.

### 4. Analogical Grounding (`kg_analogy`, 3 classes)
- **Input:** Two subgraphs side by side
- **Output:** Analogous / partially analogous / not analogous
- **Why fusion required:** "teacher->student->exam" ~ "coach->player->game" (analogous --
  same relational roles). "teacher->student->exam" !~ "parent->child->school"
  (not analogous -- different relational dynamics despite similar structure).
  GNN sees structural isomorphism. LLM judges semantic role alignment.
- **Data:** Pair subgraphs by structural similarity. ~5K samples.

### 5. Semantic Domain Clustering (`kg_cluster`, 6 classes)
- **Input:** Subgraph where nodes span 2-4 semantic domains
- **Output:** Per-node domain label (science, everyday, social, spatial, temporal, abstract)
- **Why fusion required:** GNN detects community structure (clusters). LLM identifies which
  community is which domain. Tests whether the executive can use the LLM for *labeling*
  structural patterns it already detects. Core meta-cognitive operation: "I see the clusters,
  but what do they mean?"
- **Data:** Construct subgraphs from multi-domain seeds. ~5K samples.

### Sample Counts Summary
| Task | Samples/subgraph | Training total |
|------|-----------------|----------------|
| kg_relation | ~20 | ~100,000 |
| kg_concept | 1-3 | ~10,000 |
| kg_pathvalid | 2-5 | ~15,000 |
| kg_analogy | 1-2 | ~5,000 |
| kg_cluster | ~1 | ~5,000 |
| **Total** | | **~135,000** |

## Curriculum: Phase D

### Prerequisites
- Track 2 Phase C checkpoint (structural + basic semantic tasks learned)
- Real Qwen 2.5-3B-AWQ compiled and loaded on vast.ai RTX 4090
- ConceptNet data preprocessed and subgraphs extracted

### Phase D Structure (40 epochs total)
Load Track 2 Phase C checkpoint. Swap mock LLM -> real Qwen.

1. `kg_relation` (10 epochs) -- relation type prediction (largest dataset, easiest fusion)
2. `kg_concept` (8 epochs) -- masked concept category
3. `kg_pathvalid` (8 epochs) -- commonsense path validity
4. `kg_analogy` (8 epochs) -- analogical grounding
5. `kg_cluster` (6 epochs) -- semantic domain clustering

### Training Configuration
- Replay: 20% Phase A structural + 10% Phase B/C semantic
- Feature replay buffer: active, classifier-only rehearsal on previous KG tasks
- Contrastive loss: weight=0.1 on semantic features (now should be meaningful)
- Topology observer: active (fixed), self-awareness features to ControlHead
- Class weights: per-task, max_weight_ratio=10
- Accumulation steps: 4 (effective batch size 4)
- Max grad norm: 5.0
- Optimizer: 3-group (GNN/TAT lr=1e-3, Qwen adapter lr=1e-4, TopoBridge lr=1e-4)

### Success Metrics

**Primary (semantic_weight):**
- structural tasks: semantic_weight stays near 0 (no regression)
- KG tasks: semantic_weight rises above 0.3 (subconscious consulted)
- task-conditional gating = genuine meta-cognition

**Secondary (accuracy):**
- KG tasks with real Qwen: 70%+ (fusion working)
- KG tasks with mock Qwen: <60% (confirming LLM is needed)
- Structural tasks: no degradation from Phase C levels

**Ablation:**
After training, evaluate with semantic_weight forced to 0:
- Accuracy drop on KG tasks quantifies subconscious contribution
- No accuracy drop on structural tasks confirms gate selectivity

## Files Summary

| Component | Files |
|-----------|-------|
| ConceptNet pipeline | `src/data/conceptnet.py` (download, preprocess, extract) |
| KG task generators | `src/benchmarks/conceptnet_tasks.py` (5 task types) |
| TopoBridge text channel | `src/llm/qwen_backend.py` (modified), `src/llm/topo_bridge.py` (modified) |
| CellComplex text metadata | `src/cell_complex.py` (add node_texts attribute) |
| Phase D curriculum | `scripts/run_dsm_curriculum.py` (add Phase D), `config/v7_metacognition.yaml` |
| Tests | `tests/test_data/test_conceptnet.py`, `tests/test_benchmarks/test_conceptnet_tasks.py` |
| Dataset precompute | `scripts/precompute_kg_datasets.py` |

## Future: Phase 8 (Executive Query Protocol)

After Phase 7 proves the subconscious helps, Phase 8 adds explicit query agency:
- New ControlSignal field: `llm_query_gate` (should I consult the LLM this iteration?)
- Executive decides WHEN to query, not just how much weight to give
- Reduces from 5 Qwen forward passes per sample to 1-2 (efficiency)
- True meta-cognition: "I need help with this step" vs "I've got this"

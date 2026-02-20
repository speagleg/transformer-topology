# Phase 6: Interleaved DSM-GNN Symbiotic Architecture

## Problem Statement

Phase 4c evaluation proved the GNN/TAT backbone is strong (96-99% path counting, 87% topo transfer on graph completion) but the Llama 3.2 1B integration is dead weight:
- LLM gate stays <9% across all checkpoints — model learned to bypass the LLM
- Ensemble wave filters collapsed to identity — spectral dynamics unused
- Phase A gate penalty (1.0) permanently suppressed the LLM pathway
- Graph completion regressed 8.8% after Phase C sequential training
- Analogical transfer at 18.5% (below random) — fundamentally broken task design

**Root cause**: The current architecture treats the LLM as an afterthought — the GNN solves the problem before the LLM gets involved, then a learned gate decides (correctly) not to use it.

**Solution**: Replace the frozen Llama with a distilled, fully trainable 250M Semantic Model that participates in every iteration of the executive reasoning loop via attention bias injection.

---

## Architecture: Interleaved Reasoning Loop

### Core Metaphor

- **GNN Executive** = conscious structured reasoning over graph topology
- **Distilled Semantic Model (DSM)** = subconscious pattern recognition providing semantic context
- Both systems process simultaneously, each iteration. The DSM shapes *where* the GNN attends; the GNN shapes *what* the DSM sees.

### Data Flow (per iteration)

```
1. GNN Executive reads graph state
   → control_signal (frequency_gate, spatial_focus, confidence, semantic_weight)
   → gnn_out (N x 32)

2. Wave Dynamics processes gnn_out
   → wave_out (N x 32)

3. TopoBridge Encoder
   → NodeProjector: Linear(32→1024) + LayerNorm → topo_memory (N x 1024)
   → PrefixGenerator: 8 learned queries attend over topo_memory → prefix (8 x 1024)

4. DSM Forward (250M, all trainable)
   ┌─────────────────────────────────────────────────────┐
   │ Layers 0-3:  self-attn [prefix | task_tokens]       │
   │ Layer 4:     self-attn + cross-attn ← topo_memory   │
   │ Layers 5-15: self-attn                              │
   └──────────────────────────┬──────────────────────────┘
   → semantic_hidden (seq_len x 1024)

5. TopoBridge Decoder
   → NodeExtractor: cross-attn(topo_memory, semantic_hidden)
   → Linear(1024→32) → semantic_out (N x 32)
   → W_semantic: Linear(32→N) → semantic_bias (N x N) pairwise

6. TAT with Semantic Bias
   For each TAT layer:
     Q = node_embs @ W_q
     K = node_embs @ W_k
     V = node_embs @ W_v
     attn_logits = Q @ K^T / sqrt(d)
     attn_logits += semantic_weight * semantic_bias   ← NEW
     attn = softmax(attn_logits) @ V

7. Integration
   output = confidence * tat_out + (1 - confidence) * gnn_out

8. Harmonic convergence check
```

### Key Design Decisions

1. **No llm_gate**. Replaced with `semantic_weight` — a soft scalar from ControlHead, never penalized. The model learns the right weight naturally.
2. **DSM runs every iteration** (avg 2-3), seeing progressively refined graph embeddings.
3. **Semantic signal enters via attention bias**, not additive blending. The GNN/TAT still reasons; the DSM shapes attention patterns.
4. **Fully trainable DSM** — no frozen weights, no LoRA. True co-evolution with GNN.

---

## Distilled Semantic Model (DSM) — 250M

### Architecture

| Parameter | Value |
|---|---|
| Layers | 16 |
| Hidden dim | 1024 |
| Attention heads | 16 |
| FFN dim | 4096 |
| Vocabulary | Llama tokenizer (shared) |
| Cross-attention port | Layer 4 |
| Total params | ~250M |
| All trainable | Yes |

### Why Not Frozen Llama?

| Property | Frozen Llama + LoRA | Distilled DSM |
|---|---|---|
| Params | 1.2B (50M trainable) | 250M (all trainable) |
| VRAM | ~6 GB | ~6 GB |
| Adapts to graph domain | Minimally (LoRA surface) | Fully (every layer) |
| Training cost/epoch | ~17 min (4090) | ~7 min (4090) |
| Knowledge | Full pretrained LM | Distilled from Llama |
| Symbiosis | One-way (GNN → LLM query) | Two-way (co-evolution) |

---

## Distillation Stage (Stage 0)

### Setup

```
Teacher: Frozen Llama 1B (bf16)
Student: DSM 250M (fp32, random init)
  - Embeddings initialized from Llama embeddings via PCA (2048→1024)
  - Cross-attention port at layer 4 (proportional to Llama's layer 8)

Loss:
  0.5 x MSE(student_hidden[layer_8], PCA(teacher_hidden[layer_8]))
  + 0.5 x KL(student_logits, teacher_logits, temperature=2.0)

Data:
  - 50K sentences from SlimPajama (general text)
  - 10K graph-description texts from our task prompt templates

Training:
  - LR: 3e-4, cosine annealing
  - Epochs: 10-20 (until validation loss plateaus)
  - Time: ~4 hrs on RTX 4090
```

### What the Student Learns

- Vocabulary semantics (word embeddings from PCA'd Llama embeddings)
- Sentence-level understanding (self-attention patterns)
- Cross-attention interface (how to attend to external memory)
- Task-prompt parsing ("query_role=producer", "task=graph_completion")

---

## Training Strategy (Stage 1)

### Phase A: Structural Foundation (30 epochs)

- **Tasks**: diverse, bfs, hodge_class, spectral_gap, path_counting
- **LR**: 1e-3 (GNN/TAT), 5e-4 (DSM), 1e-4 (TopoBridge)
- **Data**: 5000 train / 500 val per task (precomputed locally)
- **Goal**: GNN learns structural reasoning. DSM learns to NOT interfere (semantic_weight naturally low on purely structural tasks).
- **No gate penalty**. No bypass. DSM is always active.

### Phase B: Semantic Integration (30 epochs)

- **Tasks**: Phase A tasks (20% replay) + graph_completion + labeled_reasoning
- **LR**: 5e-4 (GNN/TAT), 3e-4 (DSM), 1e-4 (TopoBridge)
- **Data**: 5000 train / 500 val per task
- **Goal**: DSM starts contributing. semantic_weight rises for semantic tasks while staying low for structural. Task replay prevents catastrophic forgetting.

### Phase C: Full Symbiosis (30 epochs)

- **Tasks**: All tasks mixed (including redesigned analogical_transfer)
- **LR**: 3e-4 (all), cosine annealing to 0
- **Data**: 5000 train / 500 val per task
- **Goal**: Convergence. Both systems working symbiotically.

### Curriculum Fixes (from v1 post-mortem)

1. **Per-task max_classes**: Use `get_max_classes(task)`, rebuild classifier head per task
2. **Task replay**: Phase B/C includes 20% structural task samples to prevent forgetting
3. **Balanced classes**: Fix labeled_reasoning to ensure all 3 classes present
4. **Cosine annealing**: Replace ReduceLROnPlateau
5. **No gate penalty ever**: semantic_weight learns naturally from task gradients

---

## Compute Analysis

### Precomputable Locally (CPU, free)

| Operation | Time/sample | Total (7500 samples) |
|---|---|---|
| Dataset generation | ~50ms | 6 min |
| Structural features | ~5ms | 0.6 min |
| Cell complex + B1/B2 | ~10ms | 1.2 min |
| Spectral decomposition | ~20ms (n=20) | 2.5 min |
| Persistence diagrams | ~30ms | 3.7 min |
| Llama tokenization | ~1ms | negligible |
| **Total** | | **~15 min** |

### GPU Training Cost

| GPU | VRAM | DSM forward | Per-sample (3 iter) | Epoch (2000) | 90 epochs | Cost |
|---|---|---|---|---|---|---|
| **RTX 4090** | 24 GB | ~8ms | ~200ms | 6.7 min | 10 hrs | **$3.00** |
| **A100 40GB** | 40 GB | ~4ms | ~110ms | 3.7 min | 5.5 hrs | **$6.05** |
| **H100 80GB** | 80 GB | ~2ms | ~45ms | 1.5 min | 2.3 hrs | **$5.75** |

Add distillation: ~4 hrs on RTX 4090 (one-time).

### VRAM Budget (RTX 4090)

| Component | VRAM |
|---|---|
| DSM 250M (fp32, trainable) | 1.0 GB |
| GNN + TAT + Wave (160K) | 0.2 GB |
| TopoBridge (80K) | 0.05 GB |
| Optimizer (250M params, AdamW) | 2.0 GB |
| Activations (3 iterations) | 2.5 GB |
| **Total** | **~5.75 GB** |
| **Headroom** | **18.25 GB** |

Fits easily on RTX 4090. Even fits on RTX 3060 (12 GB).

---

## Modified Components

### ControlSignal

```python
@dataclass
class ControlSignal:
    frequency_gate: torch.Tensor      # (num_freqs,) [0,1]
    spatial_focus: torch.Tensor        # (N,) [0,1]
    confidence_weights: torch.Tensor   # (N,) [0,1]
    diffusion_time: torch.Tensor       # scalar, positive
    wave_damping: torch.Tensor         # scalar, positive
    semantic_weight: torch.Tensor      # scalar [0,1] — replaces llm_gate
    filter_weights: torch.Tensor | None  # (num_filters,) softmax
```

`semantic_weight` replaces `llm_gate`. Never penalized. Controls how much TAT attention is biased by DSM output. The executive learns this naturally.

### TopoBridge (updated for 1024-dim DSM)

- NodeProjector: Linear(32→1024) + LayerNorm
- PrefixGenerator: 8 queries → multi-head attention → prefix (8 x 1024)
- NodeExtractor: cross-attn(topo_memory, dsm_hidden) → Linear(1024→32)
- SemanticBias: Linear(32→max_nodes) for pairwise attention bias

### TAT Spatial Attention (modified)

```python
# Existing spatial attention
attn_logits = Q @ K.T / sqrt(d_k)

# New: semantic bias injection
if semantic_bias is not None:
    attn_logits = attn_logits + control_signal.semantic_weight * semantic_bias

attn_weights = softmax(attn_logits)
```

### ExecutiveReasoningLoop (modified)

Each iteration now includes:
1. GNN → control_signal + gnn_out
2. Wave dynamics → wave_out
3. TopoBridge encode(wave_out) → prefix + topo_memory
4. DSM forward(prefix, topo_memory, task_tokens) → semantic_hidden
5. TopoBridge decode(topo_memory, semantic_hidden) → semantic_bias
6. TAT(wave_out, control_signal, semantic_bias) → tat_out
7. Integration + convergence check

---

## Validation: Proving Symbiosis

### Ablation Experiment

| Variant | Structural Tasks | Semantic Tasks | Expected |
|---|---|---|---|
| GNN-only (no DSM) | Baseline | Baseline | Strong structural, weak semantic |
| GNN + random DSM (untrained) | ~Baseline | ~Baseline | Noise shouldn't hurt |
| GNN + distilled DSM (frozen) | ~Baseline | Moderate | Some semantic benefit |
| **GNN + co-trained DSM** | **~Baseline** | **Best** | **True symbiosis** |

### Success Criteria

1. **No structural regression**: BFS, path_counting, hodge_class within 2% of GNN-only
2. **Graph completion > 85%** (up from 75%)
3. **Labeled reasoning > 80%** with all 3 classes balanced (up from 68%)
4. **Redesigned analogical_transfer > 50%** (up from 18.5%)
5. **semantic_weight is task-discriminative**: <0.1 for BFS, >0.3 for labeled_reasoning
6. **Size generalization**: Maintains n=80 OOD performance

### Analogical Transfer Redesign

- Roles assigned by **degree centrality** (not random)
- 3 roles, 2 domain pairs (reduced combinatorics)
- Role features concatenated to node embeddings
- DSM reads domain-pair text → produces semantic similarity that biases TAT toward analogous nodes

---

## Implementation Phases

1. **DSM model class** — 250M transformer with cross-attention port
2. **Distillation script** — teacher (Llama) → student (DSM), one-time
3. **Updated TopoBridge** — 1024-dim interface, semantic bias output
4. **Modified TAT** — semantic_bias parameter in spatial attention
5. **Modified ControlSignal** — semantic_weight replaces llm_gate
6. **Modified ExecutiveReasoningLoop** — DSM in every iteration
7. **Updated HierarchicalMultiHopModel** — DSM instead of LlamaBackend
8. **Fixed analogical_transfer task** — degree-centrality roles
9. **Fixed curriculum script** — per-task max_classes, task replay, cosine annealing
10. **Precomputation pipeline** — local CPU dataset generation with cached spectral/persistence
11. **Training + evaluation**

---

## Dependencies

New:
- None (removes transformers/peft dependency for training; only needed for one-time distillation)

Existing (unchanged):
- torch, torch_geometric, torchdiffeq, gudhi, networkx

For distillation only (one-time):
- transformers, sentencepiece (to load Llama teacher)

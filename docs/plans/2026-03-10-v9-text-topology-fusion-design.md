# v9 Text-Topology Fusion Design

**Date**: March 10, 2026
**Goal**: Break the semantic ceiling on KG tasks (kg_relation 23% → 40%+) by deeply fusing text features with graph reasoning.

## Problem

v8 results reveal a clear structural vs semantic divide:
- Structural tasks: 74-99% bal_acc (pathvalid 83%, BFS 99.8%, hodge 95.8%)
- Semantic tasks: 14-23% bal_acc (kg_relation 23.2%, kg_concept 14.5%)

Root cause: text features are shallowly integrated. They enter the classifier as flat concatenated vectors but never participate in graph reasoning. The GNN/TAT processes structure but is blind to semantics.

### 5 Bottlenecks Identified

1. **Path 1 detachment** (`run_comparison.py:278`): `.detach()` blocks gradients from executive loop to text projection
2. **No direct text → control pathway**: MetaCognitiveController never sees raw text features
3. **Only query/target get text in classifier**: N-2 nodes' text is invisible to the classifier
4. **No text-aware graph reasoning**: Text features are flat (N, 32), never refined by graph structure
5. **Task embedding is opaque**: No semantic grounding for task identity

## Approach: B+C Hybrid (two phases)

### Phase 1: TextReasoningHead (Approach C)

A lightweight graph-aware text transformer that processes ALL node text embeddings.

```
Input: text_features (N, text_feat_dim=32) + adjacency mask (N, N)

GraphTextTransformer:
  - Input projection: Linear(32 → 64)
  - 2 transformer layers:
      - 4-head self-attention with adjacency-based mask
        (nodes attend to neighbors + self only)
      - FFN: Linear(64→128→64) + GELU + dropout(0.1)
      - LayerNorm + residual connections
  - Output: (N, 64) per-node semantic embeddings

Classifier concat:
  text_reasoning_q = output[query_node]      # (64,)
  text_reasoning_t = output[target_node]      # (64,)
  text_reasoning_diff = q - t                 # (64,)
  → 192 extra dims appended to combined vector
```

Key design choices:
- Hidden dim 64 (more capacity than raw 32-dim projection)
- 2 layers (enough for 2-hop reasoning: node → neighbor → neighbor's neighbor)
- Adjacency mask (graph-aware, not generic text encoding)
- Full gradient flow (no detach, separate from executive loop)
- ~50K parameters

Classifier input dim: 233 → 425

### Phase 2: TextEdgeEncoder (Approach B)

Compute semantically-grounded edge features from node text embeddings, injected into GNN message-passing.

```
For each edge (u, v):
  text_edge = MLP([text_u, text_v, text_u - text_v, text_u * text_v])
            = Linear(128→64) → GELU → Linear(64→32) → LayerNorm
            → (32,) per-edge text feature

Injection: additive + gated (not concatenated)
  augmented_edge = existing_edge_emb + gate * text_edge_emb
  gate = sigmoid(learned_scalar), init at -3.0 (~0.05)
```

Vectorized via boundary operator edge endpoint extraction. ~12K parameters.

**Side channel design**: text_edge_features passed as argument to HigherOrderGNN, NOT stored in CellComplex (avoids detach). Gradients flow: text → TextEdgeEncoder → GNN message-passing → output → classifier.

## Training Strategy

**Phase 1**: TextReasoningHead enabled, TextEdgeEncoder gate frozen at 0
- Resume from v8 checkpoints
- Reuse v8 datasets (no regeneration)
- 40 epochs, early stopping patience=10
- Target: kg_relation > 30%, kg_concept > 20%

**Phase 2**: Unfreeze TextEdgeEncoder gate
- Text gradients flow through GNN for first time
- Target: kg_relation > 40%, kg_concept > 25%

**Optimizer groups**:
- Group 1: GNN/TAT/executive (lr=0.0003)
- Group 2: DSM/Bridge (lr=0.0001)
- Group 3: Classifier + TextReasoningHead + TextEdgeEncoder (lr=0.001)

## Checkpoint Compatibility

- `_load_state_filtered()` handles size mismatches via partial copy
- New modules get fresh init
- Classifier heads: partial copy for first 233 dims, random init for new 192 dims
- Existing structural weights preserved

## Files Changed

| Order | File | Action |
|-------|------|--------|
| 1 | `src/llm/text_reasoning_head.py` | Create (~80 lines) |
| 2 | `src/llm/text_edge_encoder.py` | Create (~60 lines) |
| 3 | `src/benchmarks/run_comparison.py` | Add modules, update classifier_input_dim, call TextReasoningHead |
| 4 | `src/training/batch_utils.py` | Same pattern for batched path |
| 5 | `src/gnn_executive/higher_order_gnn.py` | Accept text_edge_features side channel |
| 6 | `src/reasoning_loop/executive_loop.py` | Thread text_edge_features to GNN |
| 7 | `scripts/run_dsm_curriculum.py` | Optimizer group routing |
| 8 | Tests | TextReasoningHead + TextEdgeEncoder tests |

**Not changed**: TAT, wave dynamics, MetaCognitiveController, QwenTextFeatureExtractor, QwenBridgeAdapter, CellComplex.

## Verification

1. All existing 710+ tests pass
2. TextReasoningHead: correct shape (N, 64) with adjacency masking
3. TextEdgeEncoder: correct shape (E, 32) from node texts
4. Full forward pass: classifier sees 425-dim input
5. TextEdgeEncoder gate=0: identical to without encoder
6. Backward: gradients reach TextReasoningHead params
7. Training dry run: 2 epochs kg_relation > random

## VRAM Budget (24GB RTX 4090)

| Component | v8 | v9 Phase 1 | v9 Phase 2 |
|-----------|-----|-----------|-----------|
| Qwen embed cache | 5.6GB | 5.6GB | 5.6GB |
| Model params | ~4MB | ~4.2MB | ~4.3MB |
| TextReasoningHead | 0 | ~0.2MB | ~0.2MB |
| TextEdgeEncoder | 0 | 0 | ~0.05MB |
| Activations + optim | ~3.5GB | ~3.7GB | ~3.8GB |
| **Total** | ~9.1GB | ~9.5GB | ~9.6GB |

Comfortable on 24GB.

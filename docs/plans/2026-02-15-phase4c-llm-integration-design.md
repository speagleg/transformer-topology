# Phase 4c: LLM Integration — TopoBridge Architecture

## Overview

Integrate a frozen Llama 3.2 1B/3B as the "subconscious" in the hierarchical reasoning system. The TAT becomes a **liminal space** (TopoBridge) that translates between the GNN executive's structured graph reasoning and the LLM's associative pattern completion.

**Architecture chosen: Hybrid ("Liminal Bridge")** — combines soft prompt prefix tokens with selective cross-attention at a middle LLM layer.

---

## Architecture

### Data Flow

```
GNN Executive (60K params)
    │
    ├──→ ControlSignal (frequency_gate, spatial_focus, confidence, llm_gate)
    │
    ▼
Wave Dynamics (50K params)
    │
    ▼
TAT (50K params) ──→ tat_out (N × 32)
    │
    ▼
TopoBridge Encoder (40K params)
    ├──→ NodeProjector: Linear(32→2048) + LayerNorm → topo_memory (N × 2048)
    └──→ PrefixGenerator: 8 learned queries attend over topo_memory → prefix (8 × 2048)
              │
              ▼
         Llama 3.2 1B (FROZEN, 1.2B params)
         ┌─────────────────────────────────────────────┐
         │ Layers 0-7:  self-attn [prefix | task_text] │
         │ Layer 8:     self-attn + cross-attn ← topo  │  ← LoRA adapters (~2M)
         │ Layers 9-15: self-attn                      │
         └──────────────────────┬──────────────────────┘
                                │
                                ▼
TopoBridge Decoder (40K params)
    └──→ NodeExtractor: cross-attn(topo_memory queries, LLM hidden keys/values)
         → Linear(2048→32) + LayerNorm → llm_out (N × 32)
              │
              ▼
Three-way Integration:
    tat_weight = (1 - llm_gate) × confidence
    llm_weight = llm_gate × confidence
    gnn_weight = 1 - confidence
    output = tat_weight × tat_out + llm_weight × llm_out + gnn_weight × gnn_out
```

### Parameter Budget

| Component          | Params   | Trainable? |
|--------------------|----------|------------|
| GNN Executive      | ~60K     | Yes        |
| TAT                | ~50K     | Yes        |
| Wave Dynamics      | ~50K     | Yes        |
| TopoBridge Encoder | ~40K     | Yes        |
| TopoBridge Decoder | ~40K     | Yes        |
| Cross-attn layer   | ~33M     | LoRA only  |
| LoRA adapters      | ~2M      | Yes        |
| Llama 3.2 1B       | 1.2B     | Frozen     |
| **Total trainable**| **~2.2M**|            |

---

## TopoBridge Module

### Encoder

**NodeProjector**: Projects each TAT node embedding (32-dim) into LLM hidden space (2048-dim).
```python
self.node_proj = nn.Sequential(
    nn.Linear(32, 2048),
    nn.LayerNorm(2048),
)
# Output: topo_memory (N, 2048) — used as cross-attention keys/values
```

**PrefixGenerator**: Attention-pools N nodes into K=8 fixed-size prefix tokens.
```python
self.prefix_queries = nn.Parameter(torch.randn(8, 2048))
# Multi-head attention: queries=prefix_queries, keys/values=topo_memory
# Output: prefix_tokens (8, 2048) — prepended to LLM input
```

Different prefix tokens can specialize: some capture topology type, others spectral signature, others spatial structure.

### Decoder

**NodeExtractor**: Each node "asks" the LLM what it learned via cross-attention.
```python
# queries = topo_memory (N, 2048) — node-aligned
# keys/values = LLM final hidden states
# Output → Linear(2048, 32) → llm_out (N, 32)
```

This ensures the output is node-aligned — preserving graph structure through the LLM round trip.

---

## Modified Reasoning Loop

### New ControlSignal field

```python
@dataclass
class ControlSignal:
    frequency_gate: torch.Tensor      # (num_freqs,) [0,1]
    spatial_focus: torch.Tensor        # (N,) [0,1]
    confidence_weights: torch.Tensor   # (N,) [0,1]
    diffusion_time: torch.Tensor       # scalar, positive
    wave_damping: torch.Tensor         # scalar, positive
    llm_gate: torch.Tensor             # NEW: scalar [0,1]
```

`llm_gate` lets the GNN executive decide whether to invoke the LLM. When `llm_gate < 0.1`, the LLM forward pass is skipped entirely — saving ~80% of per-iteration compute. The model should learn:
- BFS, hodge_class, spectral_gap → gate ≈ 0 (no LLM needed)
- Graph completion, labeled reasoning → gate > 0.5 (LLM helps)
- Analogical transfer → gate ≈ 1.0 (LLM essential)

### Loop per iteration

```
1. GNN Executive → control_signal + gnn_out
2. Wave dynamics → gnn_out + wave_residual
3. TAT executes with control signals → tat_out
4. IF llm_gate > 0.1:
     TopoBridge encode → Llama forward → TopoBridge decode → llm_out
   ELSE:
     llm_out = zeros
5. Three-way integration (tat_weight × tat + llm_weight × llm + gnn_weight × gnn)
6. Harmonic convergence check
```

### LLM Text Input

The task tokens are a short text prompt describing the current reasoning state:

```
"[TOPO] nodes=20 edges=45 topology=ba beta1=3 | task=graph_completion |
 iter=3/5 conf=0.72 [/TOPO]"
```

Generated from control signal + graph statistics. The prefix tokens carry the dense structural information; the text prompt gives the LLM semantic anchoring.

---

## Cross-Attention at Layer 8

Insert a cross-attention module after the self-attention in Llama layer 8 (middle of network):

```python
class TopoCrossAttention(nn.Module):
    def __init__(self, hidden_dim=2048, num_heads=8):
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        # LoRA adapters on Q, K, V projections

    def forward(self, hidden_states, topo_memory):
        # hidden_states: (1, seq_len, 2048) from self-attn output
        # topo_memory: (1, N, 2048) from TopoBridge encoder
        attn_out, _ = self.cross_attn(hidden_states, topo_memory, topo_memory)
        return self.norm(hidden_states + attn_out)
```

Why layer 8: Early layers (0-7) handle token-level features. Late layers (9-15) handle task-level reasoning. Layer 8 is where representations are most "conceptual" — the right place to inject structural knowledge. Empirically tunable.

---

## Tasks & Validation

### Regression Tasks (existing — LLM should NOT hurt)

| Task          | Purpose                                    | Expected llm_gate |
|---------------|--------------------------------------------|--------------------|
| BFS           | Multi-hop traversal                        | ≈ 0                |
| Hodge class   | Topological decomposition                  | ≈ 0                |
| Spectral gap  | Frequency structure                        | ≈ 0                |

### New LLM-Dependent Tasks

#### Tier 1: Graph Completion
- **Setup**: Known-structure graph (20 nodes, community structure). Remove 10-20% of edges.
- **Task**: Predict which missing edges should exist (binary classification per candidate pair).
- **Why LLM helps**: GNN sees local structure; LLM recognizes "community members connect to each other."
- **Metric**: Edge prediction F1 (target > 0.7).
- **Dataset**: 2000 train / 500 val.

#### Tier 2: Labeled Graph Reasoning
- **Setup**: Nodes/edges carry short text labels (e.g., causal graph: "rain"→"flood"→"evacuation" with "causes"/"prevents"/"enables" edges).
- **Task**: Given source+target, classify relationship type (causal chain, contradiction, independence).
- **Why LLM helps**: GNN traverses path; LLM understands "prevents" inverts causality.
- **Metric**: Relationship classification accuracy (target > 70% vs ~33% random baseline).
- **Control**: Scrambled labels (same structure, random label assignment) — accuracy should drop.
- **Dataset**: 1000 train / 200 val.

#### Tier 3: Analogical Transfer
- **Setup**: Two graphs with different topology but analogous semantic structure (food chain vs supply chain).
- **Task**: Given labeled query node in graph A, identify analogous node in graph B.
- **Why LLM helps**: Topologically different — analogy is semantic, not structural.
- **Metric**: Analogy accuracy (target > 50% vs ~20% random baseline).
- **Dataset**: 500 train / 100 val.

### Validation Matrix

| Task               | GNN-only | GNN+TAT | GNN+TAT+LLM | Expected           |
|--------------------|----------|---------|--------------|---------------------|
| BFS                | baseline | better  | same (gate≈0)| LLM doesn't hurt    |
| Hodge class        | baseline | better  | same (gate≈0)| LLM doesn't hurt    |
| Graph completion   | weak     | moderate| **best**     | LLM adds value      |
| Labeled reasoning  | random   | weak    | **best**     | LLM essential        |
| Analogical transfer| random   | random  | **best**     | LLM essential        |

---

## Training Strategy

### Phase A: Regression Lock (2-3 hours, A100)

Train on existing tasks only. Auxiliary loss penalizes `llm_gate > 0.1`. Validates that adding the LLM infrastructure doesn't break existing performance.

**Success**: All existing task accuracies within 2% of current baselines.

### Phase B: Graph Completion (4-6 hours, A100)

Introduce graph completion. Remove gate penalty — let model discover that opening the gate helps. Train TopoBridge + LoRA.

**Success**: Graph completion F1 > 0.7. `llm_gate` > 0.5 on this task, < 0.1 on BFS/hodge.

### Phase C: Language + Analogy (6-8 hours, A100)

Introduce labeled reasoning and analogical transfer. Full training on all tasks with no gate constraints.

**Success**: Labeled reasoning > 70% accuracy. Analogical transfer > 50%.

**Total training estimate**: ~12-17 hours on A100 (~$22 on Lambda).

---

## Dependencies

New:
- `transformers` (HuggingFace) — Llama 3.2 model loading
- `peft` — LoRA adapter implementation
- `sentencepiece` / `tokenizers` — Llama tokenizer

Existing (unchanged):
- `torch`, `torch_geometric`, `torchdiffeq`, `gudhi`, `networkx`

---

## Implementation Phases

1. **TopoBridge module** — encoder (projector + prefix gen) and decoder (extractor)
2. **LLM wrapper** — load frozen Llama, insert cross-attention at layer 8, LoRA setup
3. **Modified reasoning loop** — add llm_gate to ControlSignal, three-way integration
4. **Graph completion task** — dataset generator, training loop
5. **Labeled reasoning task** — text label vocabulary, dataset generator
6. **Analogical transfer task** — paired graph generator, dataset
7. **Three-phase curriculum training**
8. **Evaluation and analysis**

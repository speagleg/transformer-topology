# V6 Semantic Dual-Track Design: Pre-trained DSM vs Frozen LLM + Adapter

**Date:** 2026-02-23
**Status:** Design approved, pending implementation plan
**Branch:** TBD (from `phase7-computation-graph-topology`)

## Research Question

Can a semantic model learn useful representations of graph topology, and does pre-trained language knowledge help? Secondary: does active topological feedback from the embedding analyzer improve semantic model contribution?

## V5 Diagnosis (Why DSM Contributes Zero)

The v5 analysis (`data/dsm_results_v5/v5_full_analysis.json`) showed:

- **DSM ablation delta = 0.000** on all tested tasks (with vs without DSM identical)
- **Catastrophic forgetting**: structural tasks collapsed (diverse 98%→4%, bfs 99%→5%)
- **Class imbalance**: labeled_reasoning class 2 = 0% accuracy
- **Filter ensemble collapsed**: entropy = 0.0 (single filter dominant)
- `semantic_weight` learned to ~0.35 but the bias itself carried no information

**Root causes:**
1. No direct gradient signal to DSM (only contributes via attention bias)
2. `semantic_bias_proj` initialized at `std=0.01` — too small to matter
3. Fixed 8 prefix tokens with no task conditioning
4. Classifier rebuilt per task → weight destruction
5. 20% replay insufficient to prevent catastrophic forgetting
6. DSM trained from random init — 205M params with no prior knowledge

## Design Overview

Two parallel experiments sharing common infrastructure fixes:

| | Track 1 | Track 2 |
|--|---------|---------|
| **Semantic model** | WikiText-pretrained DSM (250M) | Frozen Qwen2.5-3B (4-bit quantized) |
| **Training** | Full fine-tune (gradual unfreeze) | GraphFormer adapter only (~20M) |
| **Compute** | GPU Instance 1 | GPU Instance 2 |
| **Topology feedback** | active_mode=true | active_mode=true |

---

## Part 1: Shared Infrastructure Fixes

These apply to both tracks and fix the v5 failure modes.

### 1a. Multi-Head Classifier

Replace `_rebuild_classifier()` (which destroys weights) with persistent per-task heads:

```python
class MultiHeadClassifier(nn.Module):
    def __init__(self, input_dim, task_classes):
        # task_classes: {"diverse": 11, "bfs": 16, "graph_completion": 2, ...}
        self.heads = nn.ModuleDict({
            task: nn.Linear(input_dim, n_classes)
            for task, n_classes in task_classes.items()
        })
    def forward(self, x, task):
        return self.heads[task](x)
```

Replaces the current `model.classifier` (single `nn.Sequential`). Heads persist across phases — no more weight destruction. New tasks get freshly initialized heads; existing heads retain learned weights.

### 1b. Stronger Replay (50%)

Increase replay ratio from 20% to 50% in Phase B/C. Additionally implement **feature replay**:
- At end of Phase A, save GNN penultimate-layer activations for a subset of training samples
- During Phase B/C replay, feed saved activations directly to classifier (skip GNN forward)
- Cheaper than full forward pass, preserves Phase A decision boundaries

### 1c. Auxiliary Contrastive Loss on Semantic Features

Give the semantic model a direct training signal independent of classification:

```python
def semantic_contrastive_loss(semantic_features, adjacency, labels):
    """InfoNCE: nodes with same structural role should have similar features."""
    # Positive pairs: same connected component, similar centrality
    # Negative pairs: different components, different roles
    # Temperature-scaled cosine similarity
    sim = F.cosine_similarity(features_i, features_j) / temperature
    return -log(exp(sim_pos) / sum(exp(sim_neg)))
```

- Weight: 0.1x main classification loss
- Applied to semantic model output (Track 1: DSM output, Track 2: adapter output)
- Provides gradient signal that doesn't depend on classifier accuracy

### 1d. Richer Bridge Output

Current TopoBridge only produces `semantic_bias` (N x N attention modifier). Expand to:

| Output | Shape | Usage |
|--------|-------|-------|
| `semantic_features` | (N, embed_dim) | Concatenated with GNN output before TAT |
| `semantic_bias` | (N, N) | Added to TAT attention logits (existing) |
| `graph_embedding` | (1, embed_dim) | Concatenated to pooled features before classifier |

`semantic_bias_proj` initialization increased from `std=0.01` to `std=0.1`.

### 1e. Active Topology Feedback

Enable `topology_observer.active_mode: true` in config. The 6-feature topology health vector from the embedding analyzer feeds into ControlHead, giving the GNN executive real-time information about:
- Attention entropy and head agreement
- Embedding manifold curvature and intrinsic dimension
- Weight space spectral properties
- Sheaf consistency

This was wired in the Phase 7-8 implementation and is ready to use.

### 1f. Batched Training (4-8x speedup)

V5 curriculum training uses per-sample sequential forward passes despite `train_epoch_batched()` and `evaluate_batched()` existing in `src/training/batch_utils.py`. Switch v6 to batched training:

- **Replace** `train_epoch()` / `train_epoch_with_replay()` calls with `train_epoch_batched()` / `evaluate_batched()`
- **Batch size:** 8 (default, fits comfortably on 4090)
- **How it works:** Groups B graphs → per-graph GNN+wave (cheap, sequential) → single batched DSM/LLM call (expensive, NOW BATCHED) → per-graph TAT+integration (cheap, sequential)
- **Applies to both tracks:** Track 1 batches DSM calls, Track 2 batches frozen Qwen inference
- Need to extend `_forward_batch()` in batch_utils to pass `topo_features`, `task_id`, and use `MultiHeadClassifier`
- Need batched version of Track 2's `QwenGraphBackend.forward()` (pad variable-size graph token sequences, single LLM call)

This is the single biggest speed improvement — v5 spent most of its time on sequential 205M-param DSM forward passes.

### 1g. Class-Weighted Loss

Address the class imbalance (labeled_reasoning class 2 = 0%):

```python
class_weights = 1.0 / class_counts.float()
class_weights /= class_weights.sum()
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

Recompute weights per task from training set class distribution.

---

## Part 2: Track 1 — WikiText-Pretrained DSM

### 2a. DSM Pre-training Phase

New script `scripts/pretrain_dsm.py`:

**Architecture (pre-training):**
- Same DSM core: 16 layers, 1024-dim, 16 heads, 4096 FFN
- Temporary text layers: nn.Embedding(vocab_size, 1024) + LM head (Linear 1024→vocab_size)
- Standard causal language modeling (next-token prediction)

**Dataset:** WikiText-103 (516M tokens, ~100MB download)

**Training:**
- ~5 epochs, cosine LR schedule, AdamW with weight decay 0.01
- Batch size 32, sequence length 512
- ~2-4 hours on a single 4090
- Save only transformer layers (discard embedding + LM head)

### 2b. Integration into Graph Pipeline

When loading pre-trained DSM:
1. Build full DSM with 16 transformer layers
2. Load WikiText checkpoint into the 16 layers
3. Cross-attention block at layer 4: freshly initialized (didn't exist during pre-training)
4. TopoBridge NodeProjector (Linear 32→1024) replaces text embedding
5. TopoBridge decoder replaces LM head

### 2c. Fine-tuning Strategy

- **Epochs 1-5 of Phase A:** DSM frozen, only TopoBridge + GNN/TAT train (let bridge learn the modality mapping)
- **Epochs 6-15:** Unfreeze top 4 DSM layers (layers 13-16), LR = 1e-5
- **Epochs 16+:** Unfreeze all DSM layers, LR = 1e-5
- TopoBridge LR: 1e-4 (unchanged from v5)
- GNN/TAT LR: 1e-3 (unchanged from v5)

### 2d. Task-Conditioned Prefix

Replace fixed 8-query PrefixGenerator with task-aware version:

```python
class TaskConditionedPrefixGenerator(nn.Module):
    def __init__(self, num_tasks=14, num_prefix=8, llm_dim=1024):
        self.task_embedding = nn.Embedding(num_tasks, llm_dim)
        self.prefix_queries = nn.Parameter(torch.randn(num_prefix, llm_dim))
        self.cross_attn = nn.MultiheadAttention(llm_dim, 8)

    def forward(self, node_features, task_id):
        # Combine task embedding with learned queries
        task_emb = self.task_embedding(task_id).unsqueeze(0)  # (1, llm_dim)
        queries = self.prefix_queries + task_emb  # broadcast task context
        prefix, _ = self.cross_attn(queries, node_features, node_features)
        return prefix
```

This lets the prefix generation adapt based on what task is being solved.

---

## Part 3: Track 2 — Frozen Qwen2.5-3B + GraphFormer Adapter

### 3a. Model Selection: Qwen2.5-3B

- 3B parameters, 32 layers, 2048 hidden dim
- AWQ 4-bit quantization → ~2GB VRAM (leaves 22GB for training)
- Apache 2.0 license
- Install: `pip install transformers autoawq`
- Model: `Qwen/Qwen2.5-3B-AWQ` from HuggingFace

### 3b. GraphFormer Adapter (~20M params)

New module `src/llm/graph_adapter.py`:

**Encoder (graph → LLM tokens):**

```python
class GraphFormerEncoder(nn.Module):
    def __init__(self, topo_dim=32, llm_dim=2048, num_tokens=16, num_layers=2):
        self.input_proj = nn.Linear(topo_dim, llm_dim)
        self.graph_queries = nn.Parameter(torch.randn(num_tokens, llm_dim))
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(llm_dim, 8, dim_feedforward=4*llm_dim)
            for _ in range(num_layers)
        ])
        self.task_embedding = nn.Embedding(14, llm_dim)

    def forward(self, node_embeddings, task_id):
        # node_embeddings: (N, 32) from GNN
        memory = self.input_proj(node_embeddings)  # (N, 2048)
        queries = self.graph_queries + self.task_embedding(task_id)
        for layer in self.layers:
            queries = layer(queries, memory)
        return queries  # (16, 2048) — "graph tokens" for LLM
```

- 16 graph tokens (more than DSM's 8 — graphs need more representational capacity)
- Task-conditioned queries (same idea as Track 1)
- 2-layer cross-attention transformer

**Decoder (LLM states → graph features):**

```python
class GraphFormerDecoder(nn.Module):
    def __init__(self, llm_dim=2048, topo_dim=32, num_layers=2):
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(llm_dim, 8, dim_feedforward=4*llm_dim)
            for _ in range(num_layers)
        ])
        self.feature_proj = nn.Linear(llm_dim, topo_dim)
        self.bias_proj = nn.Linear(topo_dim, topo_dim)
        self.graph_pool = nn.Linear(llm_dim, topo_dim)

    def forward(self, node_queries, llm_hidden_states):
        # node_queries: (N, 2048) projected node embeddings
        # llm_hidden_states: (16, 2048) from middle LLM layer
        for layer in self.layers:
            node_queries = layer(node_queries, llm_hidden_states)
        features = self.feature_proj(node_queries)      # (N, 32)
        bias = features @ self.bias_proj(features).T     # (N, N)
        graph_emb = self.graph_pool(llm_hidden_states.mean(0))  # (32,)
        return features, bias, graph_emb
```

### 3c. LLM Integration

```python
class QwenGraphBackend(BaseLLMBackend):
    def __init__(self, config):
        self.llm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-AWQ", device_map="cuda", torch_dtype=torch.float16
        )
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False  # Fully frozen
        self.encoder = GraphFormerEncoder(...)
        self.decoder = GraphFormerDecoder(...)

    def forward(self, node_embeddings, task_id):
        graph_tokens = self.encoder(node_embeddings, task_id)
        # Feed graph tokens through frozen LLM
        with torch.no_grad():
            llm_out = self.llm(inputs_embeds=graph_tokens.unsqueeze(0),
                              output_hidden_states=True)
        # Extract from middle layer (layer 16 of 32)
        hidden = llm_out.hidden_states[16].squeeze(0)
        # Decode back to graph space
        node_proj = self.encoder.input_proj(node_embeddings)
        features, bias, graph_emb = self.decoder(node_proj, hidden)
        return features, bias, graph_emb
```

- LLM is fully frozen, 4-bit quantized
- Only GraphFormer encoder + decoder are trainable (~20M params)
- No LoRA (cleaner experiment than Phase 4c)

### 3d. Training Strategy

- Same curriculum as Track 1 (Phase A → B → C)
- Same multi-head classifier, same replay, same auxiliary loss
- Optimizer: only GraphFormer params + bridge + GNN/TAT + classifiers
- LRs: GraphFormer 1e-4, GNN/TAT 1e-3 (no DSM group needed)

---

## Part 4: Dual Topology Integration (Embedding + Computation Graph)

Both topology analysis systems from Phase 7-8 are integrated via `TopologyObserver` (`src/topology_observer.py`), which coordinates them during training.

### 4a. Embedding Topology Analyzer (DSM/LLM internal health → active feedback)

**Module:** `src/topology_analyzer/` → `TransformerTopologyAnalyzer`
**Wrapper:** `DSMAnalysisWrapper` (Track 1) / needs new `QwenAnalysisWrapper` (Track 2)

Analyzes the semantic model's internal state every N epochs, producing a `TopologicalProfile` with:
- Per-layer Hodge decomposition ratios (gradient/curl/harmonic)
- Per-layer spectral gaps and effective rank
- Cross-layer sheaf coherence gap
- Per-head attention curl patterns
- Weight space spectral properties

**6-feature `active_features()` vector for ControlHead:**

| Index | Feature | What it measures |
|-------|---------|-----------------|
| 0 | `gradient_ratio` | Mean gradient energy across layers |
| 1 | `curl_ratio` | Mean curl energy (circular attention patterns) |
| 2 | `spectral_gap` | Mean spectral connectivity quality |
| 3 | `embedding_coherence` | Cross-layer sheaf gap (representation bottleneck) |
| 4 | `attention_curl` | Mean attention curl across all heads |
| 5 | `weight_alignment` | Cross-layer weight sheaf gap |

**Alert system** (`EmbeddingTopologyAdvisor`): Fires on sheaf gap collapse, effective rank collapse, high attention curl, high condition numbers, high harmonic ratios.

### 4b. Computation Graph Topology (autograd structure → monitoring + alerts)

**Module:** `src/computation_graph/` → `TrainingTopologyMonitor`
**Wrapper:** `CellComplexModelWrapper` + `analyze_computation_graph_cc()`

Captures PyTorch autograd graph as a CellComplex during one forward+backward pass, then computes:

| Metric | What it measures |
|--------|-----------------|
| `gradient_energy_ratio` | Hodge gradient component of computation flow |
| `curl_energy_ratio` | Circular/loop patterns in computation graph |
| `harmonic_energy_ratio` | Dead zones in computation graph |
| `spectral_gap` | Information flow connectivity (lambda_2) |
| `num_operations` | Nodes in autograd graph |
| `num_data_flows` | Edges in autograd graph |
| `num_composites` | 2-cells (composite operations) |

**Alert system** (`TrainingTopologyMonitor`): Fires on high curl energy (>0.4), high harmonic energy (>0.15), low spectral gap (<0.01). Also provides trend-based suggestions (rising curl → increase damping, declining spectral gap → widen layers).

### 4c. Active Mode Configuration

```yaml
topology_observer:
  enabled: true
  analyze_every: 5
  active_mode: true        # Changed from false — feeds 6-feature vector to ControlHead
  dsm_num_heads: 16        # Track 1 (adjust for Track 2: Qwen has 16 heads too)
  dsm_hidden_dim: 1024     # Track 1 (Track 2: Qwen hidden_dim=2048)

model:
  use_topo_feedback: true
  use_embedding_topo_feedback: true
```

**Data flow:** TopologyObserver.run_analysis() → profile.active_features() → cached → get_topo_features() → passed to train_epoch → model.forward(topo_features=...) → GNNExecutive → ControlHead

**For Track 2:** Create `QwenAnalysisWrapper` similar to `DSMAnalysisWrapper` — wraps the frozen Qwen model so the embedding topology analyzer can hook into its layers and analyze attention/hidden state patterns.

For ablation, run each track with `active_mode: false` as baseline comparison.

---

## Part 5: Parallel Training Setup

### GPU Instance Recommendations

**Track 1 (Pre-trained DSM 250M): RTX 4090 24GB**
- DSM is only 250M params (~1GB fp32) — 4090 has plenty of VRAM
- Batched training (batch_size=8): ~2GB model + ~6GB activations = ~8GB peak
- 4090 has 82 TFLOPS fp32, 165 TFLOPS fp16 — fastest single-GPU for this model size
- **Cost:** ~$0.30-0.50/hr on vast.ai
- **Estimated time:** ~8-12 hrs curriculum (with batching) + 2-4 hrs pre-training = ~14 hrs total

**Track 2 (Frozen Qwen2.5-3B + adapter): RTX 4090 24GB**
- Qwen2.5-3B AWQ 4-bit: ~2GB VRAM (frozen, inference only)
- GraphFormer adapter: ~80MB trainable
- GNN/TAT + activations: ~6GB
- Total: ~10GB peak — fits 4090 comfortably
- 4-bit LLM inference is INT4 tensor-core bound — 4090 Ada has 660 TOPS INT8, competitive with A100
- Batched inference (8 graphs × 16 tokens = 128 tokens per batch): negligible latency at 3B scale
- **Cost:** ~$0.30-0.50/hr on vast.ai
- **Estimated time:** ~12-16 hrs curriculum (LLM inference overhead per batch)

**Recommendation: 2x RTX 4090 24GB** — cheapest option that maximizes speed. Both models fit easily, 4090 has the best price/perf for this scale. No need for A100 unless we hit unexpected VRAM issues.

If 4090 availability is limited on vast.ai, acceptable alternatives:
- RTX A6000 48GB (~$0.40-0.60/hr) — more VRAM, slightly slower compute
- A100 80GB SXM (~$0.80-1.50/hr) — overkill but guaranteed to work, 2x memory bandwidth helps LLM inference

### Instance Configuration

| | Instance 1 (Track 1) | Instance 2 (Track 2) |
|--|----------------------|----------------------|
| **GPU** | RTX 4090 24GB | RTX 4090 24GB |
| **Model** | WikiText-pretrained DSM (250M) | Frozen Qwen2.5-3B-AWQ + adapter (20M trainable) |
| **VRAM peak** | ~8GB (with batch_size=8) | ~10GB (with batch_size=8) |
| **Batch size** | 8 (batched DSM forward) | 8 (batched Qwen inference) |
| **Training time** | ~14 hrs (pre-train + curriculum) | ~12-16 hrs (curriculum only) |
| **Estimated cost** | ~$5-7 | ~$5-8 |

### Shared Assets (rsync before training)

Both instances receive:
- Pregenerated datasets (`data/dsm_datasets/`) — ~3GB, already backed up locally
- Updated codebase with all v6 fixes
- Track-specific config files (`config/v6_track1.yaml`, `config/v6_track2.yaml`)

### Pre-training (Track 1 only)

Before curriculum training, Instance 1 runs `scripts/pretrain_dsm.py`:
- Download WikiText-103 (~180MB) → pre-train DSM → save checkpoint
- ~2-4 hours on 4090
- Then proceed with curriculum

### Speed Impact of Batching

V5 per-epoch times (single-sample, sequential):
- Phase A: ~640s/epoch (DSM disabled) → ~1843s/epoch (DSM enabled)
- Phase B/C: ~1800-2000s/epoch

V6 estimated per-epoch times (batch_size=8):
- Track 1: ~300-500s/epoch (batched DSM, 4-8x improvement)
- Track 2: ~400-600s/epoch (batched Qwen inference + adapter backprop)

---

## Part 6: Evaluation & Comparison

### Evaluation Script

Both tracks use `scripts/analyze_v5_results.py` (already built) with additions:
- Semantic feature t-SNE visualization
- Auxiliary loss convergence curves
- Topology feedback impact (active_mode=true vs false comparison)

### Primary Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Ablation delta | > 5% on >= 1 semantic task | **Key metric**: proves semantic model contributes |
| Structural retention | Within 5% of Phase A best | No catastrophic forgetting |
| Auxiliary loss | Decreasing over training | Semantic features are learning |

### Secondary Metrics

- Per-class accuracy (no class at 0%)
- Size generalization (n=20, 40, 80)
- Topology transfer (train vs held-out)
- Control signal adaptation (semantic_weight varies by task?)
- Filter weights entropy (> 0.5 = multiple filters active)
- Topology feedback utilization (topo_features influence on control signals)

### Comparison Matrix

Run 4 experiments total (2 tracks x 2 topology modes):

| Experiment | Semantic Model | Active Topo | Instance |
|------------|---------------|-------------|----------|
| T1-passive | Pretrained DSM | false | GPU 1 |
| T1-active | Pretrained DSM | true | GPU 1 |
| T2-passive | Qwen2.5-3B + adapter | false | GPU 2 |
| T2-active | Qwen2.5-3B + adapter | true | GPU 2 |

Run passive first (shorter, establishes baseline), then active.

---

## Files Summary

### New Files
| File | Description |
|------|-------------|
| `scripts/pretrain_dsm.py` | WikiText-103 pre-training for DSM |
| `src/llm/graph_adapter.py` | GraphFormer encoder/decoder for Track 2 |
| `src/llm/qwen_backend.py` | Frozen Qwen2.5-3B backend implementing BaseLLMBackend |
| `src/topology_analyzer/qwen_wrapper.py` | Wrap Qwen for embedding topology analysis (Track 2) |
| `src/training/contrastive_loss.py` | Auxiliary contrastive loss for semantic features |
| `src/training/multi_head_classifier.py` | Persistent per-task classifier heads |
| `src/training/feature_replay.py` | Feature replay buffer for catastrophic forgetting prevention |
| `config/v6_track1.yaml` | Track 1 config (pretrained DSM + batching + active topo) |
| `config/v6_track2.yaml` | Track 2 config (Qwen + adapter + batching + active topo) |

### Modified Files
| File | Changes |
|------|---------|
| `src/llm/topo_bridge.py` | Richer output (features + bias + graph_embedding), larger init, task_id param |
| `src/llm/dsm_backend.py` | Load WikiText checkpoint, gradual unfreeze support |
| `src/benchmarks/run_comparison.py` | MultiHeadClassifier, graph_embedding input to classifier |
| `src/benchmarks/run_benchmark_suite.py` | Config support for new params |
| `src/reasoning_loop/executive_loop.py` | Pass semantic_features + graph_embedding, task_id to bridge |
| `src/tat/spatial_attention.py` | Accept semantic_features (concatenation with node embeddings) |
| `src/gnn_executive/control_head.py` | Accept graph_embedding input |
| `src/training/batch_utils.py` | Extend `_forward_batch()` for topo_features, task_id, MultiHeadClassifier, Track 2 batched Qwen |
| `src/topology_observer.py` | Support Track 2 Qwen wrapper, configurable analyzer params per track |
| `scripts/run_dsm_curriculum.py` | Switch to `train_epoch_batched()` / `evaluate_batched()`, multi-head classifier, 50% replay, class weights, contrastive loss, gradual unfreeze |
| `config/dsm_training.yaml` | topology_observer.active_mode=true |
| `scripts/analyze_v5_results.py` | t-SNE visualization, contrastive loss tracking, topology alert history |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Qwen2.5-3B too large for 4090 | Fall back to Qwen2.5-1.5B-AWQ (~1GB) or use A100 |
| WikiText pre-training doesn't help DSM | Still has all shared fixes; compare with v5 baseline |
| Both tracks show zero contribution | Shared fixes (multi-head classifier, replay) independently valuable |
| Catastrophic forgetting persists | Feature replay + 50% ratio is aggressive; if still failing, switch to EWC |
| Contrastive loss dominates training | Weight at 0.1x, anneal down if classification loss stalls |

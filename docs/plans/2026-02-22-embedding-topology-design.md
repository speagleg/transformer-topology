# LLM Embedding Topology Analyzer: Design Document

> **Status**: Approved design, pending implementation.
> **Branch**: `phase8-embedding-topology` (from `main`, cherry-pick computation graph + batching commits from phase7)
> **Date**: 2026-02-22

Build a dual-mode framework (observer + active) for discovering, predicting, and acting on topological structure in LLM internal representations. General toolkit tested on our DSM first, portable to any transformer.

---

## Motivation

Standard interpretability tools (attention visualization, probing classifiers, representation similarity) treat LLM internals as flat vector spaces. They miss the *topological structure* of how representations organize: clusters forming and dissolving across layers, circular attention patterns, coherence of learned subspaces across depth.

This project applies the topological toolkit we have already built (CellComplex, Hodge decomposition, spectral analysis, persistence homology, sheaf diffusion) to the hidden states, attention matrices, and weight matrices of transformer models. The question: does the topology of LLM internals carry actionable information that flat metrics miss?

---

## Architecture Overview

```
                        +---------------------------+
                        |   Target Transformer      |
                        |   (DSM / any model)       |
                        +-----|---------|-----+-----+
                              |         |     |
                    hidden    |  attn   |     | weights
                    states    |  maps   |     |
                              v         v     v
                     +--------+---------+-----+--------+
                     |     TransformerHookManager       |
                     |  (forward hooks, captures all)   |
                     +--------+---------+-----+--------+
                              |         |     |
                              v         v     v
                +-------------+--+ +----+---+ +---+------------+
                | Layer 1:       | | Layer 2:| | Layer 3:       |
                | Embedding      | | Attn    | | Weight Space   |
                | Manifold       | | Flow    | | Geometry       |
                +-------+--------+ +----+----+ +------+---------+
                        |              |              |
                        v              v              v
                  +-----+----+  +------+-----+  +----+-------+
                  | Per-layer |  | Per-head   |  | Per-layer  |
                  | CellCompx |  | CellCompx  |  | SVD + PD   |
                  | + sheaf   |  | + sheaf    |  | + sheaf    |
                  +-----+-----+ +------+------+ +-----+------+
                        |              |               |
                        +---------+----+----+----------+
                                  |         |
                                  v         v
                        +---------+--+ +----+-----------+
                        | Topological | | Active Mode    |
                        | Profile     | | (DSM feedback) |
                        +-------------+ +----------------+
```

---

## Layer 1: Embedding Manifold Analyzer

Analyzes the geometry and topology of hidden state representations at each transformer layer.

### Cell Complex Construction

For each transformer layer `l`, build a CellComplex from the hidden states `H_l` of shape `(seq_len, d_model)`:

| Cell Dim | Construction | Semantics |
|----------|-------------|-----------|
| 0-cell | One per token position | Token embedding at layer `l` |
| 1-cell | k-NN or epsilon-ball neighbors in embedding space | Semantic proximity between tokens |
| 2-cell | Triangles (3-cliques in the neighbor graph) | Local manifold patches |

```
Tokens at layer l:       CellComplex:

  "The" -----.            0 --- 1
  "cat"  ----+--k-NN-->  / \ / \
  "sat"  ----+          2 ---3---4   (triangles = 2-cells)
  "on"   ----+           \ / \ /
  "the"  ----'            5 ---6
```

**Neighborhood construction strategies** (configurable):
- `knn`: Fixed k nearest neighbors (default k=5). Stable cell count across layers.
- `epsilon`: All neighbors within epsilon-ball. Adaptive density, but variable cell count.
- `mutual_knn`: Edge only if both tokens are in each other's k-NN. Sparser, more reliable topology.

### Per-Layer Invariants

From each layer's CellComplex, extract:

1. **Persistence diagrams** (via `compute_persistence_diagram`):
   - H0 features: connected components = semantic clusters. Long-lived H0 = stable token groupings.
   - H1 features: 1-cycles = holes in the embedding manifold. Tokens arranged in a ring structure suggest circular/periodic relationships.

2. **Betti numbers**: `beta_0` = number of clusters, `beta_1` = number of holes. Track across layers to see when the model merges clusters (beta_0 drops) or creates loops (beta_1 rises).

3. **Spectral gap of L0** (via `spectral_decomposition(cc, dim=0)`):
   - Large gap: tokens form a well-connected manifold (information diffuses quickly).
   - Small gap: near-disconnected components (possible information isolation).

4. **Hodge decomposition of token-to-token signals**:
   - Signal: pairwise cosine similarity on edges.
   - Gradient component: similarity flows hierarchically (tree-like organization).
   - Curl component: similarity flows in cycles (circular token relationships).
   - Harmonic component: similarity trapped in topology (dead zones in the representation).

### Cross-Layer Sheaf

Build a layer-level CellComplex to measure global representation coherence across depth:

```
Layer Graph:

  L0 ---- L1 ---- L2 ---- ... ---- L_n
   |        |        |                |
   |  sheaf restriction maps          |
   |  F_{Li -> edge_ij} = Linear(d, d)|
   |                                   |

0-cells: layers (N = num_layers)
1-cells: consecutive layer pairs
Sheaf: F_{Li -> edge} is a learned Linear(d_model, d_model)
```

**Sheaf Laplacian spectral gap** of this layer graph measures global representation coherence:
- Large gap: representations transform smoothly across layers (high coherence).
- Small gap: some layer transition is a bottleneck (information lost between layers i and i+1).
- Critical insight: a low sheaf spectral gap between layers `i` and `i+1` localizes a representation bottleneck that flat metrics (like CKA) can only detect globally.

**Implementation**: Reuse `SheafLaplacian` from `src/spectral/sheaf_diffusion.py`. SVD spectral normalization (sigma_max <= 1) and PSD enforcement from Phase 4d apply directly, since the layer graph restriction maps face the same stability challenges as the graph-level ones.

### Computational Notes

- Hidden states captured via forward hooks (no backward needed for observer mode).
- k-NN computed via `torch.cdist` on `(seq_len, d_model)` -- O(seq_len^2 * d_model).
- For seq_len > 1024: subsample tokens or use approximate NN (FAISS).
- Persistence via gudhi Rips complex: O(seq_len^3) worst case, but triangle cap keeps it manageable for seq_len <= 512.

---

## Layer 2: Attention Flow Topology

Analyzes the topological structure of attention patterns per head and across heads.

### Cell Complex from Attention

For each attention head `h` at layer `l`, given attention matrix `A_h` of shape `(seq_len, seq_len)`:

| Cell Dim | Construction | Semantics |
|----------|-------------|-----------|
| 0-cell | One per token position | Token as attention source/target |
| 1-cell | Edge (i, j) where `A_h[i,j] > threshold` | Token i attends to token j |
| 2-cell | Triangles: i->j->k->i all above threshold | Attention cycles |

**Threshold selection**: Adaptive per-head. Use `mean + 0.5 * std` of attention weights as default. This preserves the structural backbone while filtering noise.

### Per-Head Hodge Decomposition

Apply `hodge_decomposition(cc, attention_signal, dim=1)` where the edge signal is the attention weight:

- **Gradient component** (exact forms): Direct reasoning chains. Token A attends to B, B attends to C -- information flows hierarchically. High gradient ratio = attention implements tree-structured information routing.
- **Curl component** (co-exact forms): Circular attention. A->B->C->A. Information circulates rather than flowing. High curl ratio = attention heads doing iterative refinement or "deliberation."
- **Harmonic component**: Attention energy trapped in topological structure -- neither flowing nor circulating. High harmonic ratio = dead attention patterns, capacity waste.

```
Head Classification by Hodge Profile:

  gradient-dominant     curl-dominant       harmonic-dominant
  (tree routing)        (iterative)         (dead/degenerate)

    A                   A ---> B            A     B
    |                   ^       |
    v                   |       v           C     D
    B --- C             D <--- C
          |                                 (disconnected,
          v                                  no information
          D                                  flow)
```

### Multi-Head Sheaf

Build a CellComplex over attention heads within a layer to discover head relationships:

```
Head Graph:

  H0 ---- H1 ---- H2
   \      |      /
    \     |     /
     \    |    /
      H3--+--H4

0-cells: attention heads
1-cells: heads attending to overlapping token sets (Jaccard > threshold on top-k tokens)
Sheaf: restriction maps capture how heads represent shared attended tokens differently
```

**Sheaf Laplacian spectral gap** of the head graph:
- Large gap: heads are complementary (diverse coverage of token relationships).
- Small gap: heads are redundant (similar attention patterns, candidates for pruning/merging).

### Cross-Layer Attention Persistence

Track how attention topological features evolve across layers:

1. Compute persistence diagrams for each layer's attention CellComplex.
2. Compute Wasserstein distance between consecutive layers' diagrams.
3. Large Wasserstein distance = attention pattern undergoes topological phase transition at that layer.
4. Persistent H1 features (cycles that survive many layers) = stable iterative attention patterns.

---

## Layer 3: Weight Space Geometry

Analyzes the topological structure of learned weight matrices.

### Per-Layer SVD Analysis

For each weight matrix `W` of shape `(d_out, d_in)`:

1. **SVD**: `U, S, V = torch.linalg.svd(W)`
2. **Singular value persistence**: Treat singular values as a 1D point cloud. Compute persistence of `{s_1, s_2, ..., s_r}`:
   - Long-lived H0 features = well-separated singular value clusters = distinct learned subspaces.
   - Short-lived H0 = continuous spectrum = no clear subspace decomposition.
3. **Effective rank**: `exp(entropy(S/sum(S)))` -- smooth measure of how many dimensions carry meaningful information.
4. **Spectral gap of W^T @ W**: Gap between largest and second-largest eigenvalue. Measures how dominant the primary learned direction is.
5. **Condition number**: `S_max / S_min`. High condition = numerically unstable, gradient-sensitive directions.

### Cross-Layer Weight Sheaf

Build a CellComplex measuring alignment of learned subspaces across depth:

```
Weight Sheaf:

  W0          W1          W2          W3
  V0_right -> V1_left    V1_right -> V2_left    ...
        F_{0->01}   F_{1->01}   F_{1->12}   F_{2->12}

0-cells: layer weight matrices
1-cells: consecutive layer pairs
Sheaf restriction: F maps layer i's right singular vectors to layer i+1's left singular vectors
```

**Restriction map construction**: For edge between layers `i` and `i+1`:
- `F_{i -> edge}` = `Linear(rank_i, d_stalk)` initialized from `V_i[:rank_i, :]`
- `F_{i+1 -> edge}` = `Linear(rank_j, d_stalk)` initialized from `U_{i+1}[:, :rank_j]`

**Sheaf Laplacian spectral gap** measures alignment coherence:
- Large gap: layer subspaces are well-aligned (smooth transformation across depth).
- Small gap: misaligned subspaces (information must be re-encoded between layers, possible bottleneck).

### Training Dynamics

Track weight topology across training checkpoints:

1. Save persistence diagrams at each checkpoint.
2. Compute Wasserstein distance between consecutive checkpoints' diagrams.
3. **Phase transition detection**: Sudden spike in Wasserstein distance = topological phase transition in learned representations. Correlates with loss plateaus and learning rate schedule transitions.
4. **Effective rank trajectory**: Plot effective rank per layer over training. Collapsing rank = representation collapse. Expanding rank = capacity utilization growing.

```
Training Timeline:

  Epoch:     1    5    10   15   20   25   30
  W-dist:    |....|..*.|....|.*..|....|....|
                   ^         ^
                   |         |
            phase transition  phase transition
            (sudden topo      (LR drop causes
             restructure)      rank compression)
```

---

## Active Mode (DSM Only)

When analyzing our own DSM, the observer becomes a participant -- topological features feed back into the model's control loop.

### Real-Time Topo Feedback

The existing `ControlHead(use_topo_feedback=True)` accepts a 3-element `topo_features` vector `[gradient_ratio, curl_ratio, spectral_gap]`. Extend this to 6 features by adding LLM-internal signals:

| Feature | Source | What It Measures |
|---------|--------|-----------------|
| `gradient_ratio` | Existing: computation graph Hodge | Clean gradient flow |
| `curl_ratio` | Existing: computation graph Hodge | Gradient circulation |
| `spectral_gap` | Existing: computation graph spectral | Information bottleneck |
| `embedding_coherence` | Layer 1: cross-layer sheaf gap | Representation smoothness across depth |
| `attention_curl` | Layer 2: per-head Hodge curl | Circular attention patterns |
| `weight_alignment` | Layer 3: cross-layer weight sheaf gap | Subspace alignment |

The ControlHead trunk input dimension increases from `embedding_dim + 2 + 3` to `embedding_dim + 2 + 6`. The three new features modulate:
- `embedding_coherence` low -> increase `diffusion_time` (more wave smoothing to compensate for layer bottleneck)
- `attention_curl` high -> increase `wave_damping` (break circular attention patterns)
- `weight_alignment` low -> shift `filter_weights` toward higher-frequency filters (compensate for inter-layer misalignment)

### Epoch-Level Advisor

Extend `TrainingTopologyMonitor` from `src/computation_graph/diagnostics.py` with new alert conditions:

| Condition | Alert | Suggested Action |
|-----------|-------|------------------|
| Sheaf spectral gap collapse (Layer 1) | Layer bottleneck at depth i | Add skip connection or widen layer i |
| Attention curl accumulation (Layer 2) | Heads {h1, h2} cycling | Increase head dropout or reduce those heads' LR |
| Sheaf coherence drop between heads (Layer 2) | Redundant heads {h3, h4} | Merge or prune heads |
| Effective rank deficiency (Layer 3) | Layer i rank collapsed to k | Reinitialize or apply spectral regularization |
| Wasserstein spike (Layer 3) | Phase transition detected at step N | Log and correlate with LR schedule |
| Persistence H1 born (Layer 1) | New topological hole at layer i | Representation developing circular structure -- monitor |

### Offline Architecture Search

Accumulate topological profiles across a training run. Use them to guide architecture decisions:

1. **Pruning**: Layers with consistently low sheaf coherence + harmonic-dominant attention heads are candidates for removal.
2. **Skip connections**: Layer pairs with low sheaf spectral gap benefit from direct skip connections.
3. **Head merging**: Attention heads with high sheaf coherence between them (redundant) can be merged.
4. **Depth allocation**: Layers where persistence diagram complexity peaks are doing the most representational work -- allocate more capacity there.

---

## Data Structures

### TopologicalProfile

```python
@dataclass
class TopologicalProfile:
    """Complete topological profile of a transformer at one point in time."""

    # Layer 1: Embedding Manifold
    per_layer_persistence: dict[int, list[np.ndarray]]  # layer -> [H0_diagram, H1_diagram]
    per_layer_betti: dict[int, tuple[int, int]]          # layer -> (beta_0, beta_1)
    per_layer_spectral_gap: dict[int, float]             # layer -> lambda_2 of L0
    per_layer_hodge_ratios: dict[int, tuple[float, float, float]]  # grad, curl, harm
    cross_layer_sheaf_gap: float                         # sheaf Laplacian spectral gap

    # Layer 2: Attention Flow
    per_head_hodge_ratios: dict[tuple[int, int], tuple[float, float, float]]  # (layer, head) -> ratios
    per_head_classification: dict[tuple[int, int], str]  # "gradient" | "curl" | "harmonic"
    per_layer_head_sheaf_gap: dict[int, float]           # layer -> head redundancy measure
    cross_layer_attention_wasserstein: dict[int, float]  # layer -> W-dist to next layer

    # Layer 3: Weight Space
    per_layer_effective_rank: dict[int, float]
    per_layer_condition_number: dict[int, float]
    per_layer_sv_persistence: dict[int, list[np.ndarray]]
    cross_layer_weight_sheaf_gap: float

    # Metadata
    model_name: str
    num_layers: int
    num_heads: int
    hidden_dim: int
    seq_len: int
    timestamp: float
```

### LayerCellComplex

Wrapper that stores the CellComplex alongside the metadata needed to interpret it:

```python
@dataclass
class LayerCellComplex:
    """CellComplex built from one transformer layer's hidden states."""
    cc: CellComplex
    layer_idx: int
    source: str  # "embedding" | "attention" | "weight"
    head_idx: int | None  # for attention complexes
    neighborhood: str  # "knn" | "epsilon" | "mutual_knn"
    k_or_epsilon: float
```

---

## Implementation Structure

```
src/topology_analyzer/
    __init__.py
    hooks.py                  # TransformerHookManager
    embedding_analyzer.py     # Layer 1: hidden state topology
    attention_analyzer.py     # Layer 2: attention flow topology
    weight_analyzer.py        # Layer 3: weight space geometry
    sheaf_analyzer.py         # Cross-layer sheaf construction + coherence
    profile.py                # TopologicalProfile + LayerCellComplex dataclasses
    feedback.py               # Active mode: topo -> ControlHead, epoch advisor
```

### Module Responsibilities

**hooks.py -- TransformerHookManager**

```python
class TransformerHookManager:
    """Register forward hooks on all transformer layers, capture hidden states
    and attention weights. Works with any nn.Module that has a standard
    transformer structure (layers attribute, self_attn sub-module)."""

    def __init__(self, model: nn.Module):
        ...

    def __enter__(self) -> 'TransformerHookManager':
        # Register hooks on all layers
        ...

    def __exit__(self, *args):
        # Remove hooks
        ...

    def get_hidden_states(self) -> dict[int, torch.Tensor]:
        # layer_idx -> (seq_len, d_model)
        ...

    def get_attention_maps(self) -> dict[tuple[int, int], torch.Tensor]:
        # (layer_idx, head_idx) -> (seq_len, seq_len)
        ...
```

Distinct from `ComputationGraphCapture` in `src/computation_graph/capture.py`, which hooks into autograd for computation graph topology. `TransformerHookManager` hooks into forward passes to capture internal representations -- different object, different purpose.

**embedding_analyzer.py -- EmbeddingManifoldAnalyzer**

```python
class EmbeddingManifoldAnalyzer:
    """Layer 1: Build CellComplex from hidden states, compute topological invariants."""

    def __init__(self, neighborhood: str = 'knn', k: int = 5, epsilon: float = 1.0):
        ...

    def build_cell_complex(self, hidden_states: torch.Tensor, layer_idx: int) -> LayerCellComplex:
        # hidden_states: (seq_len, d_model)
        # Returns CellComplex with 0-cells=tokens, 1-cells=neighbors, 2-cells=triangles
        ...

    def analyze_layer(self, lcc: LayerCellComplex) -> dict:
        # Returns persistence, betti, spectral_gap, hodge_ratios
        ...
```

**attention_analyzer.py -- AttentionFlowAnalyzer**

```python
class AttentionFlowAnalyzer:
    """Layer 2: Build CellComplex from attention, Hodge decompose, classify heads."""

    def __init__(self, threshold_mode: str = 'adaptive'):
        ...

    def build_cell_complex(self, attn_map: torch.Tensor,
                           layer_idx: int, head_idx: int) -> LayerCellComplex:
        # attn_map: (seq_len, seq_len)
        ...

    def classify_head(self, hodge_ratios: tuple[float, float, float]) -> str:
        # Returns "gradient" | "curl" | "harmonic" based on dominant component
        ...
```

**weight_analyzer.py -- WeightSpaceAnalyzer**

```python
class WeightSpaceAnalyzer:
    """Layer 3: SVD, singular value persistence, effective rank, condition number."""

    def analyze_weight_matrix(self, W: torch.Tensor, layer_idx: int) -> dict:
        # Returns effective_rank, condition_number, sv_persistence, spectral_gap
        ...

    def training_dynamics(self, profiles: list[TopologicalProfile]) -> dict:
        # Wasserstein distances between consecutive profiles, phase transition detection
        ...
```

**sheaf_analyzer.py -- CrossLayerSheafAnalyzer**

```python
class CrossLayerSheafAnalyzer:
    """Build cross-layer sheaf for any of the three layers. Reuses SheafLaplacian."""

    def __init__(self, feature_dim: int, num_layers: int):
        self.sheaf_lap = SheafLaplacian(
            n_edges=num_layers - 1,
            feature_dim=feature_dim,
        )
        ...

    def compute_coherence(self, per_layer_features: dict[int, torch.Tensor]) -> float:
        # Build layer CellComplex, compute sheaf Laplacian, return spectral gap
        ...
```

**feedback.py -- TopologyFeedback**

```python
class TopologyFeedback:
    """Active mode: convert TopologicalProfile into ControlHead topo_features."""

    def profile_to_features(self, profile: TopologicalProfile) -> torch.Tensor:
        # Returns (6,) tensor for ControlHead topo_feedback input
        ...

class EmbeddingTopologyAdvisor:
    """Epoch-level advisor: extends TrainingTopologyMonitor with LLM-internal alerts."""

    def __init__(self, thresholds: dict[str, float] | None = None):
        ...

    def check_alerts(self, profile: TopologicalProfile) -> list[str]:
        ...

    def suggest_architecture_changes(self, history: list[TopologicalProfile]) -> list[str]:
        ...
```

---

## Reused Components

All of these exist in the codebase today and are used directly:

| Component | Location | Used For |
|-----------|----------|----------|
| `CellComplex` | `src/cell_complex/cell_complex.py` | All three layers: stores topology |
| `hodge_decomposition()` | `src/spectral/decomposition.py` | Layers 1+2: signal decomposition |
| `spectral_decomposition()` | `src/spectral/decomposition.py` | All layers: eigendecomposition of Hodge Laplacians |
| `hodge_laplacian_0/1/2()` | `src/spectral/laplacian.py` | All layers: Laplacian construction |
| `compute_persistence_diagram()` | `src/spectral/persistence.py` | Layers 1+3: persistence homology |
| `vectorize_persistence()` | `src/spectral/persistence.py` | Feature extraction from diagrams |
| `SheafLaplacian` | `src/spectral/sheaf_diffusion.py` | Cross-layer sheaf construction |
| `SheafDiffusion` | `src/spectral/sheaf_diffusion.py` | Active mode: sheaf-based smoothing |
| `TopologicalDiagnostics` | `src/computation_graph/diagnostics.py` | Extended with embedding-level diagnostics |
| `TrainingTopologyMonitor` | `src/computation_graph/diagnostics.py` | Extended by `EmbeddingTopologyAdvisor` |
| `ControlHead` | `src/gnn_executive/control_head.py` | Active mode: topo_feedback input (extended to 6 features) |
| `TopologicalConfidence` | `src/computation_graph/confidence.py` | Extended with embedding-level confidence |

---

## Implementation Plan

### Phase A: Observer Mode (Layers 1-3, Passive)

1. **TransformerHookManager** -- forward hooks, capture hidden states and attention maps
2. **EmbeddingManifoldAnalyzer** -- k-NN CellComplex from hidden states, persistence, Betti, spectral gap, Hodge decomposition
3. **AttentionFlowAnalyzer** -- attention CellComplex, per-head Hodge decomposition, head classification
4. **WeightSpaceAnalyzer** -- SVD, singular value persistence, effective rank, condition number
5. **CrossLayerSheafAnalyzer** -- cross-layer sheaf for all three layers, coherence scores
6. **TopologicalProfile** -- aggregate all analyses into one dataclass
7. **Tests**: Unit tests for each analyzer on synthetic data. Integration test on DSM.

### Phase B: Training Diagnostics

8. **EmbeddingTopologyAdvisor** -- threshold-based alerts from TopologicalProfile
9. **Training dynamics** -- Wasserstein distance between checkpoint profiles, phase transition detection
10. **Visualization** -- Profile plots (Betti trajectories, sheaf gap curves, head classification heatmaps)
11. **Validation**: Run on DSM training, correlate topological features with loss/accuracy

### Phase C: Active Mode (DSM Only)

12. **TopologyFeedback** -- profile-to-features conversion for ControlHead
13. **ControlHead extension** -- 3 -> 6 topo_features, new modulation targets
14. **Feedback training loop** -- train DSM with topo feedback enabled, measure improvement
15. **Offline architecture search** -- use accumulated profiles to suggest pruning/restructuring

---

## Success Criteria

### Tier 1: Discovery (Phase A)

Find non-obvious topological structure in LLM internals that flat metrics miss.

- [ ] Persistence diagrams show meaningful H0/H1 features that change across layers (not just noise)
- [ ] Attention head Hodge classification reveals distinct head types (not all gradient-dominant)
- [ ] Cross-layer sheaf spectral gap varies meaningfully (identifies specific bottleneck layers)
- [ ] Weight effective rank trajectories are non-trivial (layers have different rank profiles)

### Tier 2: Predictive (Phase B)

Topological features correlate with model quality metrics.

- [ ] Cross-layer sheaf gap correlates with validation accuracy (r > 0.5)
- [ ] Attention curl ratio predicts task-specific failure modes
- [ ] Wasserstein distance spikes align with known training dynamics (LR drops, phase transitions)
- [ ] Head classification predicts pruning candidates (harmonic-dominant heads removable without accuracy loss)

### Tier 3: Actionable (Phase C)

Topology guides training decisions and improves model quality.

- [ ] Active topo feedback improves DSM accuracy over baseline (statistically significant)
- [ ] Architecture suggestions from accumulated profiles match manual expert judgments
- [ ] Phase transition detection enables adaptive LR scheduling (auto-detect when to reduce LR)

---

## Open Questions

1. **Neighborhood construction**: k-NN vs epsilon-ball vs mutual-kNN. Likely need to benchmark all three on DSM and pick the one that produces the most informative persistence diagrams.

2. **Sequence length scaling**: For seq_len > 512, k-NN distances become less meaningful in high dimensions (curse of dimensionality). May need to project to lower dimension before building CellComplex (PCA or random projection).

3. **Sheaf restriction map training**: Cross-layer sheaf restriction maps are learned parameters. In observer mode, how to train them? Options: (a) fit post-hoc on captured hidden states via reconstruction loss, (b) use SVD alignment as initialization and skip learning, (c) use the coboundary operator from the chain complex as a proxy.

4. **Cost**: Full topological profiling on every step is too expensive. Profile every N steps (N=100?), or profile a single batch per epoch. Active mode feedback needs to be cheap enough for every step -- likely only the 6 aggregate features, not full profiles.

5. **Portability**: Observer mode should work on any `nn.Module` with named layers. Hook registration needs to handle different transformer implementations (HuggingFace, fairseq, custom). The `TransformerHookManager` should auto-detect common patterns (`model.layers`, `model.transformer.h`, `model.encoder.layer`).

# Computation Graph Topology: Design Document

> **Status**: Future development — to be implemented after DSM curriculum training concludes.

**Goal**: Apply the topological analysis toolkit (Hodge decomposition, spectral gap, sheaf diffusion) to PyTorch's dynamic computation graph during LLM training, enabling training diagnostics, architecture optimization, and inference-time output refinement.

**Approach**: Phased rollout (A → C). Phase A captures computation graphs and builds diagnostics. Phase B uses diagnostics for architecture optimization. Phase C feeds topology back into inference for output refinement.

**Key insight**: The computation graph of a neural network IS a cell complex. Operations are 0-cells, data flow edges are 1-cells, composite operations (residual blocks, attention heads) are 2-cells. Our entire topological toolkit applies directly.

---

## Section 1: Graph Capture Layer

### ComputationGraphCapture

A context manager that hooks into PyTorch's autograd to build a `CellComplex` from the dynamic computation graph.

```python
with ComputationGraphCapture(model) as capture:
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

cc = capture.to_cell_complex()  # CellComplex with gradient/activation signals
```

### Cell Structure

| Cell Dim | What | Edge Signal (forward) | Edge Signal (backward) |
|----------|------|-----------------------|------------------------|
| 0-cell | Operation (Linear, LayerNorm, Attention, GNN layer, etc.) | Activation magnitude | Gradient magnitude |
| 1-cell | Data flow edge (op A output → op B input) | Activation tensor norm | Gradient tensor norm |
| 2-cell | Composite operation (residual block, attention head, executive loop iteration) | Aggregate activation | Aggregate gradient |

### Hook Strategy

- **Forward hooks**: `register_forward_hook` on every `nn.Module`. Records activation norms as 1-cell signals. Builds 0-cell (operation) and 1-cell (data flow) structure.
- **Backward hooks**: `register_full_backward_hook` on every `nn.Module`. Records gradient norms as backward 1-cell signals.
- **2-cell detection**: Automatically identifies composite operations by module hierarchy. `model.executive_loop.gnn` → all operations inside form a 2-cell. Residual connections detected by shared tensor identity.

### Dynamic Unrolling

Our architecture has variable-length computation:
- Executive loop runs 1-5 iterations (convergence-dependent)
- Each iteration is a separate subgraph in the cell complex
- The CellComplex grows per iteration, with inter-iteration edges connecting corresponding operations
- This naturally captures how the executive loop's topology changes per sample

### Boundary Operators

- **B1** (edges → nodes): Standard incidence matrix from data flow graph.
- **B2** (faces → edges): Boundary of composite operations. A residual block's 2-cell has boundary = {input edge, skip edge, output edge} with signs from data flow direction.
- **B1 @ B2 = 0**: Enforced by construction — composite operations are bounded by their input/output edges, which connect to the same operations that the composite contains.

---

## Section 2: Topological Analysis Pipeline

Three analysis passes on each captured computation graph cell complex.

### Pass 1: Hodge Decomposition of Gradient Flow

Apply `hodge_decomposition(cc, backward_signal)` to the gradient magnitude signal on 1-cells:

- **Gradient component** (exact): Normal loss-driven gradient flow from output → parameters. Measures what fraction of gradient energy follows source-to-sink paths.
- **Curl component** (co-exact): Gradient circulation in loops — skip connections, residual paths, attention re-entry. High curl energy means gradients are "spinning" through cycles rather than flowing to parameters.
- **Harmonic component**: Gradient energy trapped in topological holes — dead subnetworks where gradients neither flow through nor circulate. High harmonic energy = wasted capacity.

**Diagnostic outputs**: `gradient_energy_ratio`, `curl_energy_ratio`, `harmonic_energy_ratio` per training step. A healthy training run: high gradient ratio, low curl, near-zero harmonic.

**Novel diagnostic**: Curl energy in gradient flow is a direct signal for vanishing/exploding gradient diagnosis that no existing tool captures. Standard gradient norms measure magnitude; Hodge decomposition measures *flow structure*.

### Pass 2: Spectral Gap Analysis

Compute Hodge Laplacian L1 of the computation graph, extract spectral gap:

- **Large spectral gap** → information flows efficiently (well-connected computation graph)
- **Small spectral gap** → bottleneck in information flow (gradient starvation risk)
- **Per-layer spectral gaps** → identifies which layers are bottlenecks vs. well-connected

Directly relates to our benchmark finding: spectral gap classification accuracy correlates with architecture information propagation quality.

### Pass 3: Sheaf Analysis of Activation Flow

Apply `SheafWaveDynamics` (amortized MLP variant) to forward-pass activation signals:

- Learn per-edge restriction maps describing how each operation transforms input → output feature spaces
- Restriction maps reveal feature routing patterns: which dimensions are preserved, mixed, or discarded
- Compare restriction map alignment with B1/B2 — high alignment means information routing matches topological structure

**Reveals**: Redundant attention heads, unused skip connections, DSM-GNN bridge routing effectiveness.

### Output

```python
@dataclass
class TopologicalDiagnostics:
    # Hodge decomposition of gradient flow
    gradient_energy_ratio: float
    curl_energy_ratio: float
    harmonic_energy_ratio: float

    # Spectral analysis
    spectral_gap: float
    per_layer_spectral_gaps: dict[str, float]

    # Sheaf analysis
    restriction_map_rank: float
    b1_alignment: float
    b2_alignment: float

    # Metadata
    num_operations: int      # 0-cells
    num_data_flows: int      # 1-cells
    num_composites: int      # 2-cells
    executive_iterations: int
```

Logged alongside standard training metrics (loss, accuracy, learning rate).

---

## Section 3: Architecture Optimization Engine

Turns topological diagnostics into actionable architecture recommendations.

### Mode 1: Training-Time Alerts (Passive)

Threshold-based alerts during training, zero intervention:

| Condition | Alert | Suggested Action |
|-----------|-------|------------------|
| `curl_energy_ratio > 0.4` | Gradient circulation | Reduce skip connection density or add gradient clipping on residual paths |
| `harmonic_energy_ratio > 0.15` | Dead subnetwork | Prune inactive layers between X-Y |
| `spectral_gap < 0.01` | Information bottleneck | Widen layer Z or add skip connection |
| `filter_weight_entropy < 0.3` | Ensemble collapse | Increase filter diversity penalty |

Logged alongside standard training metrics. Researcher monitors like loss curves.

### Mode 2: Hyperparameter Suggestions (Semi-Active)

After N training steps, aggregate diagnostics and suggest tuning:

- Rising curl energy → increase `wave_damping` or reduce `num_iterations` in executive loop
- Spectral gap varies wildly across layers → per-layer learning rates
- Low-rank sheaf restriction maps → reduce `sheaf_rank` to save compute
- Harmonic energy concentrates in specific 2-cells → prune those composite operations

Output: Ranked suggestion list after each epoch with expected impact estimates.

### Mode 3: Architecture Search Guidance (Active)

Use topological diagnostics as NAS objective:

- Hodge energy ratios define "architecture health score"
- Search over: GNN layers, attention heads, spectral filter types, executive loop iterations, DSM integration points
- Objective: maximize accuracy while minimizing curl + harmonic energy
- Topological health is cheaper to compute than full training accuracy (one forward+backward pass with hooks vs. full training run) — serves as fast proxy objective

---

## Section 4: Inference-Time Topological Refinement

The most novel contribution. Uses computation graph topology at inference time to refine model outputs — the topology of *how* the model processes a specific input carries information about output quality.

### Mechanism 1: Topological Confidence Calibration

After producing an output, compute Hodge decomposition of the computation graph's gradient flow for that input:

- **High gradient ratio** → clean source-to-sink reasoning → high confidence
- **High curl ratio** → reasoning looped through cycles → lower confidence, flag for review
- **High harmonic ratio** → parts of model didn't participate → may be missing information

This gives **topology-grounded confidence scores** fundamentally different from softmax entropy. Softmax measures "how sure the classifier is." Hodge ratios measure "how cleanly the model reasoned." A model can be confidently wrong (high softmax, high curl) — topology catches that.

### Mechanism 2: Iterative Topological Feedback

Feed computation graph topology back into the executive loop:

1. After iteration K, compute spectral gap + Hodge ratios of computation graph so far
2. Inject as additional inputs to ControlHead at iteration K+1
3. Executive learns to adjust control signals based on its own computation graph topology
4. If high curl detected (reasoning going in circles) → increase wave damping to break cycle
5. If low spectral gap (information bottleneck) → widen spatial focus

This is the model reasoning about its own reasoning process — meta-cognition via topology.

### Mechanism 3: Topological Output Correction

Train a lightweight correction head:

```
correction_input = [original_output, topological_features]
refined_output = correction_head(correction_input)
```

- Learns which topological signatures correspond to specific failure modes
- Sheaf restriction map misalignment with B1 → structurally inconsistent answer → re-weight
- Topology-informed ensemble of the model with itself: same parameters, output refined by how those parameters processed this specific input

### Why This Is Novel

Standard transformers treat the computation graph as fixed infrastructure. Our architecture already has dynamic computation (executive loop iterations, wave filter selection, DSM gating), but doesn't observe the *topology* of that computation. Adding topological observation creates a feedback loop:

1. Process an input
2. Observe how it processed it (via topology)
3. Use that observation to improve the processing
4. Repeat

Closest analogues (adaptive computation time, universal transformers) only adapt depth, not full topological structure.

---

## Phased Implementation

| Phase | Name | What | Depends On |
|-------|------|------|------------|
| A | Graph Capture + Diagnostics | ComputationGraphCapture + TopologicalDiagnostics | DSM training complete |
| B | Architecture Optimization | Modes 1-3 of optimization engine | Phase A validated |
| C | Meta-Architecture | Inference-time refinement (Mechanisms 1-3) | Phase B insights |

Each phase is independently valuable. Phase A alone provides a novel training diagnostic tool. Phase B provides actionable architecture guidance. Phase C is the full vision of topology-aware self-refining models.

---

## Connection to Existing Findings

| Finding | Application Here |
|---------|-----------------|
| Hodge inverse scaling (39% → 71%) | Larger computation graphs (deeper models) should produce cleaner Hodge decomposition — predicts that topology-grounded confidence improves with model scale |
| Sheaf 104% size retention | Sheaf analysis of computation graphs should generalize across model sizes — train diagnostic model on small architecture, deploy on large |
| Curl preservation bug | Proved that gradient circulation is real and measurable in our architecture — the same phenomenon will appear in computation graph analysis |
| Spectral filter specialization | Different filters excel at different tasks — computation graph spectral analysis should reveal which filters are active for which input types |
| Executive convergence detection | Already uses harmonic energy — extending to full Hodge decomposition of the computation graph is a natural generalization |

# Inverse Size Scaling in Learned Hodge Classification — Multi-Scale Laplacian + Sheaf Diffusion

## Research Objectives

1. **Validate inverse scaling on v11 architecture** — Reproduce the Phase 4b result (39%→71% hodge_class accuracy as graph size grows from n=20 to n=80) on the v11 architecture WITHOUT explicit hodge features in the classifier. This is the clean test that inverse scaling is a property of the spectral pipeline, not an artifact of cheating.

2. **Fix curl blindness via multi-scale L0/L1/L2 filtering** — Phase 4b showed 0% curl accuracy across ALL conditions, ALL filters, ALL sizes. Root cause: spectral filters operate on L0 (node Laplacian) only, bypassing L1 (edge) and L2 (triangle) Laplacians where curl structure lives. Add parallel processing on all three Hodge Laplacians.

3. **Re-benchmark sheaf diffusion post-stability-fix** — Phase 4b sheaf results (104% average retention, 81% peak hodge) were measured before Phase 4d stability fixes. NaN bail-outs contaminated results. Get clean numbers and test sheaf + multi-scale interaction.

4. **Characterize scaling limits** — Extend OOD evaluation to n=320. Find saturation point. Measure per-topology and per-class behavior. Produce publication-ready scaling curves with error bars.

5. **Metacognition integration** — Use scaling curve data to calibrate a `spectral_confidence` signal in the ControlHead for adaptive reasoning.

## Background

### The Discovery (Phase 4b, February 2026)

During Phase 4b benchmarking on Lambda A100/H100, hodge_class (3-class: gradient/curl/harmonic) showed accuracy that **increases** with out-of-distribution graph size:

| Filter | n=20 (ID) | n=40 (2x) | n=80 (4x) |
|--------|-----------|-----------|-----------|
| Chebyshev | 38.4% | 49.8% | 71.0% |
| Wave cosine | 38.4% | 50.4% | 70.2% |
| Sheaf | 38.0% | 48.6% | 70.0% |
| Bandpass | 39.4% | 50.6% | 71.2% |
| Nowave | 40.6% | 48.6% | **24.8%** (collapsed) |

Every other task (BFS, diverse, spectral_gap, propagation_delay) shows normal degradation with size. Hodge is the sole exception. Nowave's collapse to 24.8% proves the spectral pathway is essential.

### Why It Happens (Hypotheses)

1. **Spectral resolution**: L1 eigenvalues cluster into gradient/curl/harmonic groups. n=20 has ~40 edges → 40 eigenvalues (poorly separated). n=80 has ~200 edges → 200 eigenvalues (clear separation).

2. **Topological stability**: Homology groups stabilize as complexes grow. n=20 graphs have 0-3 independent cycles (noisy). n=80 graphs have predictable cycle structure.

3. **Small graph degeneracy**: With n=20, B2 is often sparse/zero (few triangles), making curl subspace near-empty. At n=80 all three Hodge classes are well-represented.

4. **Learned spectral transfer**: Spectral filters learn eigenvalue-dependent transformations on normalized spectra. More eigenvalues → better resolution without retraining.

### The Curl Problem

0% curl accuracy across ALL configurations. The model classifies everything as gradient or harmonic. Root cause: GNN/TAT operates on L0, completely missing L1/L2 where curl lives. Curl is in the image of B2 (triangles→edges), invisible to node-level spectral processing.

### Sheaf Results (Phase 4b)

Sheaf is the only filter averaging >100% retention across all tasks:

| Metric | Sheaf | Chebyshev | Wave Cosine | Heat | Nowave |
|--------|-------|-----------|-------------|------|--------|
| BFS retention (n=80/n=20) | **96%** | 85% | 74% | 43% | 85% |
| Diverse retention | **82%** | 66% | 56% | 52% | 73% |
| Hodge retention | 184% | 185% | 183% | 95% | 61% |
| **Average retention** | **104%** | 88% | 92% | 60% | 71% |

Peak: 81% hodge_class (sheaf trained n=40, tested n=80). Wave cosine is the most consistent (size-invariant features). Sheaf achieves highest peaks but less stable at extremes.

### Confounding Factor

Phase 4b models used the old classifier path with explicit `_compute_hodge_features()` — 3 hodge decomposition norms fed directly to the classifier. This "cheating" inflated absolute accuracy but did NOT explain the scaling trend (the cheating failed equally at all sizes — evidenced by 0% curl despite the explicit norms being computed correctly).

v11 architecture uses AttentionReadout (no explicit hodge features) — the clean test.

### Connection: Sheaf and Multi-Scale Laplacian

Sheaf diffusion and Hodge decomposition are linked through chain complex structure:
- Standard Hodge Laplacians (L0, L1, L2) assume uniform edge weights
- Sheaf Laplacian allows boundary operators to be learned per-edge
- A well-trained sheaf recovers restriction maps aligned with topological structure in B1/B2
- Sheaf is essentially learning a data-adaptive Hodge decomposition

This raises the question: are sheaf and multi-scale L0/L1/L2 complementary (different mechanisms) or redundant (same mechanism, different parameterizations)?

## Architecture: Multi-Scale Laplacian Filter

### Current State

Spectral filter ensemble processes only L0 (node Laplacian):
```
node_features → SpectralFilterEnsemble(L0) → filtered_node_features
```

### Proposed: MultiScaleLaplacianFilter

Parallel processing on all three Hodge Laplacians:
```
CellComplex
  ├── L0 path (existing): node features → SpectralFilter(L0) → node output
  ├── L1 path (NEW):      edge features → SpectralFilter(L1) → edge output
  └── L2 path (NEW):      triangle features → SpectralFilter(L2) → triangle output
                                    ↓
                           Boundary projection → node-level features
                                    ↓
                           Learnable gated fusion → combined output
```

### Design Decisions

1. **Reuse existing SpectralFilterEnsemble** — operates on `(eigenvalues, eigenvectors, signal)`. Eigenvalue-agnostic. Feed it L1/L2 eigendecompositions unchanged.

2. **Projection to nodes** — L1 output lives on edges, L2 on triangles. Project back:
   - `node_from_edges = B1 @ edge_output` (boundary operator, maps edges→nodes)
   - `node_from_triangles = I_nt @ triangle_output` (node-triangle incidence matrix)

   **NOTE**: Cannot use `B1 @ B2` for triangle→node projection because `B1 @ B2 = 0` by the chain complex property (∂² = 0). Instead, compute a node-triangle incidence matrix `I_nt` where `I_nt[node, triangle] = 1` if node is a vertex of that triangle. This is NOT a boundary operator — it's a membership/containment relation.

3. **Learnable gated fusion** — Small gating network:
   ```
   output = gate_0 * L0_out + gate_1 * proj(L1_out) + gate_2 * proj(L2_out)
   ```
   Gates are sigmoid-activated learned scalars, allowing the model to weight each scale.

4. **Drop-in replacement** — `MultiScaleLaplacianFilter` has the same interface as existing spectral filter. Controlled by config flag `use_multiscale_laplacian: true`.

### New File

`src/spectral/multiscale_filter.py`:
- `MultiScaleLaplacianFilter(nn.Module)` — wraps existing ensemble with L1/L2 paths
- Computes L1, L2 eigendecompositions (cached on CellComplex)
- Handles degenerate cases (no 2-cells → skip L2, no edges → skip L1)

### Modified Files

- `src/wave/wave_propagation.py` — option to use `MultiScaleLaplacianFilter`
- `config/` — experiment configs with `use_multiscale_laplacian` flag
- No changes to existing `SpectralFilter` or `SpectralFilterEnsemble`

## Experiment Suite

### Protocol

- **Training**: Uniform random n sampled from [16, 32] per sample (matching Phase 4b mixed-size setup)
- **Evaluation**: n=20, 40, 80, 120, 160, 200, 320
- **Task**: hodge_class (3-class: gradient/curl/harmonic)
- **Controls**: BFS + diverse at same sizes (expect degradation)
- **Seeds**: 5 for main experiments, 3 for ablations and per-topology
- **Epochs**: 30 max, patience 15
- **Optimizer**: AdamW, lr=0.0003, weight_decay=0.01, cosine annealing
- **Metrics**: balanced accuracy (overall + per-class), scaling retention ratio
- **Statistical test**: One-sided paired permutation test (or bootstrap 95% CI on slope) comparing accuracy at size n vs 2n across seeds. "Inverse scaling" claim requires p < 0.05 for at least 3 consecutive size doublings.
- **Sheaf stability criterion**: If NaN bail-out rate exceeds 5% at any test size, that data point is flagged. If >20% at any size, sheaf experiment is paused for stability investigation before continuing.

### Experiments

| # | Name | Config | Seeds | Key Question |
|---|------|--------|-------|-------------|
| 0a | Class balance audit | — | — | What is gradient/curl/harmonic distribution at each test size? Is curl underrepresented at small n? |
| 0b | Exact decomposition baseline | Non-learned: exact hodge_decomposition + argmax | — | Does exact decomposition achieve 100% at all sizes? (no GPU needed) |
| 1 | Baseline | v11 ensemble (cheby+wave+heat), L0 only | 5 | Does 39%→71% reproduce without explicit hodge features? |
| 2a | Multi-scale | same ensemble, L0+L1+L2 | 5 | Does L1 fix curl? How does scaling curve change? |
| 2b | Sheaf only | sheaf filter alone (post-stability-fix) | 5 | Clean sheaf numbers — does 104% retention hold? |
| 2c | Sheaf in ensemble | cheby+wave+heat+sheaf, L0 | 5 | Does executive learn selective sheaf gating? |
| 2d | Sheaf + multi-scale | sheaf, L0+L1+L2 | 5 | Complementary or redundant? |
| 3a | Ablation: no-L1 | L0+L2 only | 3 | Isolate L1's curl contribution |
| 3b | Ablation: no-L2 | L0+L1 only | 3 | Isolate L2's contribution |
| 3c | Ablation: nowave | no spectral filtering | 3 | Confirm spectral pathway essential |
| 3d | Ablation: no-TAT | GNN only, no transformer | 3 | Isolate TAT's role |
| 3e | Ablation: no-wave-dynamics | filter only, no neural ODE | 3 | Isolate wave propagation |
| 4 | Per-topology | best model from 1/2a-d | 3 | ER, BA, WS, lattice, SBM separately |

### Sheaf-Specific Analysis (Experiment 2b/2c)

Beyond accuracy metrics:
- **NaN bail-out rate**: How often does the stability guard trigger? At which graph sizes?
- **Restriction map analysis**: Extract learned maps, compute alignment with B1/B2
- **Executive filter weights**: In 2c, track per-filter weight evolution — does sheaf weight increase with graph size?
- **Diffusion diagnostics**: Compare diffusion_time, wave_damping, freq_gate entropy across sizes

### Deliverables

1. **Scaling curves** — accuracy vs graph size with error bars, all configs
2. **Per-class breakdown** — gradient/curl/harmonic at each size (curl recovery is the headline)
3. **Sheaf retention table** — clean post-fix numbers for all tasks
4. **Restriction map visualization** — t-SNE colored by edge topological role
5. **Executive weight evolution** — sheaf gating over training epochs
6. **Per-topology heatmap** — accuracy × graph_size × topology_type
7. **Spectral confidence calibration** — spectral_gap vs accuracy curve for metacognition

## File Structure

```
scripts/experiments/inverse_scaling/
  train.py                  # Train hodge_class, configurable filter/scale/seeds
  evaluate_ood.py           # OOD evaluation at 7 sizes
  audit_class_balance.py    # Exp 0a: class distribution at each test size
  exact_baseline.py         # Exp 0b: exact hodge decomposition + argmax
  run_ablations.py          # Ablation suite
  run_all.py                # Full orchestrator
  analyze_sheaf.py          # Restriction map extraction + B1/B2 alignment
  plot_scaling_curves.py    # Generate publication figures (scaling curves, heatmaps)
  analyze_results.py        # Statistical tests, per-class breakdown, confidence calibration
  configs/
    baseline.yaml           # Exp 1: ensemble, L0 only
    multiscale.yaml         # Exp 2a: ensemble, L0+L1+L2
    sheaf_only.yaml         # Exp 2b: sheaf alone
    sheaf_ensemble.yaml     # Exp 2c: ensemble + sheaf
    sheaf_multiscale.yaml   # Exp 2d: sheaf + L0/L1/L2
    ablation_no_l1.yaml     # Exp 3a
    ablation_no_l2.yaml     # Exp 3b
    ablation_nowave.yaml    # Exp 3c
    ablation_no_tat.yaml    # Exp 3d
    ablation_no_ode.yaml    # Exp 3e
src/spectral/
  multiscale_filter.py      # NEW: MultiScaleLaplacianFilter
docs/experiments/inverse-scaling/
  results.md                # Live results table
```

## Metacognition Integration

### Stage A: Spectral Confidence Signal

Add `spectral_confidence` field to `ControlSignal` dataclass:

```python
spectral_confidence = f(spectral_gap, num_eigenvalues, L1_gap)
```

- Computed from Hodge Laplacian eigenvalue gaps during forward pass
- Calibrated using the empirical scaling curve (accuracy vs spectral gap at each size)
- The metacognition controller uses this alongside existing `uncertainty` for decision-making
- **No architecture changes** — just a new field in existing ControlSignal

### Stage B: Adaptive Graph Expansion (future)

When `spectral_confidence` is low (small subgraph, poor resolution):
```python
if control.spectral_confidence < threshold:
    expanded_cc = expand_neighborhood(cc, hops=1)
    # re-run forward on larger context
```

Deferred until Stage A validates the signal. Experiment outputs must include per-sample `spectral_gap vs correct/incorrect` data to support Stage B threshold calibration.

## Compute Budget

| Experiment | Runs | Est. Hours |
|-----------|------|-----------|
| Exp 0a (class balance audit) | CPU only | 0 |
| Exp 0b (exact decomposition baseline) | CPU only | 0 |
| Exp 1 (baseline) | 5 seeds × 30 ep | 3 |
| Exp 2a (multi-scale) | 5 seeds × 30 ep | 4 |
| Exp 2b (sheaf only) | 5 seeds × 30 ep | 4 |
| Exp 2c (sheaf ensemble) | 5 seeds × 30 ep | 4 |
| Exp 2d (sheaf+multi-scale) | 5 seeds × 30 ep | 5 |
| Exp 3a-3e (ablations) | 15 runs × 30 ep | 4 |
| Exp 4 (per-topology) | 5 topos × 3 seeds × 7 sizes | 3 |
| OOD evaluation (all) | ~250 eval runs | 1 |
| **Total** | | **~28 GPU hours** |

## Timeline

1. **Now → v11 finishes (~3-4 days)**: Build MultiScaleLaplacianFilter, experiment scripts, unit tests. All local/CPU.
2. **GPU available**: Run full experiment suite (~26 hours on RTX 4090)
3. **After results**: Spectral confidence calibration, metacognition Stage A integration
4. **Paper prep**: Write up findings for workshop/conference submission

## Publication Target

**Core claim**: "A hierarchical GNN with multi-scale spectral wave dynamics learns size-invariant representations of Hodge-theoretic flow structure, with classification accuracy that improves monotonically with graph size on out-of-distribution test sets — including curl detection via L1 Laplacian processing."

**Venues**: LoG (Learning on Graphs), ICML/NeurIPS topological DL workshop, ICLR (if extended with real-world validation)

**Novel contributions**:
1. First demonstration of inverse size scaling for learned Hodge classification
2. Multi-scale Laplacian filtering across L0/L1/L2 for complete Hodge decomposition
3. Sheaf diffusion as a mechanism for size-invariant graph representations
4. Empirical characterization of when/why inverse scaling occurs (spectral resolution hypothesis)

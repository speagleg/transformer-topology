# Potential Spectral Computation Directions

Captured during benchmark run (2026-02-13). To be revisited after full results are in.

## Current Architecture

- **HeatDiffusion**: analytic `h(λ) = exp(-λt)`, low-pass smoother, no learnable params
- **WavePropagation**: neural ODE `df/dt = -L@f - damping*f + correction(f,t)`, first-order damped
- **WaveDynamics**: learned per-node gate blends heat and wave outputs
- Only operates on L0 (node Laplacian)

## Spectral Filter Extensions

### Band-pass filters
`h(λ) = exp(-(λ-λ₀)²/σ²)` — learnable center frequency λ₀ and bandwidth σ. Executive could learn to focus on relevant spectral bands per-task. Connects naturally to existing frequency_gate in ControlSignal.

### Spectral graph wavelets (Hammond et al. 2011)
Multi-scale `h(λ) = g(s*λ)` at multiple scales s. Gives localized multi-resolution features. Like a spectral version of multi-hop aggregation.

### Chebyshev polynomial filters
`h(λ) = Σ θ_k T_k(λ)` — learnable polynomial over spectrum. ChebNet-style. More flexible than single-kernel filters, avoids full eigendecomposition.

## Dynamics Extensions

### True wave equation (second-order)
`d²f/dt² = -L@f` → `h(λ) = cos(√λ * t)`. Oscillatory, preserves frequency content instead of damping. Current "wave" is actually damped first-order (diffusion-like). True wave gives interference patterns.

### Schrodinger equation
`i*df/dt = L@f` → `h(λ) = exp(-iλt)`. Complex-valued, norm-preserving, interference patterns. Used in quantum walk GNNs.

## Structural Extensions

### Higher-order Laplacian dynamics
Run heat/wave on L1 (edge Laplacian) or L2 (face Laplacian) instead of just L0. We already compute these Hodge Laplacians — just not using them for dynamics. Directly relevant to edge-flow and topological tasks.

### Magnetic Laplacian
Complex-valued Laplacian encoding edge directionality: `L_q[u,v] = exp(i*q*θ_uv)`. Useful for directed graphs and flow problems. Relevant to Hodge/edge-flow tasks.

### Sheaf diffusion (Bodnar et al. 2022)
Each edge gets a linear restriction map instead of scalar weight. Sheaf Laplacian generalizes graph Laplacian. Can model heterogeneous node relationships.

## Immediate Priorities (post-benchmark)

1. **Wave strength gate** — scalar bypass so executive can learn to skip waves per-task (~10 lines)
2. **Band-pass filters** — replace binary frequency_gate with learned spectral window
3. **L1 dynamics** — run existing heat/wave on edge Laplacian for topological tasks

## Benchmark Evidence Needed

- Waves help: diverse, bfs (global structure tasks)
- Waves hurt: prop_delay (edge-weight precision tasks)
- Waves neutral: spectral_gap, hodge_class (TBD)
- Key question: can the executive learn WHEN to use waves, or do we need explicit gating?
